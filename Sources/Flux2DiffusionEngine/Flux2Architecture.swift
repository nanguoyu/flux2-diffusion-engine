@preconcurrency import MLX
import CoreGraphics
import DiffusionCore
import Flux2Core

/// FLUX.2 Klein 4B as a core-engine `DiffusionArchitecture`, so the generic block-streaming engine
/// (`MLXDiffusionEngine`) can run it 1024-on-iPhone by streaming the transformer one block at a time.
///
/// The novel streaming machinery (decomposition, per-block quantized load, the denoiser) is verified
/// offline. encode / initialLatent / decode reuse the SAME public flux-mlx units the resident
/// `Flux2Pipeline` uses (`KleinTextEncoder`, `LatentUtils`, the VAE via `Flux2StreamingSupport`) and
/// in the same order, so the streamed path should match the resident path — confirmed by the on-device
/// 512 parity gate (the streamed engine vs the resident facade, same seed) before 1024 is enabled.
///
/// Per-run state (`txtIds`/`imgIds`/`validHeight`/`validWidth`) is set across `encode`+`initialLatent`
/// and read by `makeDenoiser`+`decode`. The engine drives one request at a time, so this is safe.
public final class Flux2Architecture: DiffusionArchitecture, @unchecked Sendable {

    public static let spec = ArchitectureSpec(
        family: .flux2, latentChannels: 32, defaultSampler: .flowMatchEuler,
        defaultSteps: 4, defaultGuidance: 1.0, vaeScale: 1.0, vaeShift: 0.0,
        samplerShift: 1.0, samplerShiftTerminal: 0.0)

    // Klein 4B transformer dims.
    private let dim = 3072, heads = 24, headDim = 128

    private let vaeVariant: ModelRegistry.VAEVariant
    private var encoder: KleinTextEncoder?
    private var vae: AutoencoderKLFlux2?

    private var textLength = 0
    private var txtIds: MLXArray?
    private var imgIds: MLXArray?
    private var validHeight = 0, validWidth = 0

    // i2i (reference-context) state, set in `initialLatent` and read by `makeDenoiser`. `refLatents`
    // nil ⇒ text-to-image. The output latent the engine steps stays output-only; the reference tokens
    // are appended transiently inside the denoiser's per-step embed, so decode/preview are unaffected.
    private var refLatents: MLXArray?
    private var outputSeqLen: Int?
    /// iPhone i2i reference-token budget: 768² (≈2304 tokens). 512² (≈1024) was lighter but starved the
    /// model of edge detail, so it painted a ~5px gray frame around a flat background; 768² clears that
    /// (down to a ~1px hairline, like the resident facade) while the streamed sequence (output 1024 + ref
    /// 2304 = 3328) still stays UNDER the proven 4096-token T2I-1024 that fits the phone at 3.83GB.
    private static let referenceMaxImageArea = 768 * 768

    public init(vaeVariant: ModelRegistry.VAEVariant = .smallDecoder) {
        self.vaeVariant = vaeVariant
    }

    // MARK: - Sigma schedule (architecture-owned; the core sampler cannot reproduce FLUX's)

    public func sigmas(size: ImageSize, steps: Int, sampler: any Sampler) -> [Float] {
        Flux2Sigmas.schedule(size: size, steps: steps)
    }

    // MARK: - Encode (Qwen3-4B), released before the transformer streams

    public func encode(_ prompt: String, negative: String?, source: WeightSource) async throws -> Conditioning {
        let enc = KleinTextEncoder(variant: .klein4B, quantization: .mlx4bit)
        try await enc.load()
        let embeddings = try enc.encode(prompt, upsample: false)
        self.encoder = enc
        self.textLength = embeddings.shape[1]
        return Conditioning(embeddings: embeddings)
    }

    public func releaseTextEncoder() async {
        // The ~2GB Qwen3-4B weights live in the @MainActor FluxTextEncoders singleton — niling the
        // weightless KleinTextEncoder wrapper frees NOTHING. Hop to the main actor and actually unload
        // it, AWAITED (the hook is async) so the encoder is reclaimed BEFORE the transformer streams.
        // Without this the encoder co-resides with the streamed transformer + decode → jetsam on iPhone.
        let enc = encoder
        encoder = nil
        await MainActor.run { enc?.unload() }
        MLX.GPU.clearCache()
    }

    // MARK: - Initial latent (pure noise, packed) + position ids

    public func initialLatent(size: ImageSize, seed: UInt64, reference: CGImage?, strength: Float,
                              source: WeightSource) throws -> MLXArray {
        let (vh, vw) = LatentUtils.validateDimensions(height: size.height, width: size.width)
        validHeight = vh; validWidth = vw
        // Pure-noise patchified OUTPUT latent, packed to the transformer's sequence form. NO input
        // normalize (denormalize-only happens at decode) — normalizing pure noise would corrupt it.
        // The output denoises from pure noise in BOTH T2I and reference-context i2i (strength is 1.0).
        let patchified = LatentUtils.generatePatchifiedLatents(height: vh, width: vw, seed: seed)
        let packed = LatentUtils.packPatchifiedToSequence(patchified)
        let (textIds, outputImageIds, _) = LatentUtils.combinePositionIDs(textLength: textLength, height: vh, width: vw)
        txtIds = textIds

        guard let reference else {
            // Text-to-image: the image stream is output-only.
            refLatents = nil
            outputSeqLen = nil
            imgIds = outputImageIds
            return packed
        }

        // Image-to-image (reference-context): VAE-encode the reference as conditioning, capped to the
        // iPhone streaming token budget. Its tokens are APPENDED after the output `[output ; ref]`
        // (output FIRST, matching the resident path + the velocity slice), with distinct T-coordinate
        // position-ids so the transformer separates reference from output. The output denoises from
        // pure noise while attending to the reference. The encoder VAE is freed before the transformer
        // streams, so there is no VAE ↔ transformer co-residency on the phone.
        var refVAE: AutoencoderKLFlux2? = try Flux2StreamingSupport.loadVAE(variant: vaeVariant)
        let (refLat, refIds) = Flux2StreamingSupport.encodeReferenceImage(
            reference, maxImageArea: Self.referenceMaxImageArea, vae: refVAE!)
        refVAE = nil               // drop the encoder VAE; refLat is already materialized (eval'd)
        MLX.GPU.clearCache()
        refLatents = refLat
        outputSeqLen = packed.shape[1]
        imgIds = concatenated([outputImageIds, refIds], axis: 0)
        return packed
    }

    // MARK: - Denoiser (0-block resident shell + 25 streamed blocks)

    public func makeDenoiser(source: WeightSource) throws -> any Denoiser {
        guard let composite = source as? Flux2ComponentSource,
              let txSource = composite.subSource(.transformer) else {
            throw EngineError.streamingUnavailable
        }
        guard let imgIds, let txtIds else { throw EngineError.notLoaded }

        let shell = Flux2Transformer2DModel(config: Flux2Weights.shellConfig())
        try Flux2Weights.loadShared(from: txSource, into: shell)

        // Reuse-shell: build + quantize ONE double-block and ONE single-block shell, shared across all
        // blocks of that type (and all steps). Each block swaps only its packed params in, so the ~100
        // per-image quantize() passes collapse to two.
        let doubleShell = Flux2Weights.makeDoubleShell(dim: dim, heads: heads, headDim: headDim)
        let singleShell = Flux2Weights.makeSingleShell(dim: dim, heads: heads, headDim: headDim)

        let holder = Flux2StreamHolder()
        var blocks: [any StreamableBlock] = []
        for i in 0 ..< 5 {
            blocks.append(Flux2StreamableBlock(doubleIndex: i, blockIndexInType: i, shell: doubleShell,
                                               approximateBytes: 120_000_000, holder: holder))
        }
        for j in 0 ..< 20 {
            blocks.append(Flux2StreamableBlock(singleIndex: 5 + j, blockIndexInType: j, shell: singleShell,
                                               approximateBytes: 70_000_000, holder: holder))
        }
        return Flux2Denoiser(shell: shell, holder: holder, blocks: blocks, imgIds: imgIds, txtIds: txtIds,
                             doubleShell: doubleShell, singleShell: singleShell,
                             refLatents: refLatents, outputSeqLen: outputSeqLen)
    }

    // MARK: - Decode (VAE), transformer already freed by the engine's two-phase staging

    public func decode(_ latent: MLXArray, source: WeightSource) async throws -> CGImage {
        if vae == nil { vae = try Flux2StreamingSupport.loadVAE(variant: vaeVariant) }
        guard let vae else { throw EngineError.decodeFailed }

        let patchified = LatentUtils.unpackSequenceToPatchified(latent, height: validHeight, width: validWidth)
        let denorm = LatentUtils.denormalizeLatentsWithBatchNorm(
            patchified, runningMean: vae.batchNormRunningMean, runningVar: vae.batchNormRunningVar)
        let vaeLatent = LatentUtils.unpatchifyLatents(denorm)

        // Untiled decode. The spatial-tiling path (decodeTiled) is numerically broken — clamped edge
        // tiles give a non-uniform overlap, producing a wrong-sized (1152 vs 1024) seamed image — and
        // spatial tiling is inherently seam-prone through the decoder's GroupNorm / conv receptive
        // field anyway. The decoder's dense mid-block attention is query-CHUNKED inside the VAE (exact,
        // bounds the ~1.07GB 1024 score tile to ~67MB), so the full-frame decode is the memory-safe path.
        let decoded = vae.decode(vaeLatent)

        guard let image = Flux2StreamingSupport.imageFromVAEOutput(decoded) else {
            throw EngineError.decodeFailed
        }
        // Free the VAE so it doesn't stay resident into the NEXT run's transformer streaming phase
        // (re-loading the small decoder from local cache is cheap vs holding it through a stream).
        self.vae = nil
        MLX.GPU.clearCache()
        return image
    }

    // MARK: - Cheap latent→RGB preview (no VAE) — shows the image forming during a long 1024 run

    /// Best-fit linear map from the RAW (pre-denormalize) 32-channel latent to [-1,1] RGB, derived by
    /// least-squares against the small-decoder VAE (`flux2-demo --rgbfactors`). 32 channel rows + a
    /// trailing bias row. Fitting on the raw latent (not the denormalized VAE input) means the preview
    /// needs no VAE stats mid-denoise — the affine denormalize is absorbed into these factors.
    private static let latentRGBFactors: [[Float]] = [
        [-0.00082, 0.00038, -0.00121], [-0.00323, -0.00342, -0.00124], [-0.00677, 0.00102, 0.00794],
        [0.10611, 0.10842, 0.10765], [-0.00007, -0.00033, 0.00031], [-0.00238, -0.00074, -0.00145],
        [-0.00648, 0.00867, -0.00536], [0.00095, 0.00021, 0.00066], [-0.10497, -0.10074, -0.09026],
        [-0.00023, -0.00073, -0.00136], [-0.00217, -0.00096, -0.00155], [0.01906, -0.00040, -0.02527],
        [0.00136, -0.00063, 0.00170], [-0.00179, -0.00112, -0.00033], [-0.00039, 0.00103, -0.00130],
        [-0.00887, -0.00471, -0.00084], [-0.00031, -0.00043, -0.00092], [-0.00200, -0.00186, -0.00148],
        [0.00098, 0.00028, 0.00024], [0.00945, 0.00564, 0.00647], [0.00050, 0.00005, 0.00103],
        [0.00037, 0.00027, -0.00086], [-0.00013, -0.00136, -0.00043], [-0.00090, -0.00161, -0.00154],
        [-0.00040, 0.00471, 0.00563], [-0.00124, -0.00106, 0.00044], [0.00089, 0.00121, 0.00262],
        [-0.00203, 0.00023, -0.00559], [0.00169, -0.00053, 0.00278], [-0.00534, -0.00025, 0.00509],
        [0.00159, 0.00022, 0.00149], [0.00388, 0.00182, 0.00012],
        [-0.04801, -0.11393, -0.21683],   // bias (row 33)
    ]

    public func latentPreview(_ latent: MLXArray) -> CGImage? {
        guard validHeight > 0, validWidth > 0 else { return nil }
        // Unpack the packed in-loop latent to the spatial VAE-latent grid [1, 32, H/8, W/8]; NO
        // denormalize, NO VAE — the linear factors approximate the whole decode.
        let patchified = LatentUtils.unpackSequenceToPatchified(latent, height: validHeight, width: validWidth)
        let z = LatentUtils.unpatchifyLatents(patchified)               // [1, 32, h, w]
        let h = z.shape[2], w = z.shape[3]
        let factors = Self.latentRGBFactors
        let matrix = MLXArray(factors.prefix(32).flatMap { $0 }).reshaped([32, 3]).asType(z.dtype)
        let bias = MLXArray(factors[32]).reshaped([3, 1]).asType(z.dtype)
        let zf = z.squeezed(axis: 0).reshaped([32, h * w])             // [32, h·w]
        let rgb = matrix.transposed(1, 0).matmul(zf) + bias            // [3, h·w]
        let hwc = rgb.reshaped([3, h, w]).transposed(1, 2, 0).asType(.float32)   // [h, w, 3]
        return ImageConversion.cgImage(fromHWC: hwc, range: .signed)
    }

    public func releaseCachedResources() {
        encoder = nil
        vae = nil
        // Drop any i2i reference tokens too, so an unloaded engine doesn't keep the ~0.5MB reference
        // latent (and a stale imgIds) resident until the next run re-enters initialLatent.
        refLatents = nil
        outputSeqLen = nil
        MLX.GPU.clearCache()
    }
}
