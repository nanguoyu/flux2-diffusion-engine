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
        // The streaming path is text-to-image only. Fail loudly instead of silently dropping the
        // reference and producing a T2I image (the facade's I2I encodes the reference as context).
        guard reference == nil else {
            throw EngineError.invalidRequest("image-to-image is not supported by the streaming FLUX path")
        }
        let (vh, vw) = LatentUtils.validateDimensions(height: size.height, width: size.width)
        validHeight = vh; validWidth = vw
        // T2I: pure-noise patchified latent, packed to the transformer's sequence form. NO input
        // normalize (denormalize-only happens at decode) — normalizing pure noise would corrupt it.
        let patchified = LatentUtils.generatePatchifiedLatents(height: vh, width: vw, seed: seed)
        let packed = LatentUtils.packPatchifiedToSequence(patchified)
        // Both position-id sets, generated together exactly as the resident T2I path does.
        let (textIds, imageIds, _) = LatentUtils.combinePositionIDs(textLength: textLength, height: vh, width: vw)
        txtIds = textIds
        imgIds = imageIds
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
                             doubleShell: doubleShell, singleShell: singleShell)
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

    public func releaseCachedResources() {
        encoder = nil
        vae = nil
        MLX.GPU.clearCache()
    }
}
