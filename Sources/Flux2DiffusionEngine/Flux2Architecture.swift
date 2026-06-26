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

    public func releaseTextEncoder() {
        // Drop the strong ref so ARC frees the encoder's arrays (unload() is @MainActor and this hook
        // is nonisolated); clearCache then returns the GPU memory before the transformer streams.
        encoder = nil
        MLX.GPU.clearCache()
    }

    // MARK: - Initial latent (pure noise, packed) + position ids

    public func initialLatent(size: ImageSize, seed: UInt64, reference: CGImage?, strength: Float,
                              source: WeightSource) throws -> MLXArray {
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

        let holder = Flux2StreamHolder()
        var blocks: [any StreamableBlock] = []
        for i in 0 ..< 5 {
            blocks.append(Flux2StreamableBlock(index: i, isDouble: true, blockIndexInType: i,
                                               dim: dim, heads: heads, headDim: headDim,
                                               approximateBytes: 120_000_000, holder: holder))
        }
        for j in 0 ..< 20 {
            blocks.append(Flux2StreamableBlock(index: 5 + j, isDouble: false, blockIndexInType: j,
                                               dim: dim, heads: heads, headDim: headDim,
                                               approximateBytes: 70_000_000, holder: holder))
        }
        return Flux2Denoiser(shell: shell, holder: holder, blocks: blocks, imgIds: imgIds, txtIds: txtIds)
    }

    // MARK: - Decode (VAE), transformer already freed by the engine's two-phase staging

    public func decode(_ latent: MLXArray, source: WeightSource) async throws -> CGImage {
        if vae == nil { vae = try Flux2StreamingSupport.loadVAE(variant: vaeVariant) }
        guard let vae else { throw EngineError.decodeFailed }

        let patchified = LatentUtils.unpackSequenceToPatchified(latent, height: validHeight, width: validWidth)
        let denorm = LatentUtils.denormalizeLatentsWithBatchNorm(
            patchified, runningMean: vae.batchNormRunningMean, runningVar: vae.batchNormRunningVar)
        let vaeLatent = LatentUtils.unpatchifyLatents(denorm)

        let decoded: MLXArray
        #if os(iOS)
        // Tile the dense mid-block-attention spike at >=1024 px (latent >=128); 512 stays untiled.
        if vaeLatent.shape[2] >= 128 || vaeLatent.shape[3] >= 128 {
            decoded = vae.decodeWithTiling(vaeLatent, tiling: .aggressive)
        } else {
            decoded = vae.decode(vaeLatent)
        }
        #else
        decoded = vae.decode(vaeLatent)
        #endif

        guard let image = Flux2StreamingSupport.imageFromVAEOutput(decoded) else {
            throw EngineError.decodeFailed
        }
        return image
    }

    public func releaseCachedResources() {
        encoder = nil
        vae = nil
        MLX.GPU.clearCache()
    }
}
