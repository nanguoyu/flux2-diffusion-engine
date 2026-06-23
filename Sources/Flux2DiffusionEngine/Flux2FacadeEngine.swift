import CoreGraphics
import DiffusionCore
import Flux2Core

/// A `DiffusionEngine` facade over flux-2-swift-mlx's monolithic `Flux2Pipeline`.
///
/// FLUX.2 owns its own denoise loop, scheduler, and weight loading, so this facade does NOT
/// use the block-streaming path or a `WeightSource` — it maps a `GenerationRequest` to
/// `Flux2Pipeline.generateTextToImage` and forwards progress. macOS-only (the package is
/// macOS-15+ and Metal-backed).
public actor Flux2FacadeEngine: DiffusionEngine {
    private var pipeline: Flux2Pipeline?

    public init() {}

    public static func capabilities(for model: DiffusionModel, variant: ModelVariant,
                                    on device: DeviceTier) -> EngineCapabilities {
        if device.isPhone {
            return EngineCapabilities(runnable: false, residency: .unsupported,
                                      estimatedPeakBytes: variant.approximateBytes, note: "macOS only")
        }
        // FLUX runs as a whole pipeline (its own memory management); report it resident.
        return EngineCapabilities(runnable: true, residency: .resident,
                                  estimatedPeakBytes: variant.approximateBytes, note: "Runs on Mac")
    }

    /// `source` is intentionally ignored: `Flux2Pipeline` downloads and loads its own weights
    /// from HuggingFace. The catalog model/variant select the FLUX model + quantization.
    public func load(_ model: DiffusionModel, variant: ModelVariant, source: WeightSource,
                     progress: @Sendable @escaping (Double) -> Void) async throws {
        let fluxModel: Flux2Model = .klein4B   // the catalog currently ships Klein 4B
        let quantization: Flux2QuantizationConfig = (variant.precision == .q4) ? .memoryEfficient : .balanced
        let pipeline = Flux2Pipeline(model: fluxModel, quantization: quantization)
        try await pipeline.loadModels(progressCallback: { fraction, _ in progress(fraction) })
        self.pipeline = pipeline
    }

    public func generate(_ request: GenerationRequest,
                         progress: @Sendable @escaping (GenerationProgress) -> Void) async throws -> CGImage {
        guard let pipeline else { throw EngineError.notLoaded }
        progress(.preparing)
        let image = try await pipeline.generateTextToImage(
            prompt: request.prompt,
            height: request.size.height,
            width: request.size.width,
            steps: request.steps,
            guidance: request.guidance,
            seed: request.seed,
            onProgress: { current, total in
                progress(.denoising(step: current, total: total, preview: nil))
            })
        progress(.finished(image))
        return image
    }

    public func unload() async { pipeline = nil }
}
