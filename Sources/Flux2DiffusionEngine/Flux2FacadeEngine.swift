import Foundation
import CoreGraphics
import DiffusionCore
import Flux2Core
// Selective imports: FluxTextEncoders also defines `ModelVariant`, which would clash with the
// catalog's `ModelVariant` used in this engine's signatures — pull in only what we need.
import enum FluxTextEncoders.Qwen3Variant
import class FluxTextEncoders.TextEncoderModelDownloader

/// A `DiffusionEngine` facade over flux-2-swift-mlx's monolithic `Flux2Pipeline`.
///
/// FLUX.2 owns its own denoise loop, scheduler, and weight loading, so this facade does NOT
/// use the block-streaming path or a `WeightSource` — it maps a `GenerationRequest` to
/// `Flux2Pipeline.generateTextToImage` and forwards progress. macOS-only (the package is
/// macOS-15+ and Metal-backed).
public actor Flux2FacadeEngine: DiffusionEngine {
    private var pipeline: Flux2Pipeline?
    private let quantization: Flux2QuantizationConfig

    /// `quantization` trades quality for memory/download size: `.highQuality` (bf16),
    /// `.balanced` (8-bit), or `.memoryEfficient` (4-bit text encoder + int8 transformer).
    public init(quantization: Flux2QuantizationConfig = .memoryEfficient) {
        self.quantization = quantization
    }

    public static func capabilities(for model: DiffusionModel, variant: ModelVariant,
                                    on device: DeviceTier) -> EngineCapabilities {
        if device.isPhone {
            return EngineCapabilities(runnable: false, residency: .unsupported,
                                      estimatedPeakBytes: variant.approximateBytes, note: "macOS only")
        }
        // FLUX runs as a whole resident pipeline. Estimate runtime peak above the on-disk size
        // (weights + activation/working buffers) and gate against the device's memory budget so a
        // low-RAM Mac isn't told a model is runnable that then OOMs at load.
        let estimatedPeak = variant.approximateBytes + variant.approximateBytes / 3
        let fits = estimatedPeak < device.memoryBudgetBytes
        return EngineCapabilities(runnable: fits, residency: fits ? .resident : .unsupported,
                                  estimatedPeakBytes: estimatedPeak,
                                  note: fits ? "Runs on Mac" : "Insufficient memory")
    }

    /// The Qwen3 text-encoder variants Klein 4B can use. `KleinTextEncoder` reuses whichever is
    /// present (preferring 8-bit); a fresh memory-efficient install fetches the 4-bit one.
    private static let kleinEncoderVariants: [Qwen3Variant] = [.qwen3_4B_8bit, .qwen3_4B_4bit]

    private static var encoderDownloaded: Bool {
        kleinEncoderVariants.contains { TextEncoderModelDownloader.isQwen3ModelDownloaded(variant: $0) }
    }

    /// Whether FLUX is fully on disk — transformer + VAE *and* the Qwen3 text encoder. Constructing
    /// a `Flux2Pipeline` downloads nothing, so this is a cheap on-disk check.
    public static func isDownloaded(quantization: Flux2QuantizationConfig = .memoryEfficient) -> Bool {
        Flux2Pipeline(model: .klein4B, quantization: quantization).hasRequiredModels && encoderDownloaded
    }

    /// Total bytes of FLUX weights on disk: model components + the Qwen3 text encoder.
    public static func downloadedBytes() -> Int64 {
        var total = Flux2ModelDownloader.downloadedSize()
        for variant in kleinEncoderVariants {
            if let path = TextEncoderModelDownloader.findQwen3ModelPath(for: variant) {
                total += directorySize(at: path)
            }
        }
        return total
    }

    /// Pre-download everything FLUX needs — transformer, VAE, and the Qwen3 text encoder —
    /// reporting 0...1 progress, without loading anything into memory.
    public static func download(quantization: Flux2QuantizationConfig = .memoryEfficient,
                                progress: @Sendable @escaping (Double) -> Void) async throws {
        let pipeline = Flux2Pipeline(model: .klein4B, quantization: quantization)
        let missing = pipeline.missingModels
        let needsEncoder = !encoderDownloaded
        let steps = missing.count + (needsEncoder ? 1 : 0)
        guard steps > 0 else { progress(1); return }

        var done = 0
        let modelDownloader = Flux2ModelDownloader()
        for component in missing {
            let base = done
            _ = try await modelDownloader.download(component, progress: { fraction, _ in
                progress((Double(base) + fraction) / Double(steps))
            })
            done += 1
        }
        if needsEncoder {
            let base = done
            let encoderDownloader = TextEncoderModelDownloader()
            _ = try await encoderDownloader.downloadQwen3(variant: .qwen3_4B_4bit, progress: { fraction, _ in
                progress((Double(base) + fraction) / Double(steps))
            })
        }
        progress(1)
    }

    /// Remove FLUX's downloaded weights — model components and the Qwen3 text encoder — to free space.
    public static func deleteWeights() throws {
        let dir = ModelRegistry.modelsDirectory
        if FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
        }
        for variant in kleinEncoderVariants {
            if let path = TextEncoderModelDownloader.findQwen3ModelPath(for: variant) {
                try? FileManager.default.removeItem(at: path)
            }
        }
    }

    private static func directorySize(at url: URL) -> Int64 {
        guard let walker = FileManager.default.enumerator(
            at: url, includingPropertiesForKeys: [.totalFileAllocatedSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let fileURL as URL in walker {
            total += Int64((try? fileURL.resourceValues(forKeys: [.totalFileAllocatedSizeKey]))?.totalFileAllocatedSize ?? 0)
        }
        return total
    }

    /// `source` is intentionally ignored: `Flux2Pipeline` downloads and loads its own weights
    /// from HuggingFace. The catalog model/variant select the FLUX model + quantization.
    public func load(_ model: DiffusionModel, variant: ModelVariant, source: WeightSource,
                     progress: @Sendable @escaping (Double) -> Void) async throws {
        let fluxModel: Flux2Model = .klein4B   // the catalog currently ships Klein 4B
        let pipeline = Flux2Pipeline(model: fluxModel, quantization: quantization)
        try await pipeline.loadModels(progressCallback: { fraction, _ in progress(fraction) })
        self.pipeline = pipeline
    }

    public func generate(_ request: GenerationRequest,
                         progress: @Sendable @escaping (GenerationProgress) -> Void) async throws -> CGImage {
        guard let pipeline else { throw EngineError.notLoaded }
        progress(.preparing)
        let image: CGImage
        if let reference = request.referenceImage {
            // image-to-image: FLUX.2 takes the reference image(s) directly.
            image = try await pipeline.generateImageToImage(
                prompt: request.prompt,
                images: [reference],
                height: request.size.height,
                width: request.size.width,
                steps: request.steps,
                guidance: request.guidance,
                seed: request.seed,
                onProgress: { current, total in
                    progress(.denoising(step: current, total: total, preview: nil))
                })
        } else {
            image = try await pipeline.generateTextToImage(
                prompt: request.prompt,
                height: request.size.height,
                width: request.size.width,
                steps: request.steps,
                guidance: request.guidance,
                seed: request.seed,
                onProgress: { current, total in
                    progress(.denoising(step: current, total: total, preview: nil))
                })
        }
        progress(.finished(image))
        return image
    }

    public func unload() async { pipeline = nil }
}
