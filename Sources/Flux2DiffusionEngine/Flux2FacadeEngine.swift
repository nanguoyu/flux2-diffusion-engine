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

    /// Convenience init from the app-facing precision options.
    public init(transformer: FluxTransformerPrecision, encoder: FluxEncoderPrecision) {
        self.quantization = Self.quantization(transformer: transformer, encoder: encoder)
    }

    public static func capabilities(for model: DiffusionModel, variant: ModelVariant,
                                    on device: DeviceTier) -> EngineCapabilities {
        if device.isPhone {
            // iPhone runs the two-phase pipeline: the Qwen3 text encoder is unloaded before the
            // transformer + VAE denoise, and the pre-quantized 4-bit transformer loads with no
            // float16 spike. Peak is the larger of the two phases plus an activation working set,
            // gated against the phone's memory budget (≈ half RAM).
            let working: Int64 = 1_000_000_000
            let encoderPhase = variant.components.textEncoder + working
            let denoisePhase = variant.components.transformer + variant.components.vae + working
            let peak = max(encoderPhase, denoisePhase)
            let fits = peak < device.memoryBudgetBytes
            return EngineCapabilities(runnable: fits, residency: fits ? .twoPhase : .unsupported,
                                      estimatedPeakBytes: peak,
                                      note: fits ? "Two-phase on iPhone" : "Insufficient memory")
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

    /// App-facing precision options for Klein 4B's transformer.
    public enum FluxTransformerPrecision: String, CaseIterable, Sendable, Identifiable {
        case bit16, bit8, bit4
        public var id: String { rawValue }
        public var label: String { switch self { case .bit16: return "16-bit"; case .bit8: return "8-bit"; case .bit4: return "4-bit" } }
        public var note: String {
            switch self {
            case .bit16: return "highest quality · ~8 GB"
            case .bit8: return "balanced · ~4 GB"
            case .bit4: return "lowest memory · ~8 GB, quantizes on load"
            }
        }
        var quant: TransformerQuantization { switch self { case .bit16: return .bf16; case .bit8: return .qint8; case .bit4: return .int4 } }
    }

    /// App-facing precision options for the Qwen3 text encoder.
    public enum FluxEncoderPrecision: String, CaseIterable, Sendable, Identifiable {
        case bit8, bit4
        public var id: String { rawValue }
        public var label: String { switch self { case .bit8: return "8-bit"; case .bit4: return "4-bit" } }
        public var note: String { switch self { case .bit8: return "better prompt fidelity · ~4 GB"; case .bit4: return "smaller · ~2 GB" } }
        var mistral: MistralQuantization { switch self { case .bit8: return .mlx8bit; case .bit4: return .mlx4bit } }
        var qwen3: Qwen3Variant { switch self { case .bit8: return .qwen3_4B_8bit; case .bit4: return .qwen3_4B_4bit } }
    }

    /// Build a flux quantization config from the app-facing precision options.
    public static func quantization(transformer: FluxTransformerPrecision = .bit8,
                                    encoder: FluxEncoderPrecision = .bit8) -> Flux2QuantizationConfig {
        Flux2QuantizationConfig(textEncoder: encoder.mistral, transformer: transformer.quant)
    }

    /// Precision-keyed convenience over `isDownloaded(quantization:)` so callers need not name the
    /// flux config type.
    public static func isDownloaded(transformer: FluxTransformerPrecision, encoder: FluxEncoderPrecision) -> Bool {
        isDownloaded(quantization: quantization(transformer: transformer, encoder: encoder))
    }

    /// Precision-keyed convenience over `download(quantization:progress:)`.
    public static func download(transformer: FluxTransformerPrecision, encoder: FluxEncoderPrecision,
                                progress: @Sendable @escaping (Double) -> Void) async throws {
        try await download(quantization: quantization(transformer: transformer, encoder: encoder), progress: progress)
    }

    /// Build the Klein 4B pipeline, forcing the pre-quantized 4-bit transformer on iPhone (the only
    /// Klein that fits the phone's memory budget; loads with no float16 spike). Mac uses the default
    /// on-the-fly path for every precision, so its behaviour is byte-for-byte unchanged.
    static func makePipeline(quantization: Flux2QuantizationConfig) -> Flux2Pipeline {
        #if os(iOS)
        let override: ModelRegistry.TransformerVariant? = quantization.transformer == .int4 ? .klein4B_4bit : nil
        #else
        let override: ModelRegistry.TransformerVariant? = nil
        #endif
        return Flux2Pipeline(model: .klein4B, quantization: quantization, transformerVariantOverride: override)
    }

    /// The Qwen3 text-encoder variant a given config resolves to (mirrors `KleinTextEncoder`).
    private static func qwen3Variant(for config: Flux2QuantizationConfig) -> Qwen3Variant {
        switch config.textEncoder {
        case .bf16, .mlx8bit: return .qwen3_4B_8bit
        case .mlx6bit, .mlx4bit: return .qwen3_4B_4bit
        }
    }

    /// Whether FLUX is fully on disk for the given precision — transformer + VAE *and* the matching
    /// Qwen3 text encoder. Constructing a `Flux2Pipeline` downloads nothing, so this is cheap.
    public static func isDownloaded(quantization: Flux2QuantizationConfig = .memoryEfficient) -> Bool {
        let pipeline = makePipeline(quantization: quantization)
        return pipeline.hasRequiredModels
            && TextEncoderModelDownloader.isQwen3ModelDownloaded(variant: qwen3Variant(for: quantization))
    }

    /// Total bytes of FLUX weights on disk: model components + any downloaded Qwen3 text encoders.
    public static func downloadedBytes() -> Int64 {
        var total = Flux2ModelDownloader.downloadedSize()
        for variant in [Qwen3Variant.qwen3_4B_8bit, .qwen3_4B_4bit] {
            if let path = TextEncoderModelDownloader.findQwen3ModelPath(for: variant) {
                total += directorySize(at: path)
            }
        }
        return total
    }

    /// Pre-download everything FLUX needs for the given precision — transformer, VAE, and the
    /// matching Qwen3 text encoder — reporting 0...1 progress, without loading into memory.
    public static func download(quantization: Flux2QuantizationConfig = .memoryEfficient,
                                progress: @Sendable @escaping (Double) -> Void) async throws {
        let pipeline = makePipeline(quantization: quantization)
        let missing = pipeline.missingModels
        let encoder = qwen3Variant(for: quantization)
        let needsEncoder = !TextEncoderModelDownloader.isQwen3ModelDownloaded(variant: encoder)
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
            _ = try await TextEncoderModelDownloader().downloadQwen3(variant: encoder, progress: { fraction, _ in
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
        for variant in [Qwen3Variant.qwen3_4B_8bit, .qwen3_4B_4bit] {
            if let path = TextEncoderModelDownloader.findQwen3ModelPath(for: variant) {
                try? FileManager.default.removeItem(at: path)
            }
        }
    }

    // MARK: - Component-level management

    /// One downloadable FLUX weight component (a specific transformer/encoder precision or the VAE).
    public struct Flux2ComponentInfo: Identifiable, Sendable {
        public enum Kind: String, Sendable { case transformer = "Transformer", textEncoder = "Text encoder", vae = "VAE" }
        public let id: String
        public let title: String
        public let subtitle: String
        public let kind: Kind
        public let repo: String
        public let bytes: Int64
        public let isDownloaded: Bool
    }

    /// Every individually-managed FLUX component, with on-disk size when present (else an estimate).
    public static func allComponents() -> [Flux2ComponentInfo] {
        func gib(_ g: Double) -> Int64 { Int64(g * 1_073_741_824) }
        func size(_ down: Bool, _ path: URL?, _ estimate: Int64) -> Int64 {
            (down ? path.map { directorySize(at: $0) } : nil) ?? estimate
        }
        func tx(_ v: ModelRegistry.TransformerVariant, _ est: Int64) -> (Bool, Int64) {
            let down = Flux2ModelDownloader.isDownloaded(.transformer(v))
            return (down, size(down, Flux2ModelDownloader.findModelPath(for: .transformer(v)), est))
        }
        func enc(_ v: Qwen3Variant, _ est: Int64) -> (Bool, Int64) {
            let down = TextEncoderModelDownloader.isQwen3ModelDownloaded(variant: v)
            return (down, size(down, TextEncoderModelDownloader.findQwen3ModelPath(for: v), est))
        }
        let txBF = tx(.klein4B_bf16, gib(7.2)), tx8 = tx(.klein4B_8bit, gib(3.3))
        let e8 = enc(.qwen3_4B_8bit, gib(4.0)), e4 = enc(.qwen3_4B_4bit, gib(1.9))
        let vaeDown = Flux2ModelDownloader.isDownloaded(.vae(.smallDecoder))
        let vaeBytes = size(vaeDown, Flux2ModelDownloader.findModelPath(for: .vae(.smallDecoder)), gib(0.57))
        var comps: [Flux2ComponentInfo] = [
            .init(id: "tx-bf16", title: "Klein 4B · 16-bit", subtitle: "also serves 4-bit (quantizes on load)",
                  kind: .transformer, repo: "black-forest-labs/FLUX.2-klein-4B", bytes: txBF.1, isDownloaded: txBF.0),
            .init(id: "tx-8bit", title: "Klein 4B · 8-bit", subtitle: "",
                  kind: .transformer, repo: "black-forest-labs/FLUX.2-klein-4B", bytes: tx8.1, isDownloaded: tx8.0),
            .init(id: "enc-8bit", title: "Qwen3 4B · 8-bit", subtitle: "better prompt fidelity",
                  kind: .textEncoder, repo: "lmstudio-community/Qwen3-4B-MLX-8bit", bytes: e8.1, isDownloaded: e8.0),
            .init(id: "enc-4bit", title: "Qwen3 4B · 4-bit", subtitle: "smaller",
                  kind: .textEncoder, repo: "lmstudio-community/Qwen3-4B-MLX-4bit", bytes: e4.1, isDownloaded: e4.0),
            .init(id: "vae", title: "Small VAE Decoder", subtitle: "recommended default",
                  kind: .vae, repo: "black-forest-labs/FLUX.2-small-decoder", bytes: vaeBytes, isDownloaded: vaeDown),
        ]
        #if os(iOS)
        // iPhone uses the pre-quantized 4-bit transformer — it loads directly into a QuantizedLinear
        // shell with no float16 spike (the bf16/8-bit files would not fit the phone's memory budget).
        let tx4 = tx(.klein4B_4bit, gib(2.18))
        comps.insert(.init(id: "tx-4bit", title: "Klein 4B · 4-bit", subtitle: "pre-quantized · iPhone",
                           kind: .transformer, repo: "mlx-community/flux2-klein-4b-4bit",
                           bytes: tx4.1, isDownloaded: tx4.0), at: 2)
        #endif
        return comps
    }

    /// Download a single component by its `Flux2ComponentInfo.id`, reporting 0...1 progress.
    public static func downloadComponent(_ id: String, progress: @Sendable @escaping (Double) -> Void) async throws {
        switch id {
        case "tx-bf16": _ = try await Flux2ModelDownloader().download(.transformer(.klein4B_bf16), progress: { f, _ in progress(f) })
        case "tx-8bit": _ = try await Flux2ModelDownloader().download(.transformer(.klein4B_8bit), progress: { f, _ in progress(f) })
        case "tx-4bit": _ = try await Flux2ModelDownloader().download(.transformer(.klein4B_4bit), progress: { f, _ in progress(f) })
        case "enc-8bit": _ = try await TextEncoderModelDownloader().downloadQwen3(variant: .qwen3_4B_8bit, progress: { f, _ in progress(f) })
        case "enc-4bit": _ = try await TextEncoderModelDownloader().downloadQwen3(variant: .qwen3_4B_4bit, progress: { f, _ in progress(f) })
        case "vae": _ = try await Flux2ModelDownloader().download(.vae(.smallDecoder), progress: { f, _ in progress(f) })
        default: break
        }
        progress(1)
    }

    /// Delete a single component's weights by its `Flux2ComponentInfo.id`.
    public static func deleteComponent(_ id: String) throws {
        switch id {
        case "tx-bf16": try Flux2ModelDownloader.delete(.transformer(.klein4B_bf16))
        case "tx-8bit": try Flux2ModelDownloader.delete(.transformer(.klein4B_8bit))
        case "tx-4bit": try Flux2ModelDownloader.delete(.transformer(.klein4B_4bit))
        case "enc-8bit": if let p = TextEncoderModelDownloader.findQwen3ModelPath(for: .qwen3_4B_8bit) { try FileManager.default.removeItem(at: p) }
        case "enc-4bit": if let p = TextEncoderModelDownloader.findQwen3ModelPath(for: .qwen3_4B_4bit) { try FileManager.default.removeItem(at: p) }
        case "vae": try Flux2ModelDownloader.delete(.vae(.smallDecoder))
        default: break
        }
    }

    /// The component ids a given precision actually uses (the "active recipe"). 16-bit and 4-bit
    /// transformer both run off the bf16 file (4-bit quantizes on load).
    public static func activeComponentIDs(transformer: FluxTransformerPrecision,
                                          encoder: FluxEncoderPrecision) -> [String] {
        #if os(iOS)
        // iPhone 4-bit runs off the pre-quantized file; Mac 4-bit quantizes the bf16 file on load.
        let tx = (transformer == .bit8) ? "tx-8bit" : "tx-4bit"
        #else
        let tx = (transformer == .bit8) ? "tx-8bit" : "tx-bf16"
        #endif
        let enc = (encoder == .bit8) ? "enc-8bit" : "enc-4bit"
        return [tx, enc, "vae"]
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
        // the catalog currently ships Klein 4B; makePipeline forces the pre-quantized 4-bit
        // transformer on iPhone and keeps the Mac on-the-fly path unchanged.
        let pipeline = Self.makePipeline(quantization: quantization)
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
