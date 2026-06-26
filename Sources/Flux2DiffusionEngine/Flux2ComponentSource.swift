@preconcurrency import MLX
import Foundation
import DiffusionCore
import Flux2Core

/// Resolves FLUX.2's 3-namespace collision at the `WeightSource` seam — the FLUX analogue of
/// `ZImageComponentSource`, with the identical `text_encoder/` `transformer/` `vae/` layout.
///
/// `MLXDiffusionEngine` hands the architecture ONE flat `WeightSource`, but FLUX.2 Klein is three
/// component trees whose key spaces collide on generic names. This COMPOSITE owns one sub-source per
/// component. `Flux2Architecture` downcasts the engine's `source` back to this type and pulls the
/// per-component sub-source it needs each phase (encode → transformer → vae).
///
/// `tensor(_:)` routes an explicit `"<component>/"` prefix to that component and serves any BARE key
/// from the transformer — because the engine streams the denoiser blocks by calling
/// `source.tensor("transformer_blocks.0.…")` on this composite directly. `isStreaming` /
/// `freesOnRelease` come from the TRANSFORMER sub-source: it is the one streamed block-by-block, and
/// its `freesOnRelease` gates the streaming residency plan (the engine throws `streamingUnavailable`
/// if a streaming plan meets a source that doesn't free on release).
public final class Flux2ComponentSource: WeightSource, @unchecked Sendable {

    public enum Component: String, CaseIterable, Sendable {
        case textEncoder = "text_encoder"
        case transformer = "transformer"
        case vae = "vae"
    }

    public enum SourceError: Error, CustomStringConvertible {
        case unknownComponent(String)
        case missingComponentFolder(String)
        case noSafetensors(String)
        public var description: String {
            switch self {
            case .unknownComponent(let k):
                return "Flux2ComponentSource: key '\(k)' has no '<component>/' prefix (expected one of \(Component.allCases.map(\.rawValue)))"
            case .missingComponentFolder(let c):
                return "Flux2ComponentSource: component folder '\(c)' does not exist in the model directory"
            case .noSafetensors(let c):
                return "Flux2ComponentSource: no .safetensors files in component folder '\(c)'"
            }
        }
    }

    private let lock = NSLock()
    private var sources: [Component: any WeightSource]
    /// The `tokenizer/` folder, passed through so the architecture can resolve the Qwen3 tokenizer
    /// without a hub round-trip. `nil` if the model directory has no `tokenizer/` folder.
    public let tokenizerDirectory: URL?

    public var isStreaming: Bool { lock.withLock { sources[.transformer]?.isStreaming ?? false } }
    public var freesOnRelease: Bool { lock.withLock { sources[.transformer]?.freesOnRelease ?? false } }

    public init(sources: [Component: any WeightSource], tokenizerDirectory: URL? = nil) {
        self.sources = sources
        self.tokenizerDirectory = tokenizerDirectory
    }

    /// The sub-source for one component, or `nil` if this composite wasn't given that component.
    public func subSource(_ component: Component) -> (any WeightSource)? { lock.withLock { sources[component] } }

    /// Drop a component's sub-source after its phase. Releasing the ~2 GB Qwen3 text encoder after
    /// `encode` reclaims it before the transformer streams (two-phase staging).
    public func releaseComponent(_ component: Component) {
        lock.withLock { _ = sources.removeValue(forKey: component) }
        MLX.GPU.clearCache()
    }

    /// Resolve a tensor by key. An explicit `"<component>/"` prefix routes to that component; a BARE
    /// key is served from the TRANSFORMER (the streamed component) — the engine streams blocks by
    /// calling `source.tensor("transformer_blocks.0.…")` on this composite directly.
    public func tensor(_ key: TensorKey) throws -> MLXArray {
        if let slash = key.name.firstIndex(of: "/"),
           let component = Component(rawValue: String(key.name[..<slash])),
           let sub = lock.withLock({ sources[component] }) {
            let rest = String(key.name[key.name.index(after: slash)...])
            return try sub.tensor(TensorKey(rest))
        }
        guard let transformer = lock.withLock({ sources[.transformer] }) else {
            throw SourceError.unknownComponent(key.name)
        }
        return try transformer.tensor(key)
    }
}

public extension Flux2ComponentSource {

    /// Open a FLUX.2 model directory (`text_encoder/`, `transformer/`, `vae/`, `tokenizer/`) as a
    /// composite. `streaming == true` opens the transformer with `RangedFileWeightSource` (`pread` on
    /// demand, `freesOnRelease == true`) so block-streaming actually reclaims memory; the encoder and
    /// VAE stay mmap-backed (`SafetensorsWeightSource`) since they load resident in one pass.
    static func open(modelDirectory: URL, streaming: Bool) throws -> Flux2ComponentSource {
        var sources: [Component: any WeightSource] = [:]
        for component in Component.allCases {
            let dir = modelDirectory.appendingPathComponent(component.rawValue)
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue else {
                throw SourceError.missingComponentFolder(component.rawValue)
            }
            let files = try FileManager.default
                .contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "safetensors" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            guard !files.isEmpty else { throw SourceError.noSafetensors(component.rawValue) }
            let streamThis = streaming && component == .transformer
            let sub: any WeightSource = streamThis
                ? try RangedFileWeightSource(files: files, isStreaming: true)
                : try SafetensorsWeightSource(files: files, isStreaming: false)
            sources[component] = sub
        }
        let tok = modelDirectory.appendingPathComponent("tokenizer")
        var tokIsDir: ObjCBool = false
        let tokExists = FileManager.default.fileExists(atPath: tok.path, isDirectory: &tokIsDir) && tokIsDir.boolValue
        return Flux2ComponentSource(sources: sources, tokenizerDirectory: tokExists ? tok : nil)
    }

    /// Build a TRANSFORMER-ONLY streaming source for the pre-quantized Klein 4B 4-bit checkpoint from
    /// its downloaded cache directory — the entry point the app uses for the 1024-on-iPhone streaming
    /// engine. FLUX downloads its three components to separate caches (unlike Z-Image's single model
    /// dir), and `Flux2Architecture` loads the encoder + VAE from their own caches, so only the
    /// transformer sub-source is needed: the streaming blocks + `loadShared` read through it, and the
    /// engine's `freesOnRelease` gate is satisfied by the `RangedFileWeightSource`.
    static func openKlein4BStreaming() throws -> Flux2ComponentSource {
        guard let dir = Flux2ModelDownloader.findModelPath(for: .transformer(.klein4B_4bit)) else {
            throw SourceError.missingComponentFolder("transformer (klein4B_4bit)")
        }
        let files = try FileManager.default
            .contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw SourceError.noSafetensors("transformer") }
        let txSource = try RangedFileWeightSource(files: files, isStreaming: true)
        return Flux2ComponentSource(sources: [.transformer: txSource])
    }
}
