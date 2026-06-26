import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@preconcurrency import MLX
import DiffusionCore
import Flux2DiffusionEngine

// Minimal end-to-end runner for the FLUX.2 facade engine:
//   swift run flux2-demo "a calm otter on a mossy river stone, soft morning light"
// Loads FLUX.2 Klein 4B (downloads weights from HuggingFace on first run) and writes
// flux-out.png. Requires a Metal GPU (macOS 15+).

/// The facade ignores `source` (Flux2Pipeline loads its own weights), so a null source suffices.
struct NullWeightSource: WeightSource {
    let isStreaming = false
    func tensor(_ key: TensorKey) throws -> MLXArray { throw EngineError.notLoaded }
}

func writePNG(_ image: CGImage, to url: URL) throws {
    guard let dest = CGImageDestinationCreateWithURL(
        url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw EngineError.decodeFailed
    }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else { throw EngineError.decodeFailed }
}

func run() async throws {
    let args = Array(CommandLine.arguments.dropFirst())
    let prompt = args.isEmpty
        ? "a calm otter on a mossy river stone, soft morning light, shallow depth of field"
        : args.joined(separator: " ")

    let engine = Flux2FacadeEngine()
    let model = ModelCatalog.fluxKlein4B
    guard let variant = model.variants.first(where: { $0.precision == .q4 }) else { return }

    print("Loading \(model.displayName) (\(variant.precision.label)) — downloads on first run…")
    try await engine.load(model, variant: variant, source: NullWeightSource()) { fraction in
        print(String(format: "  load %3.0f%%", fraction * 100))
    }

    print("Generating: \(prompt)")
    let image = try await engine.generate(
        GenerationRequest(prompt: prompt, steps: 6, seed: 42, size: .square1024)
    ) { progress in
        if case let .denoising(step, total, _) = progress { print("  step \(step)/\(total)") }
    }

    let url = URL(fileURLWithPath: "flux-out.png")
    try writePNG(image, to: url)
    print("Saved \(url.path) (\(image.width)×\(image.height))")
}

do {
    if CommandLine.arguments.dropFirst().contains("--parity") {
        try await runParity()
    } else {
        try await run()
    }
} catch {
    FileHandle.standardError.write(Data("Error: \(error)\n".utf8))
    exit(1)
}
