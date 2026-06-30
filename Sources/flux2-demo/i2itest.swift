import Foundation
import CoreGraphics
import ImageIO
@preconcurrency import MLX
import DiffusionCore
import Flux2Core
import Flux2DiffusionEngine

/// Repro for the "transparent-PNG reference → framed output" artifact, on a Mac (no device needed).
///
/// Builds a synthetic 1280×1280 RGBA reference: an opaque subject on a TRANSPARENT border. 1280 is above
/// the facade's 1024² reference cap, so the downscale/resize branch in `preprocessImageForVAE` runs — the
/// suspected source of a dark edge frame. Runs facade i2i at 512 (same VAE full-frame decode the iPhone
/// 512 i2i uses) and writes `i2itest-out.png` to inspect the border. It also prints the brightness of the
/// flattened+resized reference's outermost rows/cols vs just inside, isolating the reference preprocessing
/// from the model. `--ref <path>` uses a real image instead of the synthetic one.
func makeTransparentRef(side: Int = 1280) -> CGImage {
    let ctx = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
                        space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.clear(CGRect(x: 0, y: 0, width: side, height: side))           // fully transparent
    ctx.setFillColor(red: 0.85, green: 0.45, blue: 0.2, alpha: 1)      // opaque subject
    let m = side / 6
    ctx.fillEllipse(in: CGRect(x: m, y: m, width: side - 2 * m, height: side - 2 * m))
    return ctx.makeImage()!
}

func runI2ITest() async throws {
    let model = ModelCatalog.fluxKlein4B
    guard let variant = model.variants.first(where: { $0.precision == .q4 }) else { return }
    let args = Array(CommandLine.arguments.dropFirst())

    let ref: CGImage
    if let i = args.firstIndex(of: "--ref"), args.indices.contains(i + 1),
       let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: args[i + 1]) as CFURL, nil),
       let img = CGImageSourceCreateImageAtIndex(src, 0, nil) {
        ref = img
    } else {
        ref = makeTransparentRef()
    }
    print("[i2itest] reference \(ref.width)×\(ref.height), alphaInfo=\(ref.alphaInfo.rawValue)")

    // What the VAE actually sees: flatten + resize to 512. Inspect the outermost ring vs just inside it,
    // to tell a reference-preprocessing frame (this print) from a VAE-decode frame (the output image).
    let processed = Flux2Pipeline.preprocessImageForVAE(ref, targetHeight: 512, targetWidth: 512)  // [1,3,512,512]
    MLX.eval(processed)
    let r = processed[0, 0]                                            // R channel [512,512]
    func rowMean(_ y: Int) -> Float { r[y].mean().item(Float.self) }
    func colMean(_ x: Int) -> Float { r[0..., x].mean().item(Float.self) }
    print("[i2itest] reference border (1.0=white, -1.0=black):")
    print(String(format: "   topRow y0=%.3f  y2=%.3f  y8=%.3f   |   leftCol x0=%.3f  x2=%.3f  x8=%.3f",
                 rowMean(0), rowMean(2), rowMean(8), colMean(0), colMean(2), colMean(8)))

    let prompt = args.contains("--prompt")
        ? args[(args.firstIndex(of: "--prompt")! + 1)]
        : "a cute animal sticker, plain white background"

    // Decoder under test: --standard uses the wider-channel standard VAE (what an iPhone 512 render may
    // use), default is the small decoder.
    let useStd = args.contains("--standard")
    let vae: ModelRegistry.VAEVariant = useStd ? .standard : .smallDecoder
    print("[i2itest] decoder = \(useStd ? "STANDARD" : "small")")

    // 1) FACADE (Mac path): reference encoded at up to 1024².
    let facade = Flux2FacadeEngine(transformer: .bit4, encoder: .bit4, decoder: useStd ? .standard : .small)
    try await facade.load(model, variant: variant, source: NullWeightSource()) { _ in }
    let outFacade = try await facade.generate(
        GenerationRequest(prompt: prompt, steps: 4, seed: 42, size: ImageSize(width: 512, height: 512),
                          referenceImages: [ref])) { _ in }
    await facade.unload()
    try writePNG(outFacade, to: URL(fileURLWithPath: "i2itest-facade.png"))
    print("[i2itest] wrote i2itest-facade.png (facade, 1024² ref)")

    // 2) STREAMING (the iPhone path): forced block-streaming, reference capped to 512² (single ref).
    let device = DeviceTier(physicalMemoryBytes: 4_175_000_000, isPhone: false)
    let streamed = MLXDiffusionEngine(architecture: Flux2Architecture(vaeVariant: vae), device: device, streamEvalEveryK: 1)
    let source = try Flux2ComponentSource.openKlein4BStreaming()
    try await streamed.load(model, variant: variant, source: source) { _ in }
    let outStream = try await streamed.generate(
        GenerationRequest(prompt: prompt, steps: 4, seed: 42, size: ImageSize(width: 512, height: 512),
                          referenceImage: ref)) { _ in }
    await streamed.unload()
    try writePNG(outStream, to: URL(fileURLWithPath: "i2itest-stream.png"))
    print("[i2itest] wrote i2itest-stream.png (streaming, 512² ref) — the iPhone path")
}
