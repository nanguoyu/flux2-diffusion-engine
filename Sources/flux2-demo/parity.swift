import Foundation
import CoreGraphics
@preconcurrency import MLX
import DiffusionCore
import Flux2DiffusionEngine

// The 512 PARITY GATE: prove the block-streaming FLUX path (MLXDiffusionEngine + Flux2Architecture)
// produces the same 512 image as the resident facade (Flux2Pipeline) at the same seed. Run on a Mac
// with real weights:  swift run flux2-demo --parity
//
// On a Mac the streaming engine loads resident (plenty of memory), but it runs the SAME architecture
// code (encode → streamEmbed → 25 blocks → streamUnembed → decode), so passing here validates the
// whole path; the only iPhone difference is per-step block load/release (memory, not math). Both runs
// use the 4-bit (klein4B_4bit) weights so the comparison is apples-to-apples and the streaming source
// finds its transformer.

private func rgbaBytes(_ image: CGImage) -> [UInt8]? {
    let w = image.width, h = image.height
    var data = [UInt8](repeating: 0, count: w * h * 4)
    let ok = data.withUnsafeMutableBytes { ptr -> Bool in
        guard let ctx = CGContext(data: ptr.baseAddress, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return true
    }
    return ok ? data : nil
}

/// Per-channel max absolute difference and PSNR (dB) between two equally-sized images.
func compareImages(_ a: CGImage, _ b: CGImage) -> (maxDiff: Int, psnr: Double) {
    guard a.width == b.width, a.height == b.height, let pa = rgbaBytes(a), let pb = rgbaBytes(b) else {
        return (255, 0)
    }
    var maxDiff = 0
    var sse = 0.0
    var count = 0
    for i in 0 ..< pa.count where i % 4 != 3 {   // skip the alpha channel
        let d = abs(Int(pa[i]) - Int(pb[i]))
        maxDiff = max(maxDiff, d)
        sse += Double(d * d)
        count += 1
    }
    let mse = sse / Double(max(count, 1))
    let psnr = mse == 0 ? Double.infinity : 10 * log10(255.0 * 255.0 / mse)
    return (maxDiff, psnr)
}

func runParity() async throws {
    let prompt = "a red panda on a mossy rock, soft morning light, shallow depth of field"
    let seed: UInt64 = 42
    let model = ModelCatalog.fluxKlein4B
    guard let variant = model.variants.first(where: { $0.precision == .q4 }) else { return }
    let size = ImageSize.square512
    let steps = 4

    print("=== 512 parity gate (4-bit) — prompt: \(prompt) ===")

    // 1) Resident facade (the oracle).
    print("[resident] loading facade…")
    let facade = Flux2FacadeEngine(transformer: .bit4, encoder: .bit4, decoder: .small)
    try await facade.load(model, variant: variant, source: NullWeightSource()) { _ in }
    print("[resident] generating…")
    let resident = try await facade.generate(
        GenerationRequest(prompt: prompt, steps: steps, seed: seed, size: size)) { _ in }
    await facade.unload()
    try writePNG(resident, to: URL(fileURLWithPath: "parity-resident.png"))

    // 2) Streaming engine (the path under test).
    print("[streamed] loading MLXDiffusionEngine + Flux2Architecture…")
    let streamed = MLXDiffusionEngine(architecture: Flux2Architecture(),   // defaults to .smallDecoder
                                      device: .current)
    let source = try Flux2ComponentSource.openKlein4BStreaming()
    try await streamed.load(model, variant: variant, source: source) { _ in }
    print("[streamed] generating…")
    let streamedImage = try await streamed.generate(
        GenerationRequest(prompt: prompt, steps: steps, seed: seed, size: size)) { _ in }
    await streamed.unload()
    try writePNG(streamedImage, to: URL(fileURLWithPath: "parity-streamed.png"))

    // 3) Compare.
    let (maxDiff, psnr) = compareImages(resident, streamedImage)
    let verdict = psnr.isInfinite || psnr > 35 ? "PASS ✅" : "CHECK ⚠️ (encode/decode/VAE-variant divergence?)"
    print(String(format: "512 PARITY: maxPixelDiff=%d  PSNR=%.1f dB  →  %@", maxDiff, psnr, verdict))
    print("Wrote parity-resident.png and parity-streamed.png for visual inspection.")
}
