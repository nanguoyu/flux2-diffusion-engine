import Foundation
import CoreGraphics
import ImageIO
@preconcurrency import MLX
import MLXNN
import DiffusionCore
import Flux2Core
import Flux2DiffusionEngine

// Decisive parity diagnostic: does the per-block streaming loader produce the SAME weights as the
// validated whole-model pre-quantized loader? Loads the real cached klein4B_4bit checkpoint both ways
// and compares block 0's params + the shared-shell params tensor-by-tensor. Run: swift run flux2-demo --diag

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    if a.shape != b.shape { return Float.infinity }
    return (a.asType(.float32) - b.asType(.float32)).abs().max().item(Float.self)
}

func runDiag() async throws {
    guard let dir = Flux2ModelDownloader.findModelPath(for: .transformer(.klein4B_4bit)) else {
        print("transformer not downloaded — run `swift run flux2-demo` once first"); return
    }
    print("transformer dir: \(dir.path)")

    // 1) Whole-model load via the VALIDATED path.
    let full = Flux2Transformer2DModel(config: .klein4B)
    quantize(model: full, groupSize: 64, bits: 4)
    var weights = try Flux2WeightLoader.loadWeights(from: dir.path)
    try Flux2WeightLoader.applyPreQuantizedTransformerWeights(&weights, to: full)

    // 2) My per-block + shared load.
    let source = try Flux2ComponentSource.openKlein4BStreaming()
    let myBlock = Flux2TransformerBlock(dim: 3072, numHeads: 24, headDim: 128)
    try Flux2Weights.loadDoubleBlock(0, from: source, into: myBlock)

    let shell = Flux2Transformer2DModel(config: Flux2Weights.shellConfig())
    try Flux2Weights.loadShared(from: source.subSource(.transformer)!, into: shell)

    // 3) Compare block 0 params.
    let refBlock = full.doubleStreamBlock(0)
    let refParams = Dictionary(uniqueKeysWithValues: refBlock.parameters().flattened())
    var worstBlock: (String, Float) = ("", 0)
    var missing = 0
    for (k, v) in myBlock.parameters().flattened() {
        guard let r = refParams[k] else { missing += 1; print("  block0 my-only key: \(k)"); continue }
        let d = maxAbsDiff(v, r)
        if d > worstBlock.1 { worstBlock = (k, d) }
    }
    print(String(format: "BLOCK0: worst |Δ| = %.6g at %@  (missing %d)", worstBlock.1, worstBlock.0, missing))

    // 4) Compare shared-shell params (those that exist in the full model too).
    let fullParams = Dictionary(uniqueKeysWithValues: full.parameters().flattened())
    var worstShared: (String, Float) = ("", 0)
    for (k, v) in shell.parameters().flattened() {
        guard let r = fullParams[k] else { continue }   // shell has only shared keys
        let d = maxAbsDiff(v, r)
        if d > worstShared.1 { worstShared = (k, d) }
    }
    print(String(format: "SHARED: worst |Δ| = %.6g at %@", worstShared.1, worstShared.0))

    if worstBlock.1 == 0 && worstShared.1 == 0 && missing == 0 {
        print("→ weights IDENTICAL.")
    } else {
        print("→ weights DIFFER — the per-block/shared loader is the bug."); return
    }

    // 5) Real-scale forward parity: full callAsFunction vs my shell + 25 streamed blocks, same inputs.
    print("loading all 25 blocks for the forward comparison…")
    var myBlocks: [Any] = []
    for i in 0 ..< 5 {
        let b = Flux2TransformerBlock(dim: 3072, numHeads: 24, headDim: 128)
        try Flux2Weights.loadDoubleBlock(i, from: source, into: b); myBlocks.append(b)
    }
    for j in 0 ..< 20 {
        let b = Flux2SingleTransformerBlock(dim: 3072, numHeads: 24, headDim: 128)
        try Flux2Weights.loadSingleBlock(j, from: source, into: b); myBlocks.append(b)
    }

    MLXRandom.seed(7)
    let imgSeq = 1024, txtSeq = 512   // the REAL text sequence length (Qwen3 pads to 512), not a toy 16
    let latent = MLXRandom.normal([1, imgSeq, 128]) * 0.1
    let textEmb = MLXRandom.normal([1, txtSeq, 7680]) * 0.1
    let ts = MLXArray([0.7] as [Float])
    let (textIds, imageIds, _) = LatentUtils.combinePositionIDs(textLength: txtSeq, height: 512, width: 512)

    let velA = full(hiddenStates: latent, encoderHiddenStates: textEmb, timestep: ts,
                    guidance: nil, imgIds: imageIds, txtIds: textIds)

    let (h0, ctx) = shell.streamEmbed(hiddenStates: latent, encoderHiddenStates: textEmb,
                                      timestep: ts, guidance: nil, imgIds: imageIds, txtIds: textIds)
    var h = h0
    for i in 0 ..< 5 { h = Flux2Transformer2DModel.runDouble(myBlocks[i] as! Flux2TransformerBlock, hidden: h, context: ctx) }
    for j in 0 ..< 20 { h = Flux2Transformer2DModel.runSingle(myBlocks[5 + j] as! Flux2SingleTransformerBlock, hidden: h, context: ctx) }
    let velB = shell.streamUnembed(hidden: h, context: ctx)

    let fwdDiff = maxAbsDiff(velA, velB)
    print(String(format: "FORWARD: worst |Δ| velocity = %.6g", fwdDiff))
    if fwdDiff < 1e-2 {
        print("→ streaming forward MATCHES monolithic. The 512 divergence is in glue (encode/initialLatent/decode/sigmas), not the transformer.")
    } else {
        print("→ streaming forward DIVERGES at real scale — the decomposition (shell vs blocks) is the bug.")
    }

    // 6) Full GLUE test: my exact encode/init/sigmas/decode + the whole-model transformer, end-to-end,
    //    compared to the facade oracle (parity-resident.png). Isolates the glue from the engine wrapper.
    try await runGlue(full: full)
}

/// Validate the iPhone's >=1024 VAE tiling: decode a REAL 1024 latent both untiled and tiled
/// (.aggressive, the iPhone path) and compare. High PSNR = the tiling that kills the decode memory
/// spike doesn't introduce visible seams. The Mac parity runs untiled, so this is the one
/// iPhone-specific path it doesn't otherwise cover.
func runTile() async throws {
    guard let dir = Flux2ModelDownloader.findModelPath(for: .transformer(.klein4B_4bit)) else {
        print("transformer not downloaded — run `swift run flux2-demo` once first"); return
    }
    let full = Flux2Transformer2DModel(config: .klein4B)
    quantize(model: full, groupSize: 64, bits: 4)
    var weights = try Flux2WeightLoader.loadWeights(from: dir.path)
    try Flux2WeightLoader.applyPreQuantizedTransformerWeights(&weights, to: full)

    let prompt = "a red panda on a mossy rock, soft morning light, shallow depth of field"
    let enc = KleinTextEncoder(variant: .klein4B, quantization: .mlx4bit)
    try await enc.load()
    let textEmb = try enc.encode(prompt, upsample: false)
    let (vh, vw) = LatentUtils.validateDimensions(height: 1024, width: 1024)
    let patch = LatentUtils.generatePatchifiedLatents(height: vh, width: vw, seed: 42)
    var packed = LatentUtils.packPatchifiedToSequence(patch)
    let (textIds, imageIds, _) = LatentUtils.combinePositionIDs(textLength: textEmb.shape[1], height: vh, width: vw)
    let sigmas = Flux2Sigmas.schedule(width: 1024, height: 1024, steps: 4)
    print("TILE: denoising 1024 (\(packed.shape))…")
    for i in 0 ..< 4 {
        let vel = full(hiddenStates: packed, encoderHiddenStates: textEmb, timestep: MLXArray([sigmas[i]]),
                       guidance: nil, imgIds: imageIds, txtIds: textIds)
        packed = packed + (sigmas[i + 1] - sigmas[i]) * vel
    }
    let vae = try Flux2StreamingSupport.loadVAE(variant: .smallDecoder)
    let finalPatch = LatentUtils.unpackSequenceToPatchified(packed, height: vh, width: vw)
    let denorm = LatentUtils.denormalizeLatentsWithBatchNorm(
        finalPatch, runningMean: vae.batchNormRunningMean, runningVar: vae.batchNormRunningVar)
    let vaeLatent = LatentUtils.unpatchifyLatents(denorm)
    print("TILE: VAE latent \(vaeLatent.shape) — decoding untiled vs .aggressive tiled…")

    let gb = { (b: Int) in Double(b) / 1_073_741_824 }
    // Measure each decode in ISOLATION (clearCache + reset before each) so the first run's pool doesn't
    // inflate the second's peak. Tiled first.
    MLX.GPU.clearCache(); MLX.GPU.resetPeakMemory()
    guard let imgT = Flux2StreamingSupport.imageFromVAEOutput(vae.decodeWithTiling(vaeLatent, tiling: .aggressive)) else { print("tiled decode failed"); return }
    let tiledPeak = gb(MLX.GPU.peakMemory)
    MLX.GPU.clearCache(); MLX.GPU.resetPeakMemory()
    guard let imgU = Flux2StreamingSupport.imageFromVAEOutput(vae.decode(vaeLatent)) else { print("untiled decode failed"); return }
    let untiledPeak = gb(MLX.GPU.peakMemory)
    try writePNG(imgU, to: URL(fileURLWithPath: "tile-untiled.png"))
    try writePNG(imgT, to: URL(fileURLWithPath: "tile-tiled.png"))
    let (maxDiff, psnr) = compareImages(imgU, imgT)
    let verdict = psnr > 40 ? "CLEAN ✅" : psnr > 30 ? "OK ⚠️ (inspect seams)" : "SEAMS ❌"
    print(String(format: "TILE 1024: untiled-decode peak %.2f GB → tiled peak %.2f GB | tiled-vs-untiled PSNR %.1f dB maxΔ %d → %@",
                 untiledPeak, tiledPeak, psnr, maxDiff, verdict))
    print("Wrote tile-untiled.png and tile-tiled.png (both \(imgT.width)px) for visual inspection.")
}

func runGlue(full: Flux2Transformer2DModel) async throws {
    let prompt = "a red panda on a mossy rock, soft morning light, shallow depth of field"
    let seed: UInt64 = 42

    let enc = KleinTextEncoder(variant: .klein4B, quantization: .mlx4bit)
    try await enc.load()
    let textEmb = try enc.encode(prompt, upsample: false)
    let textLength = textEmb.shape[1]

    let (vh, vw) = LatentUtils.validateDimensions(height: 512, width: 512)
    let patch = LatentUtils.generatePatchifiedLatents(height: vh, width: vw, seed: seed)
    var packed = LatentUtils.packPatchifiedToSequence(patch)
    let (textIds, imageIds, _) = LatentUtils.combinePositionIDs(textLength: textLength, height: vh, width: vw)
    let sigmas = Flux2Sigmas.schedule(width: 512, height: 512, steps: 4)
    print("GLUE: textEmb \(textEmb.shape), packed \(packed.shape), sigmas \(sigmas)")

    for i in 0 ..< 4 {
        let t = MLXArray([sigmas[i]])
        let vel = full(hiddenStates: packed, encoderHiddenStates: textEmb, timestep: t,
                       guidance: nil, imgIds: imageIds, txtIds: textIds)
        packed = packed + (sigmas[i + 1] - sigmas[i]) * vel
    }

    let vae = try Flux2StreamingSupport.loadVAE(variant: .smallDecoder)
    let finalPatch = LatentUtils.unpackSequenceToPatchified(packed, height: vh, width: vw)
    let denorm = LatentUtils.denormalizeLatentsWithBatchNorm(
        finalPatch, runningMean: vae.batchNormRunningMean, runningVar: vae.batchNormRunningVar)
    let vaeLatent = LatentUtils.unpatchifyLatents(denorm)
    let decoded = vae.decode(vaeLatent)
    guard let image = Flux2StreamingSupport.imageFromVAEOutput(decoded) else { print("GLUE decode failed"); return }
    try writePNG(image, to: URL(fileURLWithPath: "diag-glue.png"))

    // Compare to the facade oracle from the parity run.
    let oracleURL = URL(fileURLWithPath: "parity-resident.png")
    if let src = CGImageSourceCreateWithURL(oracleURL as CFURL, nil),
       let oracle = CGImageSourceCreateImageAtIndex(src, 0, nil as CFDictionary?) {
        let (maxDiff, psnr) = compareImages(oracle, image)
        print(String(format: "GLUE vs facade: maxPixelDiff=%d  PSNR=%.1f dB  →  %@",
                     maxDiff, psnr, psnr > 35 ? "MATCH ✅ (glue is fine → engine wrapper is the bug)"
                                              : "DIFFER ⚠️ (the glue diverges from the facade)"))
    } else {
        print("GLUE: wrote diag-glue.png (run --parity first for parity-resident.png to compare)")
    }
}
