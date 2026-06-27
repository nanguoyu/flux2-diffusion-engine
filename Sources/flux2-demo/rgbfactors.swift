import Foundation
@preconcurrency import MLX
import Flux2Core
import Flux2DiffusionEngine

// Derive the FLUX.2 latent→RGB linear approximation used by the cheap denoise-preview.
//
// The preview wants, per denoise step:  RGB ≈ factors · unpatchify(unpack(latent)) + bias  — i.e. a
// single matmul over the 32-channel unpatchified latent, with NO VAE stats needed at preview time.
//
// Why least squares and not a basis-decode: the small-decoder VAE's GroupNorm normalizes constant
// inputs away, so single-channel basis vectors collapse — only a regression over real sampled latents
// recovers each channel's color contribution.
//
// Why the factors can be stats-free: the true path is
//     vaeLatent = unpatchify( denormalize(patched) )        // denormalize = per-channel affine x*s+m
//     RGB ≈ decode(vaeLatent)  ≈  A · vaeLatent + b          // locally affine
// Both denormalize and unpatchify are affine/rearrange, so RGB is still affine in the RAW (un-denorm)
// unpatchified latent X = unpatchify(patched). We regress on X, and the fitted [33,3] absorbs the
// denormalize affine. So the preview is literally  factors · unpatchify(unpack(latent)).
// Run:  swift run flux2-demo --rgbfactors
func runRGBFactors() throws {
    let vae = try Flux2StreamingSupport.loadVAE(variant: .smallDecoder)

    // Denormalize stats live in PATCHIFIED space — shape [128] (= 32 latent ch × 2×2 patch), NOT [32].
    let mean = vae.batchNormRunningMean      // [128]
    let varr = vae.batchNormRunningVar       // [128]
    eval(mean, varr)
    let Cpatch = mean.shape.reduce(1, *)     // expect 128
    print("VAE denorm-stat count (patchified channels): \(Cpatch)")
    print("batchNormRunningMean shape: \(mean.shape)  batchNormRunningVar shape: \(varr.shape)")

    // Latent geometry: patchified [1,128,h,w] → unpatchified VAE latent [1,32,2h,2w] → decode [1,3,16h,16w].
    let h = 16, w = 16                        // unpatchified 32x32 → decode 256x256 (cheap)
    let latentC = 32
    let N = 16                                // number of sampled latents
    MLXRandom.seed(1234)

    var xRows: [MLXArray] = []                // each [Hl*Wl, 32]
    var yRows: [MLXArray] = []                // each [Hl*Wl, 3]
    var Hl = 0, Wl = 0
    for n in 0 ..< N {
        // RAW standard-normal PATCHIFIED latent — the layout the denoise loop carries (packed form of it).
        let patched = MLXRandom.normal([1, Cpatch, h, w])

        // True VAE input: denormalize (per-128-channel affine) then unpatchify to 32 channels.
        let denorm = LatentUtils.denormalizeLatentsWithBatchNorm(
            patched, runningMean: mean, runningVar: varr)
        let vaeLatent = LatentUtils.unpatchifyLatents(denorm)          // [1, 32, 2h, 2w]
        let rgb = vae.decode(vaeLatent)                               // [1, 3, 16h, 16w] in ~[-1,1]

        // X = the RAW (un-denormalized) unpatchified latent — what the preview will see.
        let xLatent = LatentUtils.unpatchifyLatents(patched)          // [1, 32, 2h, 2w]
        Hl = xLatent.shape[2]; Wl = xLatent.shape[3]

        // Downsample RGB 8x (avg-pool) so each latent cell maps to exactly one RGB row.
        let rgbDown = MLX.mean(rgb.reshaped([1, 3, Hl, 8, Wl, 8]), axes: [3, 5])   // [1,3,Hl,Wl]

        let xN = xLatent.reshaped([latentC, Hl * Wl]).transposed(1, 0)   // [Hl*Wl, 32]
        let yN = rgbDown.reshaped([3, Hl * Wl]).transposed(1, 0)         // [Hl*Wl, 3]
        eval(xN, yN)
        xRows.append(xN)
        yRows.append(yN)
        if n == 0 { print("decode rgb \(rgb.shape) → down \(rgbDown.shape); X latent \(xLatent.shape)") }
    }

    let X = MLX.concatenated(xRows, axis: 0)         // [M, 32]
    let Y = MLX.concatenated(yRows, axis: 0)         // [M, 3]
    let M = X.shape[0]
    // Augment X with a ones column so the last solved row is the bias.
    let Xa = MLX.concatenated([X, MLX.ones([M, 1])], axis: 1)   // [M, 33]

    // Normal equations:  (XaᵀXa) w = Xaᵀ Y.
    let A = Xa.transposed(1, 0).matmul(Xa)           // [33, 33]
    let b = Xa.transposed(1, 0).matmul(Y)            // [33, 3]
    let wMat = MLX.solve(A, b, stream: .cpu)         // [33, 3]   (linalg::solve runs on CPU stream only)
    eval(wMat)

    let flat = wMat.asArray(Float.self)              // row-major [33*3]
    let rows = latentC + 1
    print("")
    print("static let latentRGBFactors: [[Float]] = [")
    for r in 0 ..< rows {
        let a = flat[r * 3 + 0], g = flat[r * 3 + 1], bl = flat[r * 3 + 2]
        let comment = r < latentC ? "// channel \(r)" : "// row \(r + 1) = bias"
        print(String(format: "    [%.5f, %.5f, %.5f],   %@", a, g, bl, comment))
    }
    print("]")

    let biasA = flat[latentC * 3 + 0], biasG = flat[latentC * 3 + 1], biasB = flat[latentC * 3 + 2]
    print(String(format: "\nsanity: bias=[%.4f, %.4f, %.4f]  (M=%d rows fit, X has %d latent channels)",
                 biasA, biasG, biasB, M, latentC))
}
