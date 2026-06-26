// Flux2DenoiserTests — verify the StreamHolder/StreamableBlock/Denoiser assembly threads the per-step
// context correctly: embed → 25 blocks → unembed through the wrappers must equal the monolithic
// transformer forward. Runs in resident mode over a tiny random-weight model (the per-block disk
// loader is verified separately in Flux2WeightsTests), so this isolates the WIRING — a holder that
// isn't set in embed, or blocks reading a stale context, would surface here.

import XCTest
@preconcurrency import MLX
import MLXRandom
@testable import Flux2DiffusionEngine
import DiffusionCore
import Flux2Core

private final class NullSource: WeightSource, @unchecked Sendable {
    let isStreaming = false
    let freesOnRelease = false
    func tensor(_ key: TensorKey) throws -> MLXArray { MLXArray(0) }
}

final class Flux2DenoiserTests: XCTestCase {

    private func tinyConfig() -> Flux2TransformerConfig {
        Flux2TransformerConfig(
            patchSize: 1, inChannels: 128, outChannels: 128,
            numLayers: 3, numSingleLayers: 4,
            attentionHeadDim: 128, numAttentionHeads: 2,   // innerDim 256
            jointAttentionDim: 256, pooledProjectionDim: 768,
            guidanceEmbeds: false, axesDimsRope: [32, 32, 32, 32], ropeTheta: 2000.0,
            mlpRatio: 2.0, activationFunction: "silu")
    }

    func testDenoiserWiringMatchesMonolithic() {
        MLXRandom.seed(424242)
        let model = Flux2Transformer2DModel(config: tinyConfig(), memoryOptimization: .disabled)
        let imgSeq = 16, txtSeq = 5
        let hs  = MLXRandom.normal([1, imgSeq, 128]) * 0.1
        let ehs = MLXRandom.normal([1, txtSeq, 256]) * 0.1
        let ts  = MLXArray([0.7] as [Float])
        let imgIds = MLXArray((0..<(imgSeq * 4)).map { Float($0 % 7) }, [imgSeq, 4])
        let txtIds = MLXArray((0..<(txtSeq * 4)).map { Float($0 % 3) }, [txtSeq, 4])

        let mono = model(hiddenStates: hs, encoderHiddenStates: ehs, timestep: ts,
                         imgIds: imgIds, txtIds: txtIds)

        // Assemble a denoiser in resident mode over the model's own blocks; shell = the model itself.
        let holder = Flux2StreamHolder()
        var blocks: [any StreamableBlock] = []
        for i in 0 ..< model.doubleStreamBlockCount {
            blocks.append(Flux2StreamableBlock(index: i, resident: model.doubleStreamBlock(i), holder: holder))
        }
        for j in 0 ..< model.singleStreamBlockCount {
            blocks.append(Flux2StreamableBlock(index: model.doubleStreamBlockCount + j,
                                               resident: model.singleStreamBlock(j), holder: holder))
        }
        let denoiser = Flux2Denoiser(shell: model, holder: holder, blocks: blocks,
                                     imgIds: imgIds, txtIds: txtIds)

        // Drive it exactly as MLXDiffusionEngine.generate does (load → run → release per block).
        let cond = Conditioning(embeddings: ehs)
        var h = denoiser.embed(latent: hs, timestep: ts, conditioning: cond)
        for block in denoiser.blocks {
            try? block.load(from: NullSource())
            h = block(h, conditioning: cond, timestep: ts)
            block.release()
        }
        let streamed = denoiser.unembed(h)

        XCTAssertEqual(streamed.shape, mono.shape)
        let diff = abs(streamed - mono).max().item(Float.self)
        XCTAssertLessThan(diff, 1e-4, "denoiser wiring diverged from monolithic forward (maxAbsDiff=\(diff))")
    }
}
