// Flux2ComponentSourceTests — the composite weight router that lets the generic engine stream FLUX's
// transformer while keeping the encoder/VAE resident. Verifies the routing the streaming path relies
// on: bare keys (the engine's per-block fetches) hit the transformer; "<component>/" keys hit that
// component; freesOnRelease/isStreaming mirror the transformer; releaseComponent drops a sub-source.

import XCTest
@preconcurrency import MLX
@testable import Flux2DiffusionEngine
import DiffusionCore

private final class StubSource: WeightSource, @unchecked Sendable {
    let isStreaming: Bool
    let freesOnRelease: Bool
    let tag: Float
    private(set) var requested: [String] = []
    init(tag: Float, isStreaming: Bool, freesOnRelease: Bool) {
        self.tag = tag; self.isStreaming = isStreaming; self.freesOnRelease = freesOnRelease
    }
    func tensor(_ key: TensorKey) throws -> MLXArray {
        requested.append(key.name)
        return MLXArray(tag)
    }
}

final class Flux2ComponentSourceTests: XCTestCase {

    private func makeComposite() -> (Flux2ComponentSource, tx: StubSource, enc: StubSource, vae: StubSource) {
        let tx = StubSource(tag: 1, isStreaming: true, freesOnRelease: true)
        let enc = StubSource(tag: 2, isStreaming: false, freesOnRelease: false)
        let vae = StubSource(tag: 3, isStreaming: false, freesOnRelease: false)
        let c = Flux2ComponentSource(sources: [.transformer: tx, .textEncoder: enc, .vae: vae])
        return (c, tx, enc, vae)
    }

    func testBareKeyRoutesToTransformer() throws {
        let (c, tx, _, _) = makeComposite()
        let v = try c.tensor(TensorKey("transformer_blocks.0.attn.to_q.weight"))
        XCTAssertEqual(v.item(Float.self), 1)   // transformer's tag
        XCTAssertEqual(tx.requested, ["transformer_blocks.0.attn.to_q.weight"])  // prefix-stripped (none to strip)
    }

    func testPrefixedKeyRoutesAndStripsComponent() throws {
        let (c, _, enc, vae) = makeComposite()
        _ = try c.tensor(TensorKey("text_encoder/model.norm.weight"))
        _ = try c.tensor(TensorKey("vae/decoder.conv_in.weight"))
        XCTAssertEqual(enc.requested, ["model.norm.weight"])      // "text_encoder/" stripped
        XCTAssertEqual(vae.requested, ["decoder.conv_in.weight"]) // "vae/" stripped
    }

    func testStreamingFlagsMirrorTransformer() {
        let (c, _, _, _) = makeComposite()
        XCTAssertTrue(c.isStreaming)        // transformer streams
        XCTAssertTrue(c.freesOnRelease)     // transformer frees on release
    }

    func testReleaseComponentDropsSubSource() {
        let (c, _, _, _) = makeComposite()
        XCTAssertNotNil(c.subSource(.textEncoder))
        c.releaseComponent(.textEncoder)
        XCTAssertNil(c.subSource(.textEncoder))
        XCTAssertNotNil(c.subSource(.transformer))   // others untouched
    }

    func testUnknownBareKeyThrowsWhenTransformerReleased() {
        let (c, _, _, _) = makeComposite()
        c.releaseComponent(.transformer)
        XCTAssertThrowsError(try c.tensor(TensorKey("transformer_blocks.0.x")))
    }
}
