// Flux2WeightsTests — offline verification that the per-block streaming loader's disk→module key
// mapping is COMPLETE and EXACT, without the 2.18 GB checkpoint. Builds the real Klein 4B block
// shells, quantizes them exactly as the loader does, and asserts a bijection between the on-disk keys
// (mflux layout) and the quantized module's parameter paths — the per-block analogue of Flux2Core's
// whole-model notFound==0 / uncovered==0 gate. A naming drift here would silently leave a block
// projection on random weights (garbled output), so this is the gate that makes the loader trustworthy.

import XCTest
@preconcurrency import MLX
import MLXNN
@testable import Flux2DiffusionEngine
import Flux2Core

final class Flux2WeightsTests: XCTestCase {

    // Klein 4B transformer dims: innerDim 3072 = 24 heads × 128.
    private let dim = 3072, heads = 24, headDim = 128

    private func assertBijection(moduleKeys: Set<String>, mapped: Set<String>,
                                 file: StaticString = #filePath, line: UInt = #line) {
        let unmatchedDisk = mapped.subtracting(moduleKeys)      // disk keys with no module param
        let uncoveredModule = moduleKeys.subtracting(mapped)    // module params with no disk key
        XCTAssertEqual(unmatchedDisk, [], "disk keys not matching a module param: \(unmatchedDisk.sorted())", file: file, line: line)
        XCTAssertEqual(uncoveredModule, [], "module params with no disk key: \(uncoveredModule.sorted())", file: file, line: line)
    }

    func testDoubleBlockKeyCoverage() throws {
        let block = Flux2TransformerBlock(dim: dim, numHeads: heads, headDim: headDim)
        quantize(model: block, groupSize: 64, bits: 4)
        let moduleKeys = Set(block.parameters().flattened().map { $0.0 })
        let map = try Flux2Weights.blockKeyMap(diskPrefix: "transformer_blocks.0.",
                                               modulePrefix: "transformerBlocks.0.",
                                               diskKeys: Flux2Weights.doubleBlockDiskKeys())
        assertBijection(moduleKeys: moduleKeys, mapped: Set(map.values))
    }

    func testSingleBlockKeyCoverage() throws {
        let block = Flux2SingleTransformerBlock(dim: dim, numHeads: heads, headDim: headDim)
        quantize(model: block, groupSize: 64, bits: 4)
        let moduleKeys = Set(block.parameters().flattened().map { $0.0 })
        let map = try Flux2Weights.blockKeyMap(diskPrefix: "single_transformer_blocks.0.",
                                               modulePrefix: "singleTransformerBlocks.0.",
                                               diskKeys: Flux2Weights.singleBlockDiskKeys())
        assertBijection(moduleKeys: moduleKeys, mapped: Set(map.values))
    }

    func testDiskKeyCounts() {
        // 12 quantized linears ×3 + 4 norms = 40 (double); 2 ×3 + 2 = 8 (single).
        XCTAssertEqual(Flux2Weights.doubleBlockDiskKeys().count, 40)
        XCTAssertEqual(Flux2Weights.singleBlockDiskKeys().count, 8)
        // 5 double + 20 single = 25 blocks → 5×40 + 20×8 = 360 block tensors (+27 non-block = 387 total).
        XCTAssertEqual(5 * 40 + 20 * 8, 360)
    }
}
