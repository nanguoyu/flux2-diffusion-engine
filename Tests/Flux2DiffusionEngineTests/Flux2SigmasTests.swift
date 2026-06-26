// Flux2SigmasTests — pin FLUX.2's architecture-owned sigma schedule to its exact values.
//
// The streamed 1024 path is validated against the resident 512 path (the parity gate), which only
// holds if both denoise on the identical sigma curve. These constants were computed from FLUX.2's
// FlowMatchEulerScheduler (empirical mu, exponential time-shift). The core engine's fixed-shift
// sampler gives a completely different step-3 sigma (~0.001 vs ~0.717), which would produce a
// different — typically washed-out/black — image; this test guards that we never regress to it.

import XCTest
@testable import Flux2DiffusionEngine

final class Flux2SigmasTests: XCTestCase {

    private func assertClose(_ got: [Float], _ want: [Float], _ tol: Float = 1e-4, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(got.count, want.count, "sigma count", file: file, line: line)
        for (g, w) in zip(got, want) {
            XCTAssertEqual(g, w, accuracy: tol, "sigma \(g) vs \(w)", file: file, line: line)
        }
    }

    func testImageSeqLen() {
        XCTAssertEqual(Flux2Sigmas.imageSeqLen(width: 512, height: 512), 1024)
        XCTAssertEqual(Flux2Sigmas.imageSeqLen(width: 1024, height: 1024), 4096)
    }

    func test512Schedule() {
        // imageSeqLen 1024, mu ≈ 2.0306897
        assertClose(Flux2Sigmas.schedule(width: 512, height: 512, steps: 4),
                    [1.0, 0.9580854, 0.8839818, 0.7174966, 0.0])
    }

    func test1024Schedule() {
        // imageSeqLen 4096, mu ≈ 2.2911799
        assertClose(Flux2Sigmas.schedule(width: 1024, height: 1024, steps: 4),
                    [1.0, 0.9673840, 0.9081439, 0.7672000, 0.0])
    }

    func testScheduleHasStepsPlusOneSigmas() {
        for steps in [4, 8, 20] {
            XCTAssertEqual(Flux2Sigmas.schedule(width: 512, height: 512, steps: steps).count, steps + 1)
        }
    }

    func testTrailingSigmaIsZero() {
        XCTAssertEqual(Flux2Sigmas.schedule(width: 1024, height: 1024, steps: 4).last, 0.0)
    }

    /// The schedule must differ sharply from a fixed-shift schedule (the core sampler default), or the
    /// whole reason for the architecture-owned hook is moot. Step 3 is the clearest divergence.
    func testDivergesFromFixedShift() {
        let s = Flux2Sigmas.schedule(width: 512, height: 512, steps: 4)
        XCTAssertGreaterThan(s[3], 0.5, "FLUX step-3 sigma must be ~0.717, nowhere near the fixed-shift ~0.001")
    }
}
