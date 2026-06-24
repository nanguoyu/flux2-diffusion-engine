// swift-tools-version: 5.10
import PackageDescription

// flux2-diffusion-engine — a facade that wraps the monolithic flux-2-swift-mlx `Flux2Pipeline`
// behind swift-diffusion-core's `DiffusionEngine`. FLUX.2 owns its own denoise loop and weight
// loading (no per-block streaming API), so it runs as a resident (Mac) or two-phase (iPhone)
// pipeline rather than via the core's block streamer. Builds for macOS and iOS: on iPhone it loads
// the pre-quantized 4-bit Klein checkpoint, which fits the phone's memory budget with no load spike.
let package = Package(
    name: "flux2-diffusion-engine",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "Flux2DiffusionEngine", targets: ["Flux2DiffusionEngine"]),
    ],
    dependencies: [
        .package(url: "https://github.com/nanguoyu/swift-diffusion-core", branch: "main"),
        .package(url: "https://github.com/nanguoyu/flux-2-swift-mlx", branch: "main"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.0"),
    ],
    targets: [
        .target(
            name: "Flux2DiffusionEngine",
            dependencies: [
                .product(name: "DiffusionCore", package: "swift-diffusion-core"),
                .product(name: "Flux2Core", package: "flux-2-swift-mlx"),
                .product(name: "FluxTextEncoders", package: "flux-2-swift-mlx"),
            ]
        ),
        // Minimal CLI: `swift run flux2-demo "your prompt"` — loads FLUX.2 Klein 4B (downloads
        // weights on first run) and writes flux-out.png. Requires a Metal GPU (macOS).
        .executableTarget(
            name: "flux2-demo",
            dependencies: [
                "Flux2DiffusionEngine",
                .product(name: "DiffusionCore", package: "swift-diffusion-core"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            linkerSettings: [
                // MLX's Cmlx links libc++ via `@rpath/libc++.1.dylib`, but `swift run` on
                // Xcode 26 / Swift 6.2 embeds rpaths that don't include the system lib dir, so
                // dyld can't resolve it. Add /usr/lib (where libc++ lives in the shared cache).
                .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "/usr/lib"]),
            ]
        ),
    ]
)
