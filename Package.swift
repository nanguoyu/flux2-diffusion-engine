// swift-tools-version: 5.10
import PackageDescription

// flux2-diffusion-engine — a macOS-only facade that wraps the monolithic flux-2-swift-mlx
// `Flux2Pipeline` behind swift-diffusion-core's `DiffusionEngine`. FLUX.2 cannot be block-
// streamed (no per-block API), so it lives outside the core (which targets iOS too) and is linked
// only by the macOS app target. macOS-only but **macOS 14+** (matches flux-2-swift-mlx), not 15.
let package = Package(
    name: "flux2-diffusion-engine",
    platforms: [.macOS(.v14)],
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
