// swift-tools-version: 5.10
import PackageDescription

// flux2-diffusion-engine — a macOS-only facade that wraps the monolithic flux-2-swift-mlx
// `Flux2Pipeline` behind swift-diffusion-core's `DiffusionEngine`. FLUX.2 cannot be block-
// streamed (no per-block API) and the package is macOS-15-only, so it lives outside the core
// (which targets iOS too) and is linked only by the macOS app target.
let package = Package(
    name: "flux2-diffusion-engine",
    platforms: [.macOS("15.0")],
    products: [
        .library(name: "Flux2DiffusionEngine", targets: ["Flux2DiffusionEngine"]),
    ],
    dependencies: [
        .package(url: "https://github.com/nanguoyu/swift-diffusion-core", branch: "main"),
        .package(url: "https://github.com/nanguoyu/flux-2-swift-mlx", branch: "main"),
    ],
    targets: [
        .target(
            name: "Flux2DiffusionEngine",
            dependencies: [
                .product(name: "DiffusionCore", package: "swift-diffusion-core"),
                .product(name: "Flux2Core", package: "flux-2-swift-mlx"),
            ]
        ),
    ]
)
