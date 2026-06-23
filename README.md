# flux2-diffusion-engine

A **macOS-only** facade that wraps [`flux-2-swift-mlx`](https://github.com/nanguoyu/flux-2-swift-mlx)'s
monolithic `Flux2Pipeline` behind [`swift-diffusion-core`](https://github.com/nanguoyu/swift-diffusion-core)'s
`DiffusionEngine`.

FLUX.2 exposes only a whole-pipeline `generateTextToImage(...)` (no per-block streaming) and the
package targets macOS 14+, so it can't use the block-streaming engine in the core and can't build
for iOS. It therefore lives in its own package, linked only by the macOS app target.

```swift
let engine: any DiffusionEngine = Flux2FacadeEngine()
try await engine.load(model, variant: variant, source: source, progress: { _ in })
let image = try await engine.generate(request, progress: { _ in })
```

## Status

The facade is written against the documented `Flux2Pipeline` API but is **not yet
compile-verified**, because flux-2-swift-mlx `main` does not currently build against
mlx-swift 0.31.4: its training optimizer `ResumableAdamW` overrides an `AdamW` API that
changed (`applySingle` state `TupleState` → `AdamState`, and the init is no longer an
overridable designated initializer). Since `Training/` lives in the same `Flux2Core`
module, this blocks inference consumers too.

**Fixed** via [flux-2-swift-mlx PR #1](https://github.com/nanguoyu/flux-2-swift-mlx/pull/1)
(`ResumableAdamW` updated for mlx-swift 0.31's `AdamState`). This package now compiles and
**runs** (validated: real FLUX.2 Klein image generated).

### Running `flux2-demo`
- **Via Xcode** (recommended): `xed .`, run the `flux2-demo` scheme. Xcode compiles MLX's
  Metal shader lib automatically.
- **Via `swift run`** on Xcode 26 needs two one-time fixes the CLI build doesn't do itself:
  1. the executable target already adds `/usr/lib` to its rpath (resolves `@rpath/libc++.1.dylib`);
  2. `swift build` does **not** compile MLX's `default.metallib`. Build once in Xcode, then copy
     its `mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` to `mlx.metallib` beside the
     `swift run` executable (`.build/<triple>/debug/mlx.metallib`). After that, `swift run flux2-demo "…"` works.

## License

Apache-2.0 (intended).
