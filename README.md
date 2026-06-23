# flux2-diffusion-engine

A **macOS-only** facade that wraps [`flux-2-swift-mlx`](https://github.com/nanguoyu/flux-2-swift-mlx)'s
monolithic `Flux2Pipeline` behind [`swift-diffusion-core`](https://github.com/nanguoyu/swift-diffusion-core)'s
`DiffusionEngine`.

FLUX.2 exposes only a whole-pipeline `generateTextToImage(...)` (no per-block streaming) and the
package targets macOS 15+, so it can't use the block-streaming engine in the core and can't build
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

**Fix in flux-2-swift-mlx** (either): move `Training/` to its own target so inference-only
consumers skip it (recommended — also faster to build), or update `ResumableAdamW` for the
new optimizer API. After that, this package compiles.

## License

Apache-2.0 (intended).
