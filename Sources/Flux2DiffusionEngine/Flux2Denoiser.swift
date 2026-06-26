@preconcurrency import MLX
import DiffusionCore
import Flux2Core

/// Per-step context shared between the denoiser's `embed`/`unembed` and every block. `embed`
/// recomputes it each step (it depends on the timestep) and stashes it here; the blocks and
/// `unembed` read it. A reference type so the denoiser and all 25 blocks see the same instance.
public final class Flux2StreamHolder: @unchecked Sendable {
    public var context: Flux2Transformer2DModel.Flux2StreamContext?
    public init() {}
}

/// A single FLUX.2 transformer block behind the core engine's `StreamableBlock` seam.
///
/// REUSE-SHELL (the production path): one quantized `Flux2TransformerBlock` shell (and one single-block
/// shell) is built + quantized ONCE and shared across all blocks of that type. Each block only swaps
/// the packed params in via `Flux2Weights.updateXBlock` — collapsing the ~100 per-image `quantize()`
/// GPU passes (≈ the pread I/O in magnitude) to one. The packed-param update happens in
/// `callAsFunction`, not `load`, so the engine's resident path (which loads every block before running
/// any) doesn't clobber the shell: MLX captures the param ARRAYS at the forward call and `update`
/// REPLACES references, so each block's graph keeps its own weights until it's evaluated.
///
/// RESIDENT (the wiring test): a pre-built block; `load`/`release` are no-ops and it forwards to it.
///
/// `callAsFunction` runs through the same static `runDouble`/`runSingle` the decomposition test
/// validates, reading the per-step context from the shared holder.
public final class Flux2StreamableBlock: StreamableBlock {
    public let index: Int
    public let approximateBytes: Int64
    private let isDouble: Bool
    private let blockIndexInType: Int
    private let holder: Flux2StreamHolder

    // reuse-shell mode: the shared shell for this block's type (one is non-nil) + the stashed source.
    private let doubleShell: Flux2TransformerBlock?
    private let singleShell: Flux2SingleTransformerBlock?
    private var source: WeightSource?

    // resident mode: a pre-built block (wiring test).
    private let residentDouble: Flux2TransformerBlock?
    private let residentSingle: Flux2SingleTransformerBlock?

    /// Reuse-shell init for a DOUBLE block.
    public init(doubleIndex: Int, blockIndexInType: Int, shell: Flux2TransformerBlock,
                approximateBytes: Int64, holder: Flux2StreamHolder) {
        self.index = doubleIndex; self.isDouble = true; self.blockIndexInType = blockIndexInType
        self.doubleShell = shell; self.singleShell = nil
        self.residentDouble = nil; self.residentSingle = nil
        self.approximateBytes = approximateBytes; self.holder = holder
    }

    /// Reuse-shell init for a SINGLE block.
    public init(singleIndex: Int, blockIndexInType: Int, shell: Flux2SingleTransformerBlock,
                approximateBytes: Int64, holder: Flux2StreamHolder) {
        self.index = singleIndex; self.isDouble = false; self.blockIndexInType = blockIndexInType
        self.doubleShell = nil; self.singleShell = shell
        self.residentDouble = nil; self.residentSingle = nil
        self.approximateBytes = approximateBytes; self.holder = holder
    }

    /// Resident init (a pre-built block; load/release no-op).
    public init(index: Int, resident block: Flux2TransformerBlock, holder: Flux2StreamHolder) {
        self.index = index; self.isDouble = true; self.blockIndexInType = index
        self.doubleShell = nil; self.singleShell = nil
        self.residentDouble = block; self.residentSingle = nil
        self.approximateBytes = 0; self.holder = holder
    }
    public init(index: Int, resident block: Flux2SingleTransformerBlock, holder: Flux2StreamHolder) {
        self.index = index; self.isDouble = false; self.blockIndexInType = index
        self.doubleShell = nil; self.singleShell = nil
        self.residentDouble = nil; self.residentSingle = block
        self.approximateBytes = 0; self.holder = holder
    }

    public func load(from source: WeightSource) throws {
        // Stash the source; the per-block param swap happens in callAsFunction so loading all blocks
        // upfront (resident path) doesn't leave the shared shell holding only the last block's weights.
        self.source = source
        if residentDouble != nil || residentSingle != nil { return }   // resident mode: nothing to fetch
        // Probe this block's first weight so a missing/corrupt download throws HERE (load is a throwing
        // context) instead of callAsFunction's swap silently failing and running the shared shell's
        // PREVIOUS block weights — a wrong image with no error.
        let prefix = isDouble ? "transformer_blocks.\(blockIndexInType)." : "single_transformer_blocks.\(blockIndexInType)."
        let keys = isDouble ? Flux2Weights.doubleBlockDiskKeys() : Flux2Weights.singleBlockDiskKeys()
        if let firstKey = keys.first { _ = try source.tensor(TensorKey(prefix + firstKey)) }
    }

    public func callAsFunction(_ x: MLXArray, conditioning: Conditioning, timestep: MLXArray) -> MLXArray {
        guard let ctx = holder.context else { return x }   // embed must run first each step
        if isDouble {
            if let r = residentDouble { return Flux2Transformer2DModel.runDouble(r, hidden: x, context: ctx) }
            let shell = doubleShell!
            if let source { try? Flux2Weights.updateDoubleBlock(blockIndexInType, from: source, into: shell) }
            return Flux2Transformer2DModel.runDouble(shell, hidden: x, context: ctx)
        } else {
            if let r = residentSingle { return Flux2Transformer2DModel.runSingle(r, hidden: x, context: ctx) }
            let shell = singleShell!
            if let source { try? Flux2Weights.updateSingleBlock(blockIndexInType, from: source, into: shell) }
            return Flux2Transformer2DModel.runSingle(shell, hidden: x, context: ctx)
        }
    }

    public func release() {
        // The shells persist (reuse); just drop the stashed source. The current block's weight arrays
        // are freed by the engine's eval+clearCache once its hidden state is materialized.
        source = nil
    }
}

/// FLUX.2 denoiser for the core engine's block-streaming loop. Holds a lightweight 0-block SHELL
/// transformer (the shared submodules, resident) that runs `streamEmbed`/`streamUnembed`, the two
/// reused block shells, and the 25 streamable blocks. `embed` packs `[txt;img]` and stashes the
/// per-step context in the holder; the blocks read it; `unembed` slices the image tokens back out.
public final class Flux2Denoiser: Denoiser {
    public let blocks: [any StreamableBlock]
    private let shell: Flux2Transformer2DModel
    private let doubleShell: Flux2TransformerBlock?
    private let singleShell: Flux2SingleTransformerBlock?
    private let holder: Flux2StreamHolder
    private let imgIds: MLXArray
    private let txtIds: MLXArray

    public init(shell: Flux2Transformer2DModel, holder: Flux2StreamHolder,
                blocks: [any StreamableBlock], imgIds: MLXArray, txtIds: MLXArray,
                doubleShell: Flux2TransformerBlock? = nil, singleShell: Flux2SingleTransformerBlock? = nil) {
        self.shell = shell
        self.holder = holder
        self.blocks = blocks
        self.imgIds = imgIds
        self.txtIds = txtIds
        self.doubleShell = doubleShell
        self.singleShell = singleShell
    }

    public func embed(latent: MLXArray, timestep: MLXArray, conditioning: Conditioning) -> MLXArray {
        // The core engine passes a SCALAR timestep (`MLXArray(Float)`), but FLUX's time embedder
        // expects a `[B]` shape — the resident pipeline feeds `MLXArray([sigma])`. A scalar produces a
        // differently-shaped `temb` and a divergent (still coherent) denoise trajectory, so normalize
        // to `[1]` to match the resident path exactly.
        let ts = timestep.ndim == 0 ? timestep.reshaped([1]) : timestep
        let (hidden, ctx) = shell.streamEmbed(hiddenStates: latent,
                                              encoderHiddenStates: conditioning.embeddings,
                                              timestep: ts, guidance: nil,
                                              imgIds: imgIds, txtIds: txtIds)
        holder.context = ctx
        return hidden
    }

    public func unembed(_ hidden: MLXArray) -> MLXArray {
        guard let ctx = holder.context else { return hidden }
        return shell.streamUnembed(hidden: hidden, context: ctx)
    }
}
