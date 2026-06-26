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

/// A single FLUX.2 transformer block behind the core engine's `StreamableBlock` seam. Two modes:
///
/// - STREAMING (the iPhone 1024 path): constructed with just an index + dims; `load(from:)` builds a
///   fresh `Flux2TransformerBlock`/`Flux2SingleTransformerBlock` and fills it via `Flux2Weights`,
///   `release()` drops it. The engine loops load → run → eval → release → clearCache so only one
///   block is resident at a time.
/// - RESIDENT (Mac / the parity test): constructed with a pre-built block; `load`/`release` are
///   no-ops and it forwards to that block every step.
///
/// `callAsFunction` runs the block through the SAME static `runDouble`/`runSingle` the decomposition
/// test validates, reading the per-step context from the shared holder — so the streamed result is
/// identical to the monolithic forward.
public final class Flux2StreamableBlock: StreamableBlock {
    public let index: Int
    public let approximateBytes: Int64
    private let isDouble: Bool
    private let blockIndexInType: Int
    private let dim: Int, heads: Int, headDim: Int
    private let holder: Flux2StreamHolder

    private var doubleBlock: Flux2TransformerBlock?
    private var singleBlock: Flux2SingleTransformerBlock?

    /// Streaming-mode init: weights are loaded per step from the source.
    public init(index: Int, isDouble: Bool, blockIndexInType: Int,
                dim: Int, heads: Int, headDim: Int, approximateBytes: Int64, holder: Flux2StreamHolder) {
        self.index = index
        self.isDouble = isDouble
        self.blockIndexInType = blockIndexInType
        self.dim = dim; self.heads = heads; self.headDim = headDim
        self.approximateBytes = approximateBytes
        self.holder = holder
    }

    /// Resident-mode init: a pre-built block; load/release are no-ops.
    public init(index: Int, resident block: Flux2TransformerBlock, holder: Flux2StreamHolder) {
        self.index = index; self.isDouble = true; self.blockIndexInType = index
        self.dim = 0; self.heads = 0; self.headDim = 0
        self.approximateBytes = 0; self.holder = holder
        self.doubleBlock = block
    }
    public init(index: Int, resident block: Flux2SingleTransformerBlock, holder: Flux2StreamHolder) {
        self.index = index; self.isDouble = false; self.blockIndexInType = index
        self.dim = 0; self.heads = 0; self.headDim = 0
        self.approximateBytes = 0; self.holder = holder
        self.singleBlock = block
    }

    public func load(from source: WeightSource) throws {
        if isDouble {
            guard doubleBlock == nil else { return }   // resident mode: already built
            let block = Flux2TransformerBlock(dim: dim, numHeads: heads, headDim: headDim)
            try Flux2Weights.loadDoubleBlock(blockIndexInType, from: source, into: block)
            doubleBlock = block
        } else {
            guard singleBlock == nil else { return }
            let block = Flux2SingleTransformerBlock(dim: dim, numHeads: heads, headDim: headDim)
            try Flux2Weights.loadSingleBlock(blockIndexInType, from: source, into: block)
            singleBlock = block
        }
    }

    public func callAsFunction(_ x: MLXArray, conditioning: Conditioning, timestep: MLXArray) -> MLXArray {
        guard let ctx = holder.context else { return x }   // embed must run first each step
        if isDouble {
            return Flux2Transformer2DModel.runDouble(doubleBlock!, hidden: x, context: ctx)
        } else {
            return Flux2Transformer2DModel.runSingle(singleBlock!, hidden: x, context: ctx)
        }
    }

    public func release() {
        // Resident blocks are kept (dims are 0 in resident mode, so a reload would be impossible);
        // streaming blocks drop their weights so the engine's clearCache reclaims the memory.
        if dim != 0 { doubleBlock = nil; singleBlock = nil }
    }
}

/// FLUX.2 denoiser for the core engine's block-streaming loop. Holds a lightweight 0-block SHELL
/// transformer (the shared submodules, resident) that runs `streamEmbed`/`streamUnembed`, plus the 25
/// streamable blocks. `embed` packs `[txt;img]` and stashes the per-step context in the holder; the
/// blocks read it; `unembed` slices the image tokens back out and projects to the velocity.
public final class Flux2Denoiser: Denoiser {
    public let blocks: [any StreamableBlock]
    private let shell: Flux2Transformer2DModel
    private let holder: Flux2StreamHolder
    private let imgIds: MLXArray
    private let txtIds: MLXArray

    public init(shell: Flux2Transformer2DModel, holder: Flux2StreamHolder,
                blocks: [any StreamableBlock], imgIds: MLXArray, txtIds: MLXArray) {
        self.shell = shell
        self.holder = holder
        self.blocks = blocks
        self.imgIds = imgIds
        self.txtIds = txtIds
    }

    public func embed(latent: MLXArray, timestep: MLXArray, conditioning: Conditioning) -> MLXArray {
        let (hidden, ctx) = shell.streamEmbed(hiddenStates: latent,
                                              encoderHiddenStates: conditioning.embeddings,
                                              timestep: timestep, guidance: nil,
                                              imgIds: imgIds, txtIds: txtIds)
        holder.context = ctx
        return hidden
    }

    public func unembed(_ hidden: MLXArray) -> MLXArray {
        guard let ctx = holder.context else { return hidden }
        return shell.streamUnembed(hidden: hidden, context: ctx)
    }
}
