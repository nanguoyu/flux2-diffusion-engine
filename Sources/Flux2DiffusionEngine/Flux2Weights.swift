@preconcurrency import MLX
import MLXNN
import DiffusionCore
import Flux2Core

/// Per-block streaming loader for the pre-quantized 4-bit Klein checkpoint.
///
/// The engine streams the transformer one block at a time: build a fresh `Flux2TransformerBlock` /
/// `Flux2SingleTransformerBlock`, fill it from the `WeightSource`, run it, drop it. This loader does
/// the fill. It reconstructs ONE block's on-disk keys (the mflux layout, the same one
/// `Flux2Core.PreQuantizedKeyMatchTests` validates), maps each to the Swift module's parameter path
/// via the VALIDATED `Flux2WeightLoader.mapMLXQuantizedTransformerKey`, quantizes the block, and pulls
/// each parameter by exact key. Disk→module key coverage is verified offline (no checkpoint) in
/// `Flux2WeightsTests` — the per-block analogue of the whole-model `notFound == 0` gate.
public enum Flux2Weights {

    /// Block-relative on-disk key suffixes (sans the `transformer_blocks.{i}.` prefix) for a
    /// DOUBLE-stream block: 12 quantized Linears (.weight/.scales/.biases) + 4 RMSNorms (.weight).
    public static func doubleBlockDiskKeys() -> [String] {
        var keys: [String] = []
        for l in ["attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
                  "attn.to_k", "attn.to_out", "attn.to_q", "attn.to_v",
                  "ff.linear_in", "ff.linear_out", "ff_context.linear_in", "ff_context.linear_out"] {
            keys += ["\(l).weight", "\(l).scales", "\(l).biases"]
        }
        for n in ["attn.norm_added_k", "attn.norm_added_q", "attn.norm_k", "attn.norm_q"] {
            keys.append("\(n).weight")
        }
        return keys
    }

    /// SINGLE-stream block: 2 quantized Linears (.weight/.scales/.biases) + 2 RMSNorms (.weight).
    public static func singleBlockDiskKeys() -> [String] {
        var keys: [String] = []
        for l in ["attn.to_out", "attn.to_qkv_mlp_proj"] { keys += ["\(l).weight", "\(l).scales", "\(l).biases"] }
        for n in ["attn.norm_k", "attn.norm_q"] { keys.append("\(n).weight") }
        return keys
    }

    public enum LoadError: Error, CustomStringConvertible {
        case unmappedKey(String)
        case missingTensor(String)
        public var description: String {
            switch self {
            case .unmappedKey(let k): return "Flux2Weights: disk key '\(k)' did not map under the expected block prefix"
            case .missingTensor(let k): return "Flux2Weights: WeightSource is missing tensor '\(k)'"
            }
        }
    }

    /// The disk(block-relative) → module(block-relative) parameter-path map for a block, derived from
    /// the validated key mapper. Throws if a key fails to map under the block prefix (a layout bug).
    public static func blockKeyMap(diskPrefix: String, modulePrefix: String, diskKeys: [String]) throws -> [String: String] {
        var map: [String: String] = [:]
        for dk in diskKeys {
            let mapped = Flux2WeightLoader.mapMLXQuantizedTransformerKey(diskPrefix + dk)
            guard mapped.hasPrefix(modulePrefix) else { throw LoadError.unmappedKey(dk) }
            map[dk] = String(mapped.dropFirst(modulePrefix.count))
        }
        return map
    }

    public static func loadDoubleBlock(_ index: Int, from source: WeightSource, into block: Flux2TransformerBlock,
                                       groupSize: Int = 64, bits: Int = 4) throws {
        try loadBlock(diskPrefix: "transformer_blocks.\(index).", modulePrefix: "transformerBlocks.\(index).",
                      diskKeys: doubleBlockDiskKeys(), from: source, into: block, groupSize: groupSize, bits: bits)
    }

    public static func loadSingleBlock(_ index: Int, from source: WeightSource, into block: Flux2SingleTransformerBlock,
                                       groupSize: Int = 64, bits: Int = 4) throws {
        try loadBlock(diskPrefix: "single_transformer_blocks.\(index).", modulePrefix: "singleTransformerBlocks.\(index).",
                      diskKeys: singleBlockDiskKeys(), from: source, into: block, groupSize: groupSize, bits: bits)
    }

    static func loadBlock(diskPrefix: String, modulePrefix: String, diskKeys: [String],
                          from source: WeightSource, into block: Module, groupSize: Int, bits: Int) throws {
        let map = try blockKeyMap(diskPrefix: diskPrefix, modulePrefix: modulePrefix, diskKeys: diskKeys)
        // Quantize every Linear (all block projections are 4-bit on disk), exactly as the validated
        // whole-model load does, so the .scales/.biases destinations exist before update.
        quantize(model: block, groupSize: groupSize, bits: bits)
        var collected: [String: MLXArray] = [:]
        for (dk, modulePath) in map {
            guard let t = try? source.tensor(TensorKey(diskPrefix + dk)) else {
                throw LoadError.missingTensor(diskPrefix + dk)
            }
            collected[modulePath] = t
        }
        block.update(parameters: ModuleParameters.unflattened(collected))
    }
}
