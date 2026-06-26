import Foundation
import CoreGraphics
import ImageIO
import Metal
import MetalKit
import MetalFX

// Evaluate MTLFXSpatialScaler (MetalFX, Metal 3) as a "fast 1024" route: instead of natively diffusing
// 1024 (4× FLOPs, hot), diffuse 512 and upscale 512→1024 on the GPU (~ms, cool).
//
// Controlled test (so PSNR is meaningful — same content): take a NATIVE 1024 image, downsample to 512,
// then MetalFX-upscale back to 1024 and compare to the original. This isolates how much true 1024
// detail the upscaler can reconstruct vs a bicubic baseline. Run: swift run flux2-demo --upscale [png]

private func loadCG(_ path: String) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil as CFDictionary?)
}

/// Resize a CGImage to w×h via CoreGraphics (high-quality interpolation = bicubic-ish).
private func resize(_ cg: CGImage, _ w: Int, _ h: Int) -> CGImage? {
    guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.interpolationQuality = .high
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    return ctx.makeImage()
}

func runUpscale() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    let inPath = args.first(where: { $0.hasSuffix(".png") }) ?? "parity-resident.png"
    guard let native = loadCG(inPath), native.width == native.height else {
        print("need a square native PNG (default parity-resident.png — run `--parity --size 1024` first)"); return
    }
    let N = native.width            // e.g. 1024
    let half = N / 2                // 512
    print("MetalFX eval — native \(N)×\(N) from \(inPath); downsample→\(half), upscale→\(N)")

    guard let down = resize(native, half, half) else { print("downsample failed"); return }
    try writePNG(down, to: URL(fileURLWithPath: "upscale-down\(half).png"))

    // Bicubic baseline: down → N via CoreGraphics.
    guard let bicubic = resize(down, N, N) else { print("bicubic failed"); return }
    try writePNG(bicubic, to: URL(fileURLWithPath: "upscale-bicubic\(N).png"))

    // MetalFX spatial upscale: down → N.
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        print("no Metal device"); return
    }
    let loader = MTKTextureLoader(device: device)
    let inTex = try loader.newTexture(cgImage: down, options: [
        .SRGB: NSNumber(value: false),
        .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
        .textureStorageMode: NSNumber(value: MTLStorageMode.shared.rawValue),
    ])
    let outDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: inTex.pixelFormat, width: N, height: N, mipmapped: false)
    outDesc.usage = [.shaderRead, .shaderWrite, .renderTarget]
    outDesc.storageMode = .shared
    guard let outTex = device.makeTexture(descriptor: outDesc) else { print("out texture failed"); return }

    let sd = MTLFXSpatialScalerDescriptor()
    sd.inputWidth = half; sd.inputHeight = half
    sd.outputWidth = N; sd.outputHeight = N
    sd.colorTextureFormat = inTex.pixelFormat
    sd.outputTextureFormat = outTex.pixelFormat
    sd.colorProcessingMode = .perceptual
    guard let scaler = sd.makeSpatialScaler(device: device) else { print("MetalFX scaler unavailable"); return }
    scaler.colorTexture = inTex
    scaler.outputTexture = outTex
    scaler.inputContentWidth = half
    scaler.inputContentHeight = half
    guard let cb = queue.makeCommandBuffer() else { print("no command buffer"); return }
    scaler.encode(commandBuffer: cb)
    cb.commit(); cb.waitUntilCompleted()

    guard let metalfx = textureToCG(outTex) else { print("readback failed"); return }
    try writePNG(metalfx, to: URL(fileURLWithPath: "upscale-metalfx\(N).png"))

    let (bd, bp) = compareImages(native, bicubic)
    let (md, mp) = compareImages(native, metalfx)
    print(String(format: "vs native (%dpx):  bicubic PSNR=%.2f dB maxΔ=%d   |   MetalFX PSNR=%.2f dB maxΔ=%d", N, bp, bd, mp, md))
    print("→ higher PSNR = closer to true native detail. Wrote upscale-{down,bicubic,metalfx}.png for visual inspection.")
}

private func textureToCG(_ tex: MTLTexture) -> CGImage? {
    let w = tex.width, h = tex.height
    var bytes = [UInt8](repeating: 0, count: w * h * 4)
    bytes.withUnsafeMutableBytes {
        tex.getBytes($0.baseAddress!, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
    }
    let isBGRA = tex.pixelFormat == .bgra8Unorm || tex.pixelFormat == .bgra8Unorm_srgb
    let bitmapInfo: UInt32 = isBGRA
        ? (CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        : (CGImageAlphaInfo.premultipliedLast.rawValue)
    guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
    return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                   space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
                   provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
}
