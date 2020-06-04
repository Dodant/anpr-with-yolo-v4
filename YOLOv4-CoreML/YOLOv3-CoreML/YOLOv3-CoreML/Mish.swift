//
//  Mish.swift
//  YOLOv3-CoreML
//
//  Created by Junggyun Oh on 2020/06/04.
//  Copyright © 2020 MachineThink. All rights reserved.
//

import Foundation
import CoreML
import Accelerate

@objc(Mish) class Mish: NSObject, MLCustomLayer {
    let mishPipeline: MTLComputePipelineState

    required init(parameters: [String : Any]) throws {

      // Create the Metal compute kernels.
      let device = MTLCreateSystemDefaultDevice()!
      let library = device.makeDefaultLibrary()!
      let mishFunction = library.makeFunction(name: "mish")!
      mishPipeline = try! device.makeComputePipelineState(function: mishFunction)

      super.init()
    }

    func setWeightData(_ weights: [Data]) throws {
    }

    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws
       -> [[NSNumber]] {
        return inputShapes
    }

    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        for i in 0..<inputs.count {
            let input = inputs[i]
            let output = outputs[i]

            let count = input.count
            let iptr = UnsafeMutablePointer<Float>(OpaquePointer(input.dataPointer))
            let optr = UnsafeMutablePointer<Float>(OpaquePointer(output.dataPointer))

            // output = exp(input)
            var countAsInt32 = Int32(count)
            vvexpf(optr, iptr, &countAsInt32)

            // output = 1 + exp(input)
            var one: Float = 1
            let vdspLength = vDSP_Length(count)
            vDSP_vsadd(
                optr, 1,
                &one,
                optr, 1,
                vdspLength)

            // output = ln(1 + exp(input))
            vvlogf(optr, optr, &countAsInt32)

            // output = tanh(ln(1 + exp(input)))
            vvtanhf(optr, optr, &countAsInt32)

            // output = x * tanh(ln(1 + exp(input)))
            vDSP_vmul(optr, 1, iptr, 1, optr, 1, vdspLength)
        }
    }
    func encode(commandBuffer: MTLCommandBuffer, inputs: [MTLTexture], outputs: [MTLTexture]) throws {

        // This method gets called when the model runs on the GPU. It is optional,
        // but recommended that you implemented it -- for the best possible speed!
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            for i in 0..<inputs.count {
                encoder.setTexture(inputs[i], index: 0)
                encoder.setTexture(outputs[i], index: 1)
                encoder.dispatch(pipeline: mishPipeline, texture: inputs[i])
                encoder.endEncoding()
            }
        }
    }
}
extension MTLComputeCommandEncoder {
    public func dispatch(pipeline: MTLComputePipelineState, texture: MTLTexture) {
        let w = pipeline.threadExecutionWidth
        let h = pipeline.maxTotalThreadsPerThreadgroup / w
        let threadGroupSize = MTLSizeMake(w, h, 1)

        let threadGroups = MTLSizeMake(
            (texture.width       + threadGroupSize.width  - 1) / threadGroupSize.width,
            (texture.height      + threadGroupSize.height - 1) / threadGroupSize.height,
            (texture.arrayLength + threadGroupSize.depth  - 1) / threadGroupSize.depth)

        setComputePipelineState(pipeline)
        dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    }
}
