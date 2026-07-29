import {
  GO_MODEL_KERNEL_RADIUS,
  GO_MODEL_KERNEL_SIZE,
  GO_MODEL_INT4_FORMAT,
  GO_MODEL_INT4_NIBBLE_BITS,
  GO_MODEL_INT4_NIBBLE_MASK,
  GO_MODEL_INT4_VALUES_PER_BYTE,
  GO_MODEL_INT4_ZERO_POINT,
  GO_MODEL_INT8_FORMAT,
  GO_MODEL_NESTED_RESIDUAL_BLOCK_KIND,
  GO_MODEL_POLICY_KERNEL_SIZE,
  GO_MODEL_STANDARD_RESIDUAL_BLOCK_KIND,
  GO_MODEL_VALUE_KERNEL_SIZE,
  GO_MODEL_VERSION,
} from "./runtime-constants";

const getElementCount = (shape: number[]) =>
  shape.reduce((elementCount, dimension) => elementCount * dimension, 1);

const isSupportedManifest = (manifest: GoModelManifest) => {
  const residualBlockKind =
    manifest.architecture.residualBlockKind ?? GO_MODEL_STANDARD_RESIDUAL_BLOCK_KIND;
  const globalResidualBlockInterval = manifest.architecture.globalResidualBlockInterval ?? 0;
  const globalResidualHiddenChannelCount =
    manifest.architecture.globalResidualHiddenChannelCount ?? 0;
  const isResidualBlockSupported =
    residualBlockKind === GO_MODEL_STANDARD_RESIDUAL_BLOCK_KIND ||
    (residualBlockKind === GO_MODEL_NESTED_RESIDUAL_BLOCK_KIND &&
      (manifest.architecture.bottleneckChannelCount ?? 0) > 0);
  const isGlobalResidualConfigurationSupported =
    (globalResidualBlockInterval === 0 && globalResidualHiddenChannelCount === 0) ||
    (residualBlockKind === GO_MODEL_NESTED_RESIDUAL_BLOCK_KIND &&
      globalResidualBlockInterval > 0 &&
      globalResidualHiddenChannelCount > 0);

  return (
    (manifest.format === GO_MODEL_INT4_FORMAT || manifest.format === GO_MODEL_INT8_FORMAT) &&
    manifest.version === GO_MODEL_VERSION &&
    isResidualBlockSupported &&
    isGlobalResidualConfigurationSupported
  );
};

const applyRelu = (values: Float32Array) => {
  for (let valueIndex = 0; valueIndex < values.length; valueIndex += 1) {
    values[valueIndex] = Math.max(0, values[valueIndex]);
  }

  return values;
};

const addAndApplyRelu = (leftValues: Float32Array, rightValues: Float32Array) => {
  for (let valueIndex = 0; valueIndex < leftValues.length; valueIndex += 1) {
    leftValues[valueIndex] = Math.max(0, leftValues[valueIndex] + rightValues[valueIndex]);
  }

  return leftValues;
};

const addChannelBias = (values: Float32Array, channelCount: number, biases: Float32Array) => {
  for (let valueIndex = 0; valueIndex < values.length; valueIndex += 1) {
    values[valueIndex] += biases[valueIndex % channelCount];
  }

  return values;
};

const getSpatialMeanAndMaximumValues = (
  values: Float32Array,
  boardSize: number,
  channelCount: number,
) => {
  const positionCount = boardSize * boardSize;
  const pooledValues = new Float32Array(channelCount * 2);
  pooledValues.fill(Number.NEGATIVE_INFINITY, channelCount);

  for (let positionIndex = 0; positionIndex < positionCount; positionIndex += 1) {
    const positionOffset = positionIndex * channelCount;

    for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
      const value = values[positionOffset + channelIndex];
      pooledValues[channelIndex] += value / positionCount;
      pooledValues[channelCount + channelIndex] = Math.max(
        pooledValues[channelCount + channelIndex],
        value,
      );
    }
  }

  return pooledValues;
};

const convolve = (
  inputs: Float32Array,
  boardSize: number,
  inputChannelCount: number,
  outputChannelCount: number,
  kernelSize: number,
  padding: number,
  weights: Float32Array,
  biases: Float32Array,
) => {
  const outputs = new Float32Array(boardSize * boardSize * outputChannelCount);

  for (let outputRow = 0; outputRow < boardSize; outputRow += 1) {
    for (let outputColumn = 0; outputColumn < boardSize; outputColumn += 1) {
      for (let outputChannel = 0; outputChannel < outputChannelCount; outputChannel += 1) {
        let sum = biases[outputChannel];

        for (let kernelRow = 0; kernelRow < kernelSize; kernelRow += 1) {
          const inputRow = outputRow + kernelRow - padding;

          if (inputRow < 0 || inputRow >= boardSize) {
            continue;
          }

          for (let kernelColumn = 0; kernelColumn < kernelSize; kernelColumn += 1) {
            const inputColumn = outputColumn + kernelColumn - padding;

            if (inputColumn < 0 || inputColumn >= boardSize) {
              continue;
            }

            const inputOffset = (inputRow * boardSize + inputColumn) * inputChannelCount;
            const weightOffset =
              ((outputChannel * kernelSize + kernelRow) * kernelSize + kernelColumn) *
              inputChannelCount;

            for (let inputChannel = 0; inputChannel < inputChannelCount; inputChannel += 1) {
              sum += inputs[inputOffset + inputChannel] * weights[weightOffset + inputChannel];
            }
          }
        }

        const outputIndex =
          (outputRow * boardSize + outputColumn) * outputChannelCount + outputChannel;
        outputs[outputIndex] = sum;
      }
    }
  }

  return outputs;
};

const applyLinear = (
  inputs: Float32Array,
  outputCount: number,
  weights: Float32Array,
  biases: Float32Array,
) => {
  const outputs = new Float32Array(outputCount);

  for (let outputIndex = 0; outputIndex < outputCount; outputIndex += 1) {
    let sum = biases[outputIndex];
    const weightOffset = outputIndex * inputs.length;

    for (let inputIndex = 0; inputIndex < inputs.length; inputIndex += 1) {
      sum += inputs[inputIndex] * weights[weightOffset + inputIndex];
    }

    outputs[outputIndex] = sum;
  }

  return outputs;
};

class GoModelRuntime {
  private readonly manifest: GoModelManifest;
  private readonly tensors: Record<string, GoModelFloatTensor>;

  private constructor(manifest: GoModelManifest, tensors: Record<string, GoModelFloatTensor>) {
    this.manifest = manifest;
    this.tensors = tensors;
  }

  static load = async (manifestUrl: string, weightsUrl: string) => {
    const [manifestResponse, weightsResponse] = await Promise.all([
      fetch(manifestUrl),
      fetch(weightsUrl),
    ]);

    if (!manifestResponse.ok || !weightsResponse.ok) {
      throw new Error("Unable to load the Go model.");
    }

    const manifest: GoModelManifest = await manifestResponse.json();
    const weightsBuffer = await weightsResponse.arrayBuffer();

    if (!isSupportedManifest(manifest) || weightsBuffer.byteLength !== manifest.weightsBytes) {
      throw new Error("Unsupported or incomplete Go model.");
    }

    return GoModelRuntime.create(manifest, weightsBuffer);
  };

  static create = (manifest: GoModelManifest, weightsBuffer: ArrayBuffer) => {
    if (!isSupportedManifest(manifest) || weightsBuffer.byteLength !== manifest.weightsBytes) {
      throw new Error("Unsupported or incomplete Go model.");
    }

    const tensors: Record<string, GoModelFloatTensor> = {};

    for (const [name, tensorManifest] of Object.entries(manifest.tensors)) {
      const elementCount = getElementCount(tensorManifest.shape);

      if (tensorManifest.dtype === "float32") {
        tensors[name] = {
          data: new Float32Array(weightsBuffer, tensorManifest.dataOffset, elementCount).slice(),
          shape: tensorManifest.shape,
        };
        continue;
      }

      if (tensorManifest.scaleOffset === undefined) {
        throw new Error(`Missing quantization scale for ${name}.`);
      }

      const outputChannelCount = tensorManifest.shape[0];
      const valuesPerOutputChannel = elementCount / outputChannelCount;
      const quantizationGroupSize = tensorManifest.quantizationGroupSize ?? valuesPerOutputChannel;
      const groupCount = Math.ceil(valuesPerOutputChannel / quantizationGroupSize);
      const scales = new Float32Array(
        weightsBuffer,
        tensorManifest.scaleOffset,
        outputChannelCount * groupCount,
      );
      const values = new Float32Array(elementCount);
      const int8Values =
        tensorManifest.dtype === "int8"
          ? new Int8Array(weightsBuffer, tensorManifest.dataOffset, elementCount)
          : null;
      const int4Values =
        tensorManifest.dtype === "int4"
          ? new Uint8Array(
              weightsBuffer,
              tensorManifest.dataOffset,
              Math.ceil(elementCount / GO_MODEL_INT4_VALUES_PER_BYTE),
            )
          : null;

      for (let outputChannel = 0; outputChannel < outputChannelCount; outputChannel += 1) {
        const outputOffset = outputChannel * valuesPerOutputChannel;

        for (
          let channelValueIndex = 0;
          channelValueIndex < valuesPerOutputChannel;
          channelValueIndex += 1
        ) {
          const valueIndex = outputOffset + channelValueIndex;
          const packedValue = int4Values?.[Math.floor(valueIndex / GO_MODEL_INT4_VALUES_PER_BYTE)];
          const encodedInt4Value =
            packedValue === undefined
              ? 0
              : valueIndex % GO_MODEL_INT4_VALUES_PER_BYTE === 0
                ? packedValue & GO_MODEL_INT4_NIBBLE_MASK
                : packedValue >> GO_MODEL_INT4_NIBBLE_BITS;
          const quantizedValue =
            int8Values?.[valueIndex] ?? encodedInt4Value - GO_MODEL_INT4_ZERO_POINT;
          const scale =
            scales[
              outputChannel * groupCount + Math.floor(channelValueIndex / quantizationGroupSize)
            ];
          values[valueIndex] = quantizedValue * scale;
        }
      }

      tensors[name] = { data: values, shape: tensorManifest.shape };
    }

    return new GoModelRuntime(manifest, tensors);
  };

  infer = (features: Float32Array): GoModelInference => {
    const architecture = this.manifest.architecture;
    const tensor = (name: string) => {
      const selectedTensor = this.tensors[name];

      if (!selectedTensor) {
        throw new Error(`Model tensor ${name} is missing.`);
      }

      return selectedTensor.data;
    };
    let trunkValues = applyRelu(
      convolve(
        features,
        architecture.boardSize,
        architecture.inputPlaneCount,
        architecture.trunkChannelCount,
        GO_MODEL_KERNEL_SIZE,
        GO_MODEL_KERNEL_RADIUS,
        tensor("stem.weight"),
        tensor("stem.bias"),
      ),
    );

    for (
      let residualBlockIndex = 0;
      residualBlockIndex < architecture.residualBlockCount;
      residualBlockIndex += 1
    ) {
      const prefix = `residual.${residualBlockIndex}`;
      const isNestedBlock = architecture.residualBlockKind === GO_MODEL_NESTED_RESIDUAL_BLOCK_KIND;
      const bottleneckChannelCount = architecture.bottleneckChannelCount ?? 0;
      const globalResidualBlockInterval = architecture.globalResidualBlockInterval ?? 0;
      const isGlobalResidualBlock =
        isNestedBlock &&
        globalResidualBlockInterval > 0 &&
        (residualBlockIndex + 1) % globalResidualBlockInterval === 0;
      const firstBlockInputs = isNestedBlock
        ? applyRelu(
            convolve(
              trunkValues,
              architecture.boardSize,
              architecture.trunkChannelCount,
              bottleneckChannelCount,
              GO_MODEL_POLICY_KERNEL_SIZE,
              0,
              tensor(`${prefix}.reduce.weight`),
              tensor(`${prefix}.reduce.bias`),
            ),
          )
        : trunkValues;
      let hiddenValues = applyRelu(
        convolve(
          firstBlockInputs,
          architecture.boardSize,
          isNestedBlock ? bottleneckChannelCount : architecture.trunkChannelCount,
          isNestedBlock ? bottleneckChannelCount : architecture.trunkChannelCount,
          GO_MODEL_KERNEL_SIZE,
          GO_MODEL_KERNEL_RADIUS,
          tensor(`${prefix}.first.weight`),
          tensor(`${prefix}.first.bias`),
        ),
      );
      if (isGlobalResidualBlock) {
        const globalValues = getSpatialMeanAndMaximumValues(
          hiddenValues,
          architecture.boardSize,
          bottleneckChannelCount,
        );
        const globalHidden = applyRelu(
          applyLinear(
            globalValues,
            architecture.globalResidualHiddenChannelCount ?? 0,
            tensor(`${prefix}.global.hidden.weight`),
            tensor(`${prefix}.global.hidden.bias`),
          ),
        );
        const globalBias = applyLinear(
          globalHidden,
          bottleneckChannelCount,
          tensor(`${prefix}.global.output.weight`),
          tensor(`${prefix}.global.output.bias`),
        );
        hiddenValues = addChannelBias(hiddenValues, bottleneckChannelCount, globalBias);
      }
      const secondBlockValues = convolve(
        hiddenValues,
        architecture.boardSize,
        isNestedBlock ? bottleneckChannelCount : architecture.trunkChannelCount,
        isNestedBlock ? bottleneckChannelCount : architecture.trunkChannelCount,
        GO_MODEL_KERNEL_SIZE,
        GO_MODEL_KERNEL_RADIUS,
        tensor(`${prefix}.second.weight`),
        tensor(`${prefix}.second.bias`),
      );
      const residualValues = isNestedBlock
        ? convolve(
            applyRelu(secondBlockValues),
            architecture.boardSize,
            bottleneckChannelCount,
            architecture.trunkChannelCount,
            GO_MODEL_POLICY_KERNEL_SIZE,
            0,
            tensor(`${prefix}.expand.weight`),
            tensor(`${prefix}.expand.bias`),
          )
        : secondBlockValues;
      trunkValues = addAndApplyRelu(trunkValues, residualValues);
    }

    const policyValues = applyRelu(
      convolve(
        trunkValues,
        architecture.boardSize,
        architecture.trunkChannelCount,
        architecture.policyChannelCount,
        GO_MODEL_POLICY_KERNEL_SIZE,
        0,
        tensor("policy.convolution.weight"),
        tensor("policy.convolution.bias"),
      ),
    );
    const policyLogits = applyLinear(
      policyValues,
      architecture.policyMoveCount,
      tensor("policy.linear.weight"),
      tensor("policy.linear.bias"),
    );
    const valueValues = applyRelu(
      convolve(
        trunkValues,
        architecture.boardSize,
        architecture.trunkChannelCount,
        architecture.valueChannelCount,
        GO_MODEL_VALUE_KERNEL_SIZE,
        0,
        tensor("value.convolution.weight"),
        tensor("value.convolution.bias"),
      ),
    );
    const valueHidden = applyRelu(
      applyLinear(
        valueValues,
        architecture.scoreHiddenChannelCount,
        tensor("value.hidden.weight"),
        tensor("value.hidden.bias"),
      ),
    );
    const valueOutput = applyLinear(
      valueHidden,
      1,
      tensor("value.output.weight"),
      tensor("value.output.bias"),
    );
    return {
      policyLogits,
      value: Math.tanh(valueOutput[0]),
    };
  };
}

export { GoModelRuntime };
