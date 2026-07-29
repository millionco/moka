export {};

declare global {
  interface GoGameState {
    board: Int8Array;
    consecutivePassCount: number;
    koMove: number;
    moveCount: number;
    moveHistory: number[];
    nextColor: number;
  }

  interface GoModelArchitecture {
    boardSize: number;
    bottleneckChannelCount?: number;
    globalResidualBlockInterval?: number;
    globalResidualHiddenChannelCount?: number;
    inputPlaneCount: number;
    policyChannelCount: number;
    policyMoveCount: number;
    residualBlockCount: number;
    residualBlockKind?: "nested-bottleneck" | "standard";
    scoreHiddenChannelCount: number;
    trunkChannelCount: number;
    valueChannelCount: number;
  }

  interface GoModelFloatTensor {
    data: Float32Array;
    shape: number[];
  }

  interface GoModelInference {
    policyLogits: Float32Array;
    value: number;
  }

  interface GoModelLoadOptions {
    isGzipCompressed?: boolean;
    manifestUrl: string;
    onProgress?: (loadedByteCount: number, totalByteCount: number) => void;
    weightsUrl: string;
  }

  interface GoModelManifest {
    architecture: GoModelArchitecture;
    format: string;
    sha256: string;
    tensors: Record<string, GoModelTensorManifest>;
    version: number;
    weightsBytes: number;
  }

  interface GoModelTensorManifest {
    dataOffset: number;
    dtype: "float32" | "int4" | "int8";
    quantizationGroupSize?: number;
    scaleOffset?: number;
    shape: number[];
  }

  interface GoModelWorkerErrorMessage {
    error: string;
    requestId: number;
    type: "error";
  }

  interface GoModelWorkerInferenceMessage {
    features: Float32Array;
    requestId: number;
    type: "infer";
  }

  interface GoModelWorkerInitializeMessage {
    manifest: GoModelManifest;
    requestId: number;
    type: "initialize";
    weightsBuffer: ArrayBuffer;
  }

  interface GoModelWorkerReadyMessage {
    requestId: number;
    type: "ready";
  }

  interface GoModelWorkerResultMessage {
    policyLogits: Float32Array;
    requestId: number;
    type: "result";
    value: number;
  }

  interface PendingGoModelRequest {
    reject: (error: Error) => void;
    resolve: (result: GoModelInference | undefined) => void;
  }
}
