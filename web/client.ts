import { getBufferSha256 } from "./utils/get-buffer-sha256";
import { decompressGzipResponse } from "./utils/decompress-gzip-response";
import { readResponseBuffer } from "./utils/read-response-buffer";

class GoModelWorkerClient {
  private nextRequestId = 0;
  private readonly pendingRequests = new Map<number, PendingGoModelRequest>();
  private readonly worker: Worker;

  constructor(worker: Worker) {
    this.worker = worker;
    this.worker.onmessage = (
      event: MessageEvent<
        GoModelWorkerErrorMessage | GoModelWorkerReadyMessage | GoModelWorkerResultMessage
      >,
    ) => {
      const message = event.data;
      const pendingRequest = this.pendingRequests.get(message.requestId);

      if (!pendingRequest) {
        return;
      }

      this.pendingRequests.delete(message.requestId);

      if (message.type === "error") {
        pendingRequest.reject(new Error(message.error));
      } else if (message.type === "result") {
        pendingRequest.resolve({
          policyLogits: message.policyLogits,
          value: message.value,
        });
      } else {
        pendingRequest.resolve(undefined);
      }
    };
  }

  private request = (
    message: GoModelWorkerInferenceMessage | GoModelWorkerInitializeMessage,
    transfer: Transferable[] = [],
  ) =>
    new Promise<GoModelInference | undefined>((resolve, reject) => {
      this.pendingRequests.set(message.requestId, { reject, resolve });
      this.worker.postMessage(message, transfer);
    });

  initialize = async ({
    isGzipCompressed = false,
    manifestUrl,
    onProgress,
    weightsUrl,
  }: GoModelLoadOptions) => {
    const [manifestResponse, weightsResponse] = await Promise.all([
      fetch(manifestUrl),
      fetch(weightsUrl),
    ]);

    if (!manifestResponse.ok || !weightsResponse.ok) {
      throw new Error("Unable to load the Go model.");
    }

    const manifest: GoModelManifest = await manifestResponse.json();
    const readableWeightsResponse = isGzipCompressed
      ? decompressGzipResponse(weightsResponse)
      : weightsResponse;
    const weightsBuffer = await readResponseBuffer(
      readableWeightsResponse,
      manifest.weightsBytes,
      onProgress,
    );
    const weightsSha256 = await getBufferSha256(weightsBuffer);

    if (weightsSha256 !== manifest.sha256) {
      throw new Error("Go model integrity check failed.");
    }

    const requestId = this.nextRequestId;
    this.nextRequestId += 1;
    await this.request(
      {
        manifest,
        requestId,
        type: "initialize",
        weightsBuffer,
      },
      [weightsBuffer],
    );
  };

  infer = async (features: Float32Array) => {
    const requestId = this.nextRequestId;
    this.nextRequestId += 1;
    const result = await this.request(
      {
        features,
        requestId,
        type: "infer",
      },
      [features.buffer],
    );

    if (!result) {
      throw new Error("Go model worker returned no inference result.");
    }

    return result;
  };

  dispose = () => {
    this.worker.terminate();

    for (const pendingRequest of this.pendingRequests.values()) {
      pendingRequest.reject(new Error("Go model worker was disposed."));
    }

    this.pendingRequests.clear();
  };
}

export { GoModelWorkerClient };
