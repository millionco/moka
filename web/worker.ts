import { GoModelRuntime } from "./runtime";

let runtime: GoModelRuntime | null = null;

self.onmessage = async (
  event: MessageEvent<GoModelWorkerInferenceMessage | GoModelWorkerInitializeMessage>,
) => {
  const message = event.data;

  try {
    if (message.type === "initialize") {
      runtime = GoModelRuntime.create(message.manifest, message.weightsBuffer);
      const response: GoModelWorkerReadyMessage = {
        requestId: message.requestId,
        type: "ready",
      };
      self.postMessage(response);
      return;
    }

    if (!runtime) {
      throw new Error("Go model worker is not initialized.");
    }

    const result = runtime.infer(message.features);
    const response: GoModelWorkerResultMessage = {
      ...result,
      requestId: message.requestId,
      type: "result",
    };
    self.postMessage(response, { transfer: [response.policyLogits.buffer] });
  } catch (error) {
    const response: GoModelWorkerErrorMessage = {
      error: error instanceof Error ? error.message : "Go model inference failed.",
      requestId: message.requestId,
      type: "error",
    };
    self.postMessage(response);
  }
};
