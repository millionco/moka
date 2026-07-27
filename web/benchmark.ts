import { GoModelWorkerClient } from "./client";
import {
  GO_MODEL_BENCHMARK_INPUT_VALUE_COUNT,
  GO_MODEL_BENCHMARK_ITERATION_COUNT,
  GO_MODEL_BENCHMARK_MANIFEST_URL,
  GO_MODEL_BENCHMARK_P50,
  GO_MODEL_BENCHMARK_P95,
  GO_MODEL_BENCHMARK_WARMUP_COUNT,
  GO_MODEL_BENCHMARK_WEIGHTS_URL,
} from "./constants";

const getPercentile = (values: number[], percentile: number) => {
  const sortedValues = [...values].sort((leftValue, rightValue) => leftValue - rightValue);
  const index = Math.min(
    sortedValues.length - 1,
    Math.round((sortedValues.length - 1) * percentile),
  );
  return sortedValues[index];
};

const runBenchmark = async () => {
  const resultsElement = document.querySelector("#results");

  if (!resultsElement) {
    return;
  }

  const client = new GoModelWorkerClient(
    new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
  );
  const initializationStartTime = performance.now();
  await client.initialize({
    manifestUrl: GO_MODEL_BENCHMARK_MANIFEST_URL,
    weightsUrl: GO_MODEL_BENCHMARK_WEIGHTS_URL,
  });
  const initializationDurationMs = performance.now() - initializationStartTime;

  for (let warmupIndex = 0; warmupIndex < GO_MODEL_BENCHMARK_WARMUP_COUNT; warmupIndex += 1) {
    await client.infer(new Float32Array(GO_MODEL_BENCHMARK_INPUT_VALUE_COUNT));
  }

  const inferenceDurationsMs: number[] = [];

  for (
    let iterationIndex = 0;
    iterationIndex < GO_MODEL_BENCHMARK_ITERATION_COUNT;
    iterationIndex += 1
  ) {
    const inferenceStartTime = performance.now();
    await client.infer(new Float32Array(GO_MODEL_BENCHMARK_INPUT_VALUE_COUNT));
    inferenceDurationsMs.push(performance.now() - inferenceStartTime);
  }

  const meanDurationMs =
    inferenceDurationsMs.reduce((durationSum, duration) => durationSum + duration, 0) /
    inferenceDurationsMs.length;
  resultsElement.textContent = JSON.stringify(
    {
      initializationMs: initializationDurationMs,
      iterations: GO_MODEL_BENCHMARK_ITERATION_COUNT,
      meanMs: meanDurationMs,
      p50Ms: getPercentile(inferenceDurationsMs, GO_MODEL_BENCHMARK_P50),
      p95Ms: getPercentile(inferenceDurationsMs, GO_MODEL_BENCHMARK_P95),
    },
    null,
    2,
  );
  client.dispose();
};

void runBenchmark();
