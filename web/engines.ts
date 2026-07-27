import * as onnxRuntime from "onnxruntime-web/wasm";
import {
  GO_ARENA_BOARD_AREA,
  GO_ARENA_BOARD_SIZE,
  GO_ARENA_KATAGO_MODEL_URL,
  GO_ARENA_KATAGO_NAME,
  GO_ARENA_KATAGO_VALUE_OUTPUT_COUNT,
  GO_ARENA_KATAGO_WASM_URL,
  GO_ARENA_PASS_MOVE,
  GO_ARENA_STUDENT_MANIFEST_URL,
  GO_ARENA_STUDENT_NAME,
  GO_ARENA_STUDENT_WEIGHTS_URL,
  GO_ARENA_TEACHER_GLOBAL_FEATURE_COUNT,
  GO_ARENA_TEACHER_SPATIAL_FEATURE_COUNT,
} from "./constants";
import { encodeStudentFeatures, encodeTeacherFeatures, selectHighestLegalMove } from "./game";
import { GoModelWorkerClient } from "./client";

class StudentArenaEngine implements GoArenaEngine {
  name = GO_ARENA_STUDENT_NAME;
  private readonly client = new GoModelWorkerClient(
    new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
  );

  initialize = async () => {
    await this.client.initialize({
      manifestUrl: GO_ARENA_STUDENT_MANIFEST_URL,
      weightsUrl: GO_ARENA_STUDENT_WEIGHTS_URL,
    });
  };

  evaluateStates = async (gameStates: GoArenaGameState[]) => {
    const results = await Promise.all(
      gameStates.map((gameState) => this.client.infer(encodeStudentFeatures(gameState))),
    );
    return results.map((result, resultIndex) => ({
      move: selectHighestLegalMove(gameStates[resultIndex], result.policyLogits),
      winProbability: (result.value + 1) / 2,
    }));
  };

  selectMoves = async (gameStates: GoArenaGameState[]) =>
    (await this.evaluateStates(gameStates)).map((evaluation) => evaluation.move);

  selectMove = async (gameState: GoArenaGameState) =>
    (await this.selectMoves([gameState]))[0] ?? GO_ARENA_PASS_MOVE;

  dispose = () => {
    this.client.dispose();
  };
}

class KataGoArenaEngine implements GoArenaEngine {
  name = GO_ARENA_KATAGO_NAME;
  private session: onnxRuntime.InferenceSession | null = null;

  initialize = async () => {
    if (GO_ARENA_KATAGO_WASM_URL) {
      onnxRuntime.env.wasm.wasmPaths = GO_ARENA_KATAGO_WASM_URL;
    }

    onnxRuntime.env.wasm.numThreads = 1;
    this.session = await onnxRuntime.InferenceSession.create(GO_ARENA_KATAGO_MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  };

  evaluateStates = async (gameStates: GoArenaGameState[]) => {
    if (!this.session) {
      throw new Error("KataGo is not initialized.");
    }

    if (gameStates.length === 0) {
      return [];
    }

    const batchSize = gameStates.length;
    const globalFeatures = new Float32Array(batchSize * GO_ARENA_TEACHER_GLOBAL_FEATURE_COUNT);
    const spatialFeatures = new Float32Array(
      batchSize * GO_ARENA_TEACHER_SPATIAL_FEATURE_COUNT * GO_ARENA_BOARD_AREA,
    );

    for (let gameIndex = 0; gameIndex < batchSize; gameIndex += 1) {
      const encodedFeatures = encodeTeacherFeatures(gameStates[gameIndex]);
      globalFeatures.set(
        encodedFeatures.globalFeatures,
        gameIndex * GO_ARENA_TEACHER_GLOBAL_FEATURE_COUNT,
      );
      spatialFeatures.set(
        encodedFeatures.spatialFeatures,
        gameIndex * GO_ARENA_TEACHER_SPATIAL_FEATURE_COUNT * GO_ARENA_BOARD_AREA,
      );
    }

    const outputs = await this.session.run({
      input_global: new onnxRuntime.Tensor("float32", globalFeatures, [
        batchSize,
        GO_ARENA_TEACHER_GLOBAL_FEATURE_COUNT,
      ]),
      input_spatial: new onnxRuntime.Tensor("float32", spatialFeatures, [
        batchSize,
        GO_ARENA_TEACHER_SPATIAL_FEATURE_COUNT,
        GO_ARENA_BOARD_SIZE,
        GO_ARENA_BOARD_SIZE,
      ]),
    });
    const boardPolicy = outputs.policy?.data;
    const passPolicy = outputs.policy_pass?.data;
    const valueData = outputs.value?.data;

    if (
      !(boardPolicy instanceof Float32Array) ||
      !(passPolicy instanceof Float32Array) ||
      !(valueData instanceof Float32Array)
    ) {
      throw new Error("KataGo returned unexpected outputs.");
    }

    return gameStates.map((gameState, gameIndex) => {
      const logits = new Float32Array(GO_ARENA_BOARD_AREA + 1);
      const policyOffset = gameIndex * GO_ARENA_BOARD_AREA;
      logits.set(boardPolicy.subarray(policyOffset, policyOffset + GO_ARENA_BOARD_AREA));
      logits[GO_ARENA_PASS_MOVE] = passPolicy[gameIndex];
      const valueOffset = gameIndex * GO_ARENA_KATAGO_VALUE_OUTPUT_COUNT;
      const winLogit = valueData[valueOffset];
      const lossLogit = valueData[valueOffset + 1];
      const maximumValueLogit = Math.max(winLogit, lossLogit);
      const winWeight = Math.exp(winLogit - maximumValueLogit);
      const lossWeight = Math.exp(lossLogit - maximumValueLogit);
      return {
        move: selectHighestLegalMove(gameState, logits),
        winProbability: winWeight / (winWeight + lossWeight),
      };
    });
  };

  selectMoves = async (gameStates: GoArenaGameState[]) =>
    (await this.evaluateStates(gameStates)).map((evaluation) => evaluation.move);

  selectMove = async (gameState: GoArenaGameState) => {
    const moves = await this.selectMoves([gameState]);
    return moves[0] ?? GO_ARENA_PASS_MOVE;
  };
}

export { KataGoArenaEngine, StudentArenaEngine };
