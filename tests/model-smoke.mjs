import assert from "node:assert/strict";
import fs from "node:fs";

import { GoModelRuntime, createGameState, encodeStudentFeatures, playMove } from "../dist/index.js";

const manifest = JSON.parse(fs.readFileSync("model/go-model.json", "utf8"));
const weights = fs.readFileSync("model/go-model.bin");
const weightsBuffer = weights.buffer.slice(
  weights.byteOffset,
  weights.byteOffset + weights.byteLength,
);
const runtime = GoModelRuntime.create(manifest, weightsBuffer);
const openingMoves = [20, 24, 56, 60];
let gameState = createGameState();

for (const move of openingMoves) {
  const nextGameState = playMove(gameState, move);
  assert.ok(nextGameState);
  gameState = nextGameState;
}

const result = runtime.infer(encodeStudentFeatures(gameState));

assert.equal(manifest.architecture.globalResidualBlockInterval, 4);
assert.equal(manifest.architecture.globalResidualHiddenChannelCount, 8);
assert.equal(result.policyLogits.length, 82);
assert.ok(result.policyLogits.every(Number.isFinite));
assert.ok(Number.isFinite(result.value));
