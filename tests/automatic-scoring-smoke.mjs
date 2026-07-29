import assert from "node:assert/strict";
import { getAreaScore, getAutomaticallyDeadMoves, removeDeadStones } from "../dist/index.js";

const BLACK_COLOR = 1;
const BOARD_AREA = 81;
const BOARD_SIZE = 9;
const EXPECTED_AUTOMATIC_DEAD_STONE_COUNT = 17;
const EXPECTED_AUTOMATIC_SCORE = 26;
const NO_KO_MOVE = -1;
const PASS_MOVE = 81;
const WHITE_COLOR = -1;
const REPORTED_ENDGAME_BOARD_ROWS = [
  "......WWW",
  ".....WWBB",
  "WWWWWWB.B",
  "WBBBWB.BB",
  "BBBWBBBBB",
  ".BWWWBBWB",
  ".BBBWWWWW",
  "BW.BBBWBW",
  ".B.BWWW.W",
];
const board = new Int8Array(BOARD_AREA);

for (let row = 0; row < BOARD_SIZE; row += 1) {
  for (let column = 0; column < BOARD_SIZE; column += 1) {
    const boardPoint = REPORTED_ENDGAME_BOARD_ROWS[row][column];
    board[row * BOARD_SIZE + column] =
      boardPoint === "B" ? BLACK_COLOR : boardPoint === "W" ? WHITE_COLOR : 0;
  }
}

const gameState = {
  board,
  consecutivePassCount: 2,
  koMove: NO_KO_MOVE,
  moveCount: 72,
  moveHistory: [PASS_MOVE, PASS_MOVE],
  nextColor: BLACK_COLOR,
};
const automaticallyDeadMoves = getAutomaticallyDeadMoves(gameState);
const automaticallyScoredGameState = removeDeadStones(gameState, automaticallyDeadMoves);

assert.equal(automaticallyDeadMoves.length, EXPECTED_AUTOMATIC_DEAD_STONE_COUNT);
assert.ok(
  automaticallyDeadMoves.every(
    (automaticallyDeadMove) => gameState.board[automaticallyDeadMove] === WHITE_COLOR,
  ),
);
assert.equal(getAreaScore(automaticallyScoredGameState), EXPECTED_AUTOMATIC_SCORE);
