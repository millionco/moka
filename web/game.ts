import {
  GO_ARENA_BLACK_COLOR,
  GO_ARENA_BOARD_AREA,
  GO_ARENA_BOARD_SIZE,
  GO_ARENA_BUTTON_FEATURE_VALUE,
  GO_ARENA_KOMI_NORMALIZATION_POINTS,
  GO_ARENA_KOMI_POINTS,
  GO_ARENA_MAXIMUM_MOVE_COUNT,
  GO_ARENA_MINIMUM_PASS_MOVE_COUNT,
  GO_ARENA_PASS_MOVE,
  GO_ARENA_RECENT_MOVE_COUNT,
  GO_ARENA_STUDENT_INPUT_PLANE_COUNT,
  GO_ARENA_TEACHER_GLOBAL_FEATURE_COUNT,
  GO_ARENA_TEACHER_SPATIAL_FEATURE_COUNT,
  GO_ARENA_WHITE_COLOR,
} from "./constants";

const createGameState = (): GoArenaGameState => ({
  board: new Int8Array(GO_ARENA_BOARD_AREA),
  consecutivePassCount: 0,
  koMove: -1,
  moveCount: 0,
  moveHistory: [],
  nextColor: GO_ARENA_BLACK_COLOR,
});

const copyGameState = (gameState: GoArenaGameState): GoArenaGameState => ({
  board: gameState.board.slice(),
  consecutivePassCount: gameState.consecutivePassCount,
  koMove: gameState.koMove,
  moveCount: gameState.moveCount,
  moveHistory: [...gameState.moveHistory],
  nextColor: gameState.nextColor,
});

const getAdjacentMoves = (move: number) => {
  const row = Math.floor(move / GO_ARENA_BOARD_SIZE);
  const column = move % GO_ARENA_BOARD_SIZE;
  const adjacentMoves: number[] = [];

  if (row > 0) {
    adjacentMoves.push(move - GO_ARENA_BOARD_SIZE);
  }
  if (row < GO_ARENA_BOARD_SIZE - 1) {
    adjacentMoves.push(move + GO_ARENA_BOARD_SIZE);
  }
  if (column > 0) {
    adjacentMoves.push(move - 1);
  }
  if (column < GO_ARENA_BOARD_SIZE - 1) {
    adjacentMoves.push(move + 1);
  }

  return adjacentMoves;
};

const getGroup = (board: Int8Array, startingMove: number) => {
  const color = board[startingMove];

  if (color === 0) {
    return { liberties: new Set<number>(), stones: [] };
  }

  const pendingMoves = [startingMove];
  const visitedMoves = new Set<number>();
  const liberties = new Set<number>();
  const stones: number[] = [];

  while (pendingMoves.length > 0) {
    const move = pendingMoves.pop();

    if (move === undefined || visitedMoves.has(move)) {
      continue;
    }

    visitedMoves.add(move);
    stones.push(move);

    for (const adjacentMove of getAdjacentMoves(move)) {
      const adjacentColor = board[adjacentMove];

      if (adjacentColor === 0) {
        liberties.add(adjacentMove);
      } else if (adjacentColor === color) {
        pendingMoves.push(adjacentMove);
      }
    }
  }

  return { liberties, stones };
};

const playMove = (gameState: GoArenaGameState, move: number) => {
  if (move === GO_ARENA_PASS_MOVE) {
    const nextState = copyGameState(gameState);
    nextState.consecutivePassCount += 1;
    nextState.koMove = -1;
    nextState.moveCount += 1;
    nextState.moveHistory.push(move);
    nextState.nextColor *= -1;
    return nextState;
  }

  if (gameState.board[move] !== 0 || gameState.koMove === move) {
    return null;
  }

  const nextState = copyGameState(gameState);
  nextState.board[move] = gameState.nextColor;
  const capturedMoves: number[] = [];

  for (const adjacentMove of getAdjacentMoves(move)) {
    if (nextState.board[adjacentMove] !== -gameState.nextColor) {
      continue;
    }

    const opponentGroup = getGroup(nextState.board, adjacentMove);

    if (opponentGroup.liberties.size > 0) {
      continue;
    }

    for (const capturedMove of opponentGroup.stones) {
      nextState.board[capturedMove] = 0;
      capturedMoves.push(capturedMove);
    }
  }

  const playedGroup = getGroup(nextState.board, move);

  if (playedGroup.liberties.size === 0) {
    return null;
  }

  nextState.consecutivePassCount = 0;
  nextState.koMove =
    capturedMoves.length === 1 &&
    playedGroup.stones.length === 1 &&
    playedGroup.liberties.size === 1
      ? capturedMoves[0]
      : -1;
  nextState.moveCount += 1;
  nextState.moveHistory.push(move);
  nextState.nextColor *= -1;
  return nextState;
};

const getLegalMoves = (gameState: GoArenaGameState) => {
  const legalMoves: number[] = [];

  for (let move = 0; move < GO_ARENA_BOARD_AREA; move += 1) {
    if (playMove(gameState, move)) {
      legalMoves.push(move);
    }
  }

  legalMoves.push(GO_ARENA_PASS_MOVE);
  return legalMoves;
};

const selectHighestLegalMove = (gameState: GoArenaGameState, logits: Float32Array) => {
  const legalMoves = getLegalMoves(gameState);
  let selectedMove =
    gameState.moveCount >= GO_ARENA_MINIMUM_PASS_MOVE_COUNT ? GO_ARENA_PASS_MOVE : legalMoves[0];
  let selectedValue = selectedMove === undefined ? Number.NEGATIVE_INFINITY : logits[selectedMove];

  for (const legalMove of legalMoves) {
    if (
      legalMove === GO_ARENA_PASS_MOVE &&
      gameState.moveCount < GO_ARENA_MINIMUM_PASS_MOVE_COUNT
    ) {
      continue;
    }

    if (logits[legalMove] > selectedValue) {
      selectedMove = legalMove;
      selectedValue = logits[legalMove];
    }
  }

  return selectedMove ?? GO_ARENA_PASS_MOVE;
};

const encodeStudentFeatures = (gameState: GoArenaGameState) => {
  const features = new Float32Array(GO_ARENA_BOARD_AREA * GO_ARENA_STUDENT_INPUT_PLANE_COUNT);
  const visitedMoves = new Set<number>();
  const setFeature = (move: number, featureIndex: number, value = 1) => {
    features[move * GO_ARENA_STUDENT_INPUT_PLANE_COUNT + featureIndex] = value;
  };

  for (let move = 0; move < GO_ARENA_BOARD_AREA; move += 1) {
    const color = gameState.board[move];

    if (color === 0) {
      continue;
    }

    const isCurrentPlayer = color === gameState.nextColor;
    setFeature(move, isCurrentPlayer ? 0 : 1);

    if (visitedMoves.has(move)) {
      continue;
    }

    const group = getGroup(gameState.board, move);

    for (const stoneMove of group.stones) {
      visitedMoves.add(stoneMove);
    }

    const featureIndex =
      group.liberties.size === 1
        ? isCurrentPlayer
          ? 2
          : 3
        : group.liberties.size === 2
          ? isCurrentPlayer
            ? 4
            : 5
          : -1;

    if (featureIndex >= 0) {
      for (const stoneMove of group.stones) {
        setFeature(stoneMove, featureIndex);
      }
    }
  }

  if (gameState.koMove >= 0) {
    setFeature(gameState.koMove, 6);
  }

  const recentFeatureIndexes = [7, 8];

  for (let historyOffset = 1; historyOffset <= recentFeatureIndexes.length; historyOffset += 1) {
    const historyMove = gameState.moveHistory.at(-historyOffset);

    if (historyMove !== undefined && historyMove < GO_ARENA_BOARD_AREA) {
      setFeature(historyMove, recentFeatureIndexes[historyOffset - 1]);
    } else if (historyMove === GO_ARENA_PASS_MOVE) {
      const passFeatureIndex = 8 + historyOffset;

      for (let move = 0; move < GO_ARENA_BOARD_AREA; move += 1) {
        setFeature(move, passFeatureIndex);
      }
    }
  }

  const perspectiveKomi =
    (-GO_ARENA_KOMI_POINTS * gameState.nextColor) / GO_ARENA_KOMI_NORMALIZATION_POINTS;

  for (let move = 0; move < GO_ARENA_BOARD_AREA; move += 1) {
    setFeature(move, 11, perspectiveKomi);
  }

  return features;
};

const encodeTeacherFeatures = (gameState: GoArenaGameState) => {
  const spatialFeatures = new Float32Array(
    GO_ARENA_TEACHER_SPATIAL_FEATURE_COUNT * GO_ARENA_BOARD_AREA,
  );
  const globalFeatures = new Float32Array(GO_ARENA_TEACHER_GLOBAL_FEATURE_COUNT);
  const setSpatialFeature = (featureIndex: number, move: number, value = 1) => {
    spatialFeatures[featureIndex * GO_ARENA_BOARD_AREA + move] = value;
  };
  const visitedMoves = new Set<number>();

  for (let move = 0; move < GO_ARENA_BOARD_AREA; move += 1) {
    setSpatialFeature(0, move);
    const color = gameState.board[move];

    if (color === 0) {
      continue;
    }

    setSpatialFeature(color === gameState.nextColor ? 1 : 2, move);

    if (visitedMoves.has(move)) {
      continue;
    }

    const group = getGroup(gameState.board, move);

    for (const stoneMove of group.stones) {
      visitedMoves.add(stoneMove);
    }

    if (group.liberties.size >= 1 && group.liberties.size <= 3) {
      for (const stoneMove of group.stones) {
        setSpatialFeature(2 + group.liberties.size, stoneMove);
      }
    }
  }

  if (gameState.koMove >= 0) {
    setSpatialFeature(6, gameState.koMove);
  }

  const recentMoves = gameState.moveHistory.slice(-GO_ARENA_RECENT_MOVE_COUNT).reverse();

  for (let recentMoveIndex = 0; recentMoveIndex < recentMoves.length; recentMoveIndex += 1) {
    const recentMove = recentMoves[recentMoveIndex];

    if (recentMove === GO_ARENA_PASS_MOVE) {
      globalFeatures[recentMoveIndex] = 1;
    } else {
      setSpatialFeature(9 + recentMoveIndex, recentMove);
    }
  }

  globalFeatures[5] =
    (-GO_ARENA_KOMI_POINTS * gameState.nextColor) / GO_ARENA_KOMI_NORMALIZATION_POINTS;
  globalFeatures[6] = 1;
  globalFeatures[7] = GO_ARENA_BUTTON_FEATURE_VALUE;
  return { globalFeatures, spatialFeatures };
};

const getAreaScore = (gameState: GoArenaGameState) => {
  let score = -GO_ARENA_KOMI_POINTS;
  const visitedMoves = new Set<number>();

  for (const color of gameState.board) {
    score += color;
  }

  for (let move = 0; move < GO_ARENA_BOARD_AREA; move += 1) {
    if (gameState.board[move] !== 0 || visitedMoves.has(move)) {
      continue;
    }

    const pendingMoves = [move];
    const territoryMoves: number[] = [];
    const borderingColors = new Set<number>();

    while (pendingMoves.length > 0) {
      const territoryMove = pendingMoves.pop();

      if (territoryMove === undefined || visitedMoves.has(territoryMove)) {
        continue;
      }

      visitedMoves.add(territoryMove);
      territoryMoves.push(territoryMove);

      for (const adjacentMove of getAdjacentMoves(territoryMove)) {
        const adjacentColor = gameState.board[adjacentMove];

        if (adjacentColor === 0 && !visitedMoves.has(adjacentMove)) {
          pendingMoves.push(adjacentMove);
        } else if (adjacentColor !== 0) {
          borderingColors.add(adjacentColor);
        }
      }
    }

    if (borderingColors.size === 1 && borderingColors.has(GO_ARENA_BLACK_COLOR)) {
      score += territoryMoves.length;
    } else if (borderingColors.size === 1 && borderingColors.has(GO_ARENA_WHITE_COLOR)) {
      score -= territoryMoves.length;
    }
  }

  return score;
};

const isGameOver = (gameState: GoArenaGameState) =>
  gameState.consecutivePassCount >= 2 || gameState.moveCount >= GO_ARENA_MAXIMUM_MOVE_COUNT;

export {
  createGameState,
  encodeStudentFeatures,
  encodeTeacherFeatures,
  getAreaScore,
  getLegalMoves,
  isGameOver,
  playMove,
  selectHighestLegalMove,
};
