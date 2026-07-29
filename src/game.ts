import {
  GO_BLACK_COLOR,
  GO_BOARD_AREA,
  GO_BOARD_SIZE,
  GO_CURRENT_PLAYER_FEATURE_INDEX,
  GO_CURRENT_PLAYER_ONE_LIBERTY_FEATURE_INDEX,
  GO_CURRENT_PLAYER_TWO_LIBERTY_FEATURE_INDEX,
  GO_EMPTY_COLOR,
  GO_GAME_OVER_PASS_COUNT,
  GO_KO_FEATURE_INDEX,
  GO_KOMI_FEATURE_INDEX,
  GO_KOMI_NORMALIZATION_POINTS,
  GO_KOMI_POINTS,
  GO_MAXIMUM_MOVE_COUNT,
  GO_MINIMUM_PASS_MOVE_COUNT,
  GO_NO_KO_MOVE,
  GO_OPPONENT_FEATURE_INDEX,
  GO_OPPONENT_ONE_LIBERTY_FEATURE_INDEX,
  GO_OPPONENT_TWO_LIBERTY_FEATURE_INDEX,
  GO_PASS_ALIVE_MAXIMUM_INTERNAL_POINT_COUNT,
  GO_PASS_ALIVE_REQUIRED_VITAL_REGION_COUNT,
  GO_PASS_FEATURE_INDEX_OFFSET,
  GO_PASS_MOVE,
  GO_RECENT_MOVE_FEATURE_INDEXES,
  GO_STUDENT_INPUT_PLANE_COUNT,
  GO_WHITE_COLOR,
} from "./constants";

const createGameState = (): GoGameState => ({
  board: new Int8Array(GO_BOARD_AREA),
  consecutivePassCount: 0,
  koMove: GO_NO_KO_MOVE,
  moveCount: 0,
  moveHistory: [],
  nextColor: GO_BLACK_COLOR,
});

const copyGameState = (gameState: GoGameState): GoGameState => ({
  board: gameState.board.slice(),
  consecutivePassCount: gameState.consecutivePassCount,
  koMove: gameState.koMove,
  moveCount: gameState.moveCount,
  moveHistory: [...gameState.moveHistory],
  nextColor: gameState.nextColor,
});

const getAdjacentMoves = (move: number) => {
  const row = Math.floor(move / GO_BOARD_SIZE);
  const column = move % GO_BOARD_SIZE;
  const adjacentMoves: number[] = [];

  if (row > 0) {
    adjacentMoves.push(move - GO_BOARD_SIZE);
  }

  if (row < GO_BOARD_SIZE - 1) {
    adjacentMoves.push(move + GO_BOARD_SIZE);
  }

  if (column > 0) {
    adjacentMoves.push(move - 1);
  }

  if (column < GO_BOARD_SIZE - 1) {
    adjacentMoves.push(move + 1);
  }

  return adjacentMoves;
};

const getGroup = (board: Int8Array, startingMove: number) => {
  const color = board[startingMove];

  if (color === GO_EMPTY_COLOR) {
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

      if (adjacentColor === GO_EMPTY_COLOR) {
        liberties.add(adjacentMove);
      } else if (adjacentColor === color) {
        pendingMoves.push(adjacentMove);
      }
    }
  }

  return { liberties, stones };
};

const playMove = (gameState: GoGameState, move: number) => {
  if (move === GO_PASS_MOVE) {
    const nextState = copyGameState(gameState);
    nextState.consecutivePassCount += 1;
    nextState.koMove = GO_NO_KO_MOVE;
    nextState.moveCount += 1;
    nextState.moveHistory.push(move);
    nextState.nextColor *= -1;
    return nextState;
  }

  if (gameState.board[move] !== GO_EMPTY_COLOR || gameState.koMove === move) {
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
      nextState.board[capturedMove] = GO_EMPTY_COLOR;
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
      : GO_NO_KO_MOVE;
  nextState.moveCount += 1;
  nextState.moveHistory.push(move);
  nextState.nextColor *= -1;
  return nextState;
};

const getLegalMoves = (gameState: GoGameState) => {
  const legalMoves: number[] = [];

  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    if (playMove(gameState, move)) {
      legalMoves.push(move);
    }
  }

  legalMoves.push(GO_PASS_MOVE);
  return legalMoves;
};

const selectHighestLegalMove = (gameState: GoGameState, logits: Float32Array) => {
  const legalMoves = getLegalMoves(gameState);
  let selectedMove =
    gameState.moveCount >= GO_MINIMUM_PASS_MOVE_COUNT ? GO_PASS_MOVE : legalMoves[0];
  let selectedValue = logits[selectedMove];

  for (const legalMove of legalMoves) {
    if (legalMove === GO_PASS_MOVE && gameState.moveCount < GO_MINIMUM_PASS_MOVE_COUNT) {
      continue;
    }

    if (logits[legalMove] > selectedValue) {
      selectedMove = legalMove;
      selectedValue = logits[legalMove];
    }
  }

  return selectedMove;
};

const getLibertyFeatureIndex = (libertyCount: number, isCurrentPlayer: boolean) => {
  if (libertyCount === 1) {
    return isCurrentPlayer
      ? GO_CURRENT_PLAYER_ONE_LIBERTY_FEATURE_INDEX
      : GO_OPPONENT_ONE_LIBERTY_FEATURE_INDEX;
  }

  if (libertyCount === 2) {
    return isCurrentPlayer
      ? GO_CURRENT_PLAYER_TWO_LIBERTY_FEATURE_INDEX
      : GO_OPPONENT_TWO_LIBERTY_FEATURE_INDEX;
  }

  return null;
};

const encodeStudentFeatures = (gameState: GoGameState) => {
  const features = new Float32Array(GO_BOARD_AREA * GO_STUDENT_INPUT_PLANE_COUNT);
  const visitedMoves = new Set<number>();
  const setFeature = (move: number, featureIndex: number, value = 1) => {
    features[move * GO_STUDENT_INPUT_PLANE_COUNT + featureIndex] = value;
  };

  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    const color = gameState.board[move];

    if (color === GO_EMPTY_COLOR) {
      continue;
    }

    const isCurrentPlayer = color === gameState.nextColor;
    setFeature(move, isCurrentPlayer ? GO_CURRENT_PLAYER_FEATURE_INDEX : GO_OPPONENT_FEATURE_INDEX);

    if (visitedMoves.has(move)) {
      continue;
    }

    const group = getGroup(gameState.board, move);

    for (const stoneMove of group.stones) {
      visitedMoves.add(stoneMove);
    }

    const libertyFeatureIndex = getLibertyFeatureIndex(group.liberties.size, isCurrentPlayer);

    if (libertyFeatureIndex !== null) {
      for (const stoneMove of group.stones) {
        setFeature(stoneMove, libertyFeatureIndex);
      }
    }
  }

  if (gameState.koMove >= 0) {
    setFeature(gameState.koMove, GO_KO_FEATURE_INDEX);
  }

  for (
    let historyOffset = 1;
    historyOffset <= GO_RECENT_MOVE_FEATURE_INDEXES.length;
    historyOffset += 1
  ) {
    const historyMove = gameState.moveHistory.at(-historyOffset);

    if (historyMove !== undefined && historyMove < GO_BOARD_AREA) {
      setFeature(historyMove, GO_RECENT_MOVE_FEATURE_INDEXES[historyOffset - 1]);
    } else if (historyMove === GO_PASS_MOVE) {
      const passFeatureIndex = GO_PASS_FEATURE_INDEX_OFFSET + historyOffset;

      for (let move = 0; move < GO_BOARD_AREA; move += 1) {
        setFeature(move, passFeatureIndex);
      }
    }
  }

  const perspectiveKomi = (-GO_KOMI_POINTS * gameState.nextColor) / GO_KOMI_NORMALIZATION_POINTS;

  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    setFeature(move, GO_KOMI_FEATURE_INDEX, perspectiveKomi);
  }

  return features;
};

const getColorGroups = (board: Int8Array, color: number) => {
  const groupKeyByMove = new Int16Array(GO_BOARD_AREA);
  groupKeyByMove.fill(GO_NO_KO_MOVE);
  const groups = new Map<number, number[]>();
  const visitedMoves = new Set<number>();

  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    if (board[move] !== color || visitedMoves.has(move)) {
      continue;
    }

    const stones = getGroup(board, move).stones;
    const groupKey = stones[0];
    groups.set(groupKey, stones);

    for (const stoneMove of stones) {
      visitedMoves.add(stoneMove);
      groupKeyByMove[stoneMove] = groupKey;
    }
  }

  return { groupKeyByMove, groups };
};

const getNonColorRegions = (board: Int8Array, color: number, groupKeyByMove: Int16Array) => {
  const regions = [];
  const visitedMoves = new Set<number>();

  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    if (board[move] === color || visitedMoves.has(move)) {
      continue;
    }

    const pendingMoves = [move];
    const regionMoves: number[] = [];
    const borderingGroupKeys = new Set<number>();
    let containsOpponent = false;
    let internalPointCount = 0;
    let vitalGroupKeys: Set<number> | null = null;

    while (pendingMoves.length > 0) {
      const regionMove = pendingMoves.pop();

      if (regionMove === undefined || visitedMoves.has(regionMove)) {
        continue;
      }

      visitedMoves.add(regionMove);
      regionMoves.push(regionMove);
      containsOpponent ||= board[regionMove] === -color;
      const adjacentGroupKeys = new Set<number>();

      for (const adjacentMove of getAdjacentMoves(regionMove)) {
        if (board[adjacentMove] === color) {
          const groupKey = groupKeyByMove[adjacentMove];
          adjacentGroupKeys.add(groupKey);
          borderingGroupKeys.add(groupKey);
        } else if (!visitedMoves.has(adjacentMove)) {
          pendingMoves.push(adjacentMove);
        }
      }

      if (adjacentGroupKeys.size === 0) {
        internalPointCount += 1;
      }

      if (board[regionMove] === GO_EMPTY_COLOR) {
        if (vitalGroupKeys === null) {
          vitalGroupKeys = adjacentGroupKeys;
        } else {
          for (const vitalGroupKey of vitalGroupKeys) {
            if (!adjacentGroupKeys.has(vitalGroupKey)) {
              vitalGroupKeys.delete(vitalGroupKey);
            }
          }
        }
      }
    }

    regions.push({
      borderingGroupKeys,
      containsOpponent,
      internalPointCount,
      moves: regionMoves,
      vitalGroupKeys: vitalGroupKeys ?? new Set<number>(),
    });
  }

  return regions;
};

const getPassAliveAnalysis = (board: Int8Array, color: number) => {
  const { groupKeyByMove, groups } = getColorGroups(board, color);
  const regions = getNonColorRegions(board, color, groupKeyByMove);
  const passAliveGroupKeys = new Set(groups.keys());
  const passAliveRegionIndexes = new Set(regions.map((_, regionIndex) => regionIndex));

  while (true) {
    const removedGroupKeys = new Set<number>();

    for (const groupKey of passAliveGroupKeys) {
      let vitalRegionCount = 0;

      for (const regionIndex of passAliveRegionIndexes) {
        if (regions[regionIndex].vitalGroupKeys.has(groupKey)) {
          vitalRegionCount += 1;
        }
      }

      if (vitalRegionCount < GO_PASS_ALIVE_REQUIRED_VITAL_REGION_COUNT) {
        removedGroupKeys.add(groupKey);
      }
    }

    if (removedGroupKeys.size === 0) {
      break;
    }

    for (const removedGroupKey of removedGroupKeys) {
      passAliveGroupKeys.delete(removedGroupKey);
    }

    for (const regionIndex of passAliveRegionIndexes) {
      const doesBorderRemovedGroup = [...removedGroupKeys].some((removedGroupKey) =>
        regions[regionIndex].borderingGroupKeys.has(removedGroupKey),
      );

      if (doesBorderRemovedGroup) {
        passAliveRegionIndexes.delete(regionIndex);
      }
    }
  }

  return { groups, passAliveGroupKeys, regions };
};

const markPassAliveAreaForColor = (board: Int8Array, color: number, area: Int8Array) => {
  const { groups, passAliveGroupKeys, regions } = getPassAliveAnalysis(board, color);

  for (const groupKey of passAliveGroupKeys) {
    for (const stoneMove of groups.get(groupKey) ?? []) {
      area[stoneMove] = color;
    }
  }

  for (const region of regions) {
    const doesBorderNonPassAliveGroup = [...region.borderingGroupKeys].some(
      (groupKey) => !passAliveGroupKeys.has(groupKey),
    );
    const isPassAliveRegion =
      region.internalPointCount <= GO_PASS_ALIVE_MAXIMUM_INTERNAL_POINT_COUNT &&
      !doesBorderNonPassAliveGroup &&
      groups.size > 0;
    const isSafeTerritory =
      !region.containsOpponent && !doesBorderNonPassAliveGroup && groups.size > 0;

    if (isPassAliveRegion || isSafeTerritory) {
      for (const regionMove of region.moves) {
        area[regionMove] = color;
      }
    } else if (!region.containsOpponent && groups.size > 0) {
      for (const regionMove of region.moves) {
        if (area[regionMove] === GO_EMPTY_COLOR) {
          area[regionMove] = color;
        }
      }
    }
  }
};

const getPassAliveArea = (board: Int8Array) => {
  const area = new Int8Array(GO_BOARD_AREA);
  markPassAliveAreaForColor(board, GO_BLACK_COLOR, area);
  markPassAliveAreaForColor(board, GO_WHITE_COLOR, area);
  return area;
};

const getAreaScore = (gameState: GoGameState) => {
  let score = -GO_KOMI_POINTS;
  const visitedMoves = new Set<number>();

  for (const color of gameState.board) {
    score += color;
  }

  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    if (gameState.board[move] !== GO_EMPTY_COLOR || visitedMoves.has(move)) {
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

        if (adjacentColor === GO_EMPTY_COLOR && !visitedMoves.has(adjacentMove)) {
          pendingMoves.push(adjacentMove);
        } else if (adjacentColor !== GO_EMPTY_COLOR) {
          borderingColors.add(adjacentColor);
        }
      }
    }

    if (borderingColors.size === 1 && borderingColors.has(GO_BLACK_COLOR)) {
      score += territoryMoves.length;
    } else if (borderingColors.size === 1 && borderingColors.has(GO_WHITE_COLOR)) {
      score -= territoryMoves.length;
    }
  }

  return score;
};

const removeDeadStones = (gameState: GoGameState, deadMoves: number[]) => {
  const nextState = copyGameState(gameState);

  for (const deadMove of deadMoves) {
    if (deadMove < 0 || deadMove >= GO_BOARD_AREA) {
      throw new Error(`Dead move must be between 0 and ${GO_BOARD_AREA - 1}.`);
    }

    nextState.board[deadMove] = GO_EMPTY_COLOR;
  }

  return nextState;
};

const getBestImmediateCapture = (gameState: GoGameState) => {
  const opponentColor = -gameState.nextColor;
  const opponentStoneCount = gameState.board.filter((color) => color === opponentColor).length;
  let bestCaptureCount = 0;
  let bestNextGameState: GoGameState | null = null;

  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    const nextGameState = playMove(gameState, move);

    if (!nextGameState) {
      continue;
    }

    const nextOpponentStoneCount = nextGameState.board.filter(
      (color) => color === opponentColor,
    ).length;
    const captureCount = opponentStoneCount - nextOpponentStoneCount;

    if (captureCount > bestCaptureCount) {
      bestCaptureCount = captureCount;
      bestNextGameState = nextGameState;
    }
  }

  return bestNextGameState;
};

const getAutomaticallyDeadMoves = (gameState: GoGameState) => {
  const passAliveArea = getPassAliveArea(gameState.board);
  const passDeadMoves: number[] = [];

  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    if (
      gameState.board[move] !== GO_EMPTY_COLOR &&
      passAliveArea[move] === -gameState.board[move]
    ) {
      passDeadMoves.push(move);
    }
  }

  let aftermathGameState = removeDeadStones(gameState, passDeadMoves);
  aftermathGameState.consecutivePassCount = 0;

  for (
    let aftermathMoveCount = 0;
    aftermathMoveCount < GO_BOARD_AREA &&
    aftermathGameState.consecutivePassCount < GO_GAME_OVER_PASS_COUNT;
    aftermathMoveCount += 1
  ) {
    const captureGameState = getBestImmediateCapture(aftermathGameState);
    const nextGameState = captureGameState ?? playMove(aftermathGameState, GO_PASS_MOVE);

    if (!nextGameState) {
      break;
    }

    aftermathGameState = nextGameState;
  }

  const deadMoves = [...passDeadMoves];

  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    if (
      gameState.board[move] !== GO_EMPTY_COLOR &&
      aftermathGameState.board[move] === GO_EMPTY_COLOR
    ) {
      deadMoves.push(move);
    }
  }

  return [...new Set(deadMoves)];
};

const isGameOver = (gameState: GoGameState) =>
  gameState.consecutivePassCount >= GO_GAME_OVER_PASS_COUNT ||
  gameState.moveCount >= GO_MAXIMUM_MOVE_COUNT;

export {
  createGameState,
  encodeStudentFeatures,
  getAutomaticallyDeadMoves,
  getAreaScore,
  getLegalMoves,
  getPassAliveArea,
  isGameOver,
  playMove,
  removeDeadStones,
  selectHighestLegalMove,
};
