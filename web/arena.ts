import {
  GO_ARENA_BLACK_COLOR,
  GO_ARENA_BOARD_AREA,
  GO_ARENA_BOARD_SIZE,
  GO_ARENA_DEFAULT_EXPERIMENT_COUNT,
  GO_ARENA_EXPERIMENT_RENDER_INTERVAL,
  GO_ARENA_MAXIMUM_EXPERIMENT_COUNT,
  GO_ARENA_MAXIMUM_MOVE_COUNT,
  GO_ARENA_MILLISECONDS_PER_SECOND,
  GO_ARENA_MINIMUM_EXPERIMENT_COUNT,
  GO_ARENA_OPENING_INDEX_MULTIPLIER,
  GO_ARENA_OPENING_MOVE_COUNT,
  GO_ARENA_OPENING_MOVE_MULTIPLIER,
  GO_ARENA_OPENING_PAIR_SIZE,
  GO_ARENA_PERCENT_MULTIPLIER,
  GO_ARENA_PROBABILITY_DECIMAL_COUNT,
  GO_ARENA_STUDENT_NAME,
} from "./constants";
import { KataGoArenaEngine, StudentArenaEngine } from "./engines";
import { createGameState, getAreaScore, getLegalMoves, isGameOver, playMove } from "./game";

const completedCountElement = document.querySelector("#completed-count");
const experimentCountInput = document.querySelector<HTMLInputElement>("#experiment-count");
const experimentForm = document.querySelector<HTMLFormElement>("#experiment-form");
const experimentGridElement = document.querySelector("#experiment-grid");
const gameResultsBodyElement = document.querySelector("#game-results-body");
const kataGoBrierScoreElement = document.querySelector("#katago-brier-score");
const kataGoWinsElement = document.querySelector("#katago-wins");
const moveCapCountElement = document.querySelector("#move-cap-count");
const runHistoryBodyElement = document.querySelector("#run-history-body");
const runExperimentsButton = document.querySelector<HTMLButtonElement>("#run-experiments");
const statusElement = document.querySelector("#status");
const studentBrierScoreElement = document.querySelector("#student-brier-score");
const studentWinsElement = document.querySelector("#student-wins");
let runNumber = 0;

const updateText = (element: Element | null, text: string) => {
  if (element) {
    element.textContent = text;
  }
};

const waitForAnimationFrame = () =>
  new Promise<void>((resolve) => {
    window.requestAnimationFrame(() => resolve());
  });

const formatProbability = (probability: number | null) =>
  probability === null
    ? "—"
    : `${(probability * GO_ARENA_PERCENT_MULTIPLIER).toFixed(GO_ARENA_PROBABILITY_DECIMAL_COUNT)}%`;

const createTableCell = (text: string, className = "px-4 py-2") => {
  const cellElement = document.createElement("td");
  cellElement.className = className;
  cellElement.textContent = text;
  return cellElement;
};

const didStudentWin = (experiment: GoArenaExperiment) =>
  experiment.result !== null &&
  ((experiment.result.winner === "black" && experiment.isStudentBlack) ||
    (experiment.result.winner === "white" && !experiment.isStudentBlack));

const getStudentWinProbability = (experiment: GoArenaExperiment, evaluation: GoArenaEvaluation) => {
  const isStudentTurn =
    (experiment.gameState.nextColor === GO_ARENA_BLACK_COLOR && experiment.isStudentBlack) ||
    (experiment.gameState.nextColor !== GO_ARENA_BLACK_COLOR && !experiment.isStudentBlack);
  return isStudentTurn ? evaluation.winProbability : 1 - evaluation.winProbability;
};

const renderBoard = (boardElement: Element, gameState: GoArenaGameState) => {
  const gridElement = document.createElement("div");
  gridElement.className = "go-grid";

  for (let move = 0; move < GO_ARENA_BOARD_AREA; move += 1) {
    const color = gameState.board[move];

    if (color === 0) {
      continue;
    }

    const stoneElement = document.createElement("span");
    const row = Math.floor(move / GO_ARENA_BOARD_SIZE);
    const column = move % GO_ARENA_BOARD_SIZE;
    stoneElement.className =
      color === GO_ARENA_BLACK_COLOR ? "go-stone go-stone-black" : "go-stone go-stone-white";
    stoneElement.style.left = `${(column / (GO_ARENA_BOARD_SIZE - 1)) * 100}%`;
    stoneElement.style.top = `${(row / (GO_ARENA_BOARD_SIZE - 1)) * 100}%`;
    gridElement.append(stoneElement);
  }

  boardElement.replaceChildren(gridElement);
};

const createExperimentElements = (experiments: GoArenaExperiment[]) => {
  if (!experimentGridElement) {
    return;
  }

  const fragment = document.createDocumentFragment();

  for (const experiment of experiments) {
    const articleElement = document.createElement("article");
    const labelElement = document.createElement("p");
    const boardElement = document.createElement("div");
    articleElement.className = "min-w-0";
    labelElement.className = "mb-1 truncate text-[11px] text-neutral-500";
    labelElement.dataset.experimentLabel = String(experiment.id);
    labelElement.textContent = `#${experiment.id + 1} · ${
      experiment.isStudentBlack ? "Moka B" : "Moka W"
    }`;
    boardElement.className = "relative aspect-square w-full border border-neutral-200 bg-white";
    boardElement.dataset.experimentBoard = String(experiment.id);
    articleElement.append(labelElement, boardElement);
    fragment.append(articleElement);
  }

  experimentGridElement.replaceChildren(fragment);
};

const renderExperiments = (experiments: GoArenaExperiment[]) => {
  for (const experiment of experiments) {
    const boardElement = document.querySelector(`[data-experiment-board="${experiment.id}"]`);
    const labelElement = document.querySelector(`[data-experiment-label="${experiment.id}"]`);

    if (boardElement) {
      renderBoard(boardElement, experiment.gameState);
    }

    if (labelElement && experiment.result) {
      labelElement.textContent = `#${experiment.id + 1} · ${
        didStudentWin(experiment) ? "Moka" : "KataGo"
      } +${Math.abs(experiment.result.score).toFixed(1)}`;
      labelElement.className = `mb-1 truncate text-[11px] ${
        didStudentWin(experiment) ? "text-neutral-900" : "text-neutral-400"
      }`;
    }
  }
};

const updateSummary = (experiments: GoArenaExperiment[]) => {
  const completedExperiments = experiments.filter((experiment) => experiment.result !== null);
  const studentWinCount = completedExperiments.filter((experiment) => {
    if (!experiment.result) {
      return false;
    }

    return didStudentWin(experiment);
  }).length;
  const moveCapCount = completedExperiments.filter(
    (experiment) => experiment.gameState.moveCount >= GO_ARENA_MAXIMUM_MOVE_COUNT,
  ).length;
  updateText(completedCountElement, `${completedExperiments.length} / ${experiments.length}`);
  updateText(studentWinsElement, String(studentWinCount));
  updateText(kataGoWinsElement, String(completedExperiments.length - studentWinCount));
  updateText(moveCapCountElement, String(moveCapCount));
};

const calculateBrierScore = (
  experiments: GoArenaExperiment[],
  getProbability: (experiment: GoArenaExperiment) => number | null,
) => {
  const squaredErrors = experiments.flatMap((experiment) => {
    const probability = getProbability(experiment);

    if (probability === null || !experiment.result) {
      return [];
    }

    const outcome = didStudentWin(experiment) ? 1 : 0;
    return [(probability - outcome) ** 2];
  });

  return squaredErrors.reduce((sum, squaredError) => sum + squaredError, 0) / squaredErrors.length;
};

const renderGameResults = (experiments: GoArenaExperiment[]) => {
  if (!gameResultsBodyElement) {
    return;
  }

  const fragment = document.createDocumentFragment();

  for (const experiment of experiments) {
    if (!experiment.result) {
      continue;
    }

    const rowElement = document.createElement("tr");
    const winnerName = didStudentWin(experiment) ? GO_ARENA_STUDENT_NAME : "KataGo";
    rowElement.append(
      createTableCell(String(experiment.id + 1), "py-2 pr-4 text-neutral-500"),
      createTableCell(experiment.isStudentBlack ? "Black" : "White"),
      createTableCell(formatProbability(experiment.studentInitialWinProbability)),
      createTableCell(formatProbability(experiment.kataGoInitialWinProbability)),
      createTableCell(winnerName),
      createTableCell(Math.abs(experiment.result.score).toFixed(1)),
      createTableCell(String(experiment.result.moveCount)),
      createTableCell(
        experiment.result.moveCount >= GO_ARENA_MAXIMUM_MOVE_COUNT ? "Move cap" : "Two passes",
        "py-2 pl-4",
      ),
    );
    fragment.append(rowElement);
  }

  gameResultsBodyElement.replaceChildren(fragment);
};

const renderRunSummary = (summary: GoArenaRunSummary) => {
  updateText(studentBrierScoreElement, summary.studentBrierScore.toFixed(3));
  updateText(kataGoBrierScoreElement, summary.kataGoBrierScore.toFixed(3));

  if (!runHistoryBodyElement) {
    return;
  }

  const rowElement = document.createElement("tr");
  rowElement.append(
    createTableCell(String(summary.runNumber), "py-2 pr-4 text-neutral-500"),
    createTableCell(String(summary.experimentCount)),
    createTableCell(String(summary.studentWinCount)),
    createTableCell(String(summary.kataGoWinCount)),
    createTableCell(String(summary.moveCapCount)),
    createTableCell(summary.studentBrierScore.toFixed(3)),
    createTableCell(summary.kataGoBrierScore.toFixed(3)),
    createTableCell(
      `${(summary.durationMs / GO_ARENA_MILLISECONDS_PER_SECOND).toFixed(1)}s`,
      "py-2 pl-4",
    ),
  );
  runHistoryBodyElement.prepend(rowElement);
};

const completeExperiment = (experiment: GoArenaExperiment) => {
  const score = getAreaScore(experiment.gameState);
  const isBlackWinner = score > 0;
  const blackName = experiment.isStudentBlack ? GO_ARENA_STUDENT_NAME : "KataGo";
  const whiteName = experiment.isStudentBlack ? "KataGo" : GO_ARENA_STUDENT_NAME;
  experiment.result = {
    blackName,
    moveCount: experiment.gameState.moveCount,
    score,
    whiteName,
    winner: isBlackWinner ? "black" : "white",
  };
};

const createOpeningGameState = (experimentIndex: number) => {
  const openingIndex = Math.floor(experimentIndex / GO_ARENA_OPENING_PAIR_SIZE);
  let gameState = createGameState();

  for (
    let openingMoveIndex = 0;
    openingMoveIndex < GO_ARENA_OPENING_MOVE_COUNT;
    openingMoveIndex += 1
  ) {
    const legalMoves = getLegalMoves(gameState).filter((move) => move < GO_ARENA_BOARD_AREA);
    const selectedMoveIndex =
      (openingIndex * GO_ARENA_OPENING_INDEX_MULTIPLIER +
        openingMoveIndex * GO_ARENA_OPENING_MOVE_MULTIPLIER) %
      legalMoves.length;
    const nextState = playMove(gameState, legalMoves[selectedMoveIndex]);

    if (!nextState) {
      break;
    }

    gameState = nextState;
  }

  return gameState;
};

const runExperiments = async (
  experimentCount: number,
  studentEngine: StudentArenaEngine,
  kataGoEngine: KataGoArenaEngine,
) => {
  const experiments = Array.from(
    { length: experimentCount },
    (_, experimentIndex): GoArenaExperiment => ({
      gameState: createOpeningGameState(experimentIndex),
      id: experimentIndex,
      isStudentBlack: experimentIndex % 2 === 0,
      kataGoInitialWinProbability: null,
      result: null,
      studentInitialWinProbability: null,
    }),
  );
  const startTime = performance.now();
  let roundIndex = 0;
  createExperimentElements(experiments);
  renderExperiments(experiments);
  updateSummary(experiments);
  updateText(studentBrierScoreElement, "—");
  updateText(kataGoBrierScoreElement, "—");
  updateText(statusElement, `Calibrating ${experimentCount} openings…`);

  const gameStates = experiments.map((experiment) => experiment.gameState);
  const [studentEvaluations, kataGoEvaluations] = await Promise.all([
    studentEngine.evaluateStates(gameStates),
    kataGoEngine.evaluateStates(gameStates),
  ]);

  for (let experimentIndex = 0; experimentIndex < experiments.length; experimentIndex += 1) {
    const experiment = experiments[experimentIndex];
    experiment.studentInitialWinProbability = getStudentWinProbability(
      experiment,
      studentEvaluations[experimentIndex],
    );
    experiment.kataGoInitialWinProbability = getStudentWinProbability(
      experiment,
      kataGoEvaluations[experimentIndex],
    );
  }

  updateText(statusElement, `Running ${experimentCount} games…`);

  while (experiments.some((experiment) => experiment.result === null)) {
    const studentExperiments: GoArenaExperiment[] = [];
    const kataGoExperiments: GoArenaExperiment[] = [];

    for (const experiment of experiments) {
      if (experiment.result) {
        continue;
      }

      const isStudentTurn =
        (experiment.gameState.nextColor === GO_ARENA_BLACK_COLOR && experiment.isStudentBlack) ||
        (experiment.gameState.nextColor !== GO_ARENA_BLACK_COLOR && !experiment.isStudentBlack);
      (isStudentTurn ? studentExperiments : kataGoExperiments).push(experiment);
    }

    const [studentMoves, kataGoMoves] = await Promise.all([
      studentEngine.selectMoves(studentExperiments.map((experiment) => experiment.gameState)),
      kataGoEngine.selectMoves(kataGoExperiments.map((experiment) => experiment.gameState)),
    ]);

    for (let moveIndex = 0; moveIndex < studentExperiments.length; moveIndex += 1) {
      const experiment = studentExperiments[moveIndex];
      const nextState = playMove(experiment.gameState, studentMoves[moveIndex]);

      if (!nextState) {
        throw new Error("Moka selected an illegal move.");
      }

      experiment.gameState = nextState;
    }

    for (let moveIndex = 0; moveIndex < kataGoExperiments.length; moveIndex += 1) {
      const experiment = kataGoExperiments[moveIndex];
      const nextState = playMove(experiment.gameState, kataGoMoves[moveIndex]);

      if (!nextState) {
        throw new Error("KataGo selected an illegal move.");
      }

      experiment.gameState = nextState;
    }

    for (const experiment of experiments) {
      if (!experiment.result && isGameOver(experiment.gameState)) {
        completeExperiment(experiment);
      }
    }

    roundIndex += 1;

    if (roundIndex % GO_ARENA_EXPERIMENT_RENDER_INTERVAL === 0) {
      renderExperiments(experiments);
      updateSummary(experiments);
      await waitForAnimationFrame();
    }
  }

  renderExperiments(experiments);
  updateSummary(experiments);
  const studentWinCount = experiments.filter(didStudentWin).length;
  const moveCapCount = experiments.filter(
    (experiment) => experiment.gameState.moveCount >= GO_ARENA_MAXIMUM_MOVE_COUNT,
  ).length;
  const summary: GoArenaRunSummary = {
    durationMs: performance.now() - startTime,
    experimentCount,
    kataGoBrierScore: calculateBrierScore(
      experiments,
      (experiment) => experiment.kataGoInitialWinProbability,
    ),
    kataGoWinCount: experiments.length - studentWinCount,
    moveCapCount,
    runNumber: (runNumber += 1),
    studentBrierScore: calculateBrierScore(
      experiments,
      (experiment) => experiment.studentInitialWinProbability,
    ),
    studentWinCount,
  };
  renderGameResults(experiments);
  renderRunSummary(summary);
  updateText(statusElement, `${experimentCount} games complete.`);
};

const initializeArena = async () => {
  if (!experimentForm || !experimentCountInput || !runExperimentsButton) {
    return;
  }

  const studentEngine = new StudentArenaEngine();
  const kataGoEngine = new KataGoArenaEngine();

  try {
    await Promise.all([studentEngine.initialize(), kataGoEngine.initialize()]);
    updateText(statusElement, "Ready.");
    runExperimentsButton.disabled = false;
    experimentForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const parsedExperimentCount = Number.parseInt(experimentCountInput.value, 10);
      const experimentCount = Math.min(
        GO_ARENA_MAXIMUM_EXPERIMENT_COUNT,
        Math.max(
          GO_ARENA_MINIMUM_EXPERIMENT_COUNT,
          Number.isFinite(parsedExperimentCount)
            ? parsedExperimentCount
            : GO_ARENA_DEFAULT_EXPERIMENT_COUNT,
        ),
      );
      experimentCountInput.value = String(experimentCount);
      runExperimentsButton.disabled = true;
      void runExperiments(experimentCount, studentEngine, kataGoEngine)
        .catch((error) => {
          updateText(
            statusElement,
            error instanceof Error ? error.message : "Experiment run failed.",
          );
        })
        .finally(() => {
          runExperimentsButton.disabled = false;
        });
    });
  } catch (error) {
    updateText(
      statusElement,
      error instanceof Error ? error.message : "Unable to initialize the arena.",
    );
  }
};

void initializeArena();
