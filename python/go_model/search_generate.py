import argparse
import json
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from go_model.arena import (
    create_opening_game_state,
    should_resign_selected_pass,
)
from go_model.board import GameState, get_legal_moves, is_game_over, play_move
from go_model.blunder import calculate_game_blunder_risks
from go_model.collect import (
    evaluate_moka_batch,
    sample_rollout_move,
    select_greedy_rollout_move,
)
from go_model.config import (
    BOARD_AREA,
    BOARD_SIZE,
    DEFAULT_RANDOM_SEED,
    DISTILLATION_OPENING_OFFSET,
    KATAGO_SIMPLE_AREA_RULES,
    KOMI_POINTS,
    MINIMUM_TEACHER_PASS_MOVE_COUNT,
    MID_GAME_MOVE_COUNT,
    ON_POLICY_BATCH_SIZE,
    OPENING_MOVE_COUNT,
    POLICY_MOVE_COUNT,
    SEARCH_TEACHER_ANALYSIS_THREAD_COUNT,
    SEARCH_TEACHER_CACHE_POWER,
    SEARCH_TEACHER_DEFAULT_GAME_COUNT,
    SEARCH_TEACHER_DEFAULT_VISIT_COUNT,
    SEARCH_TEACHER_MAX_BATCH_SIZE,
    SEARCH_TEACHER_QUERY_BATCH_SIZE,
    SEARCH_TEACHER_ROLLOUT_SIMULATION_COUNT,
    SEARCH_TEACHER_SELECTION_FRACTION,
    SEARCH_TEACHER_SHUTDOWN_TIMEOUT_SECONDS,
    SEARCH_TEACHER_THREADS_PER_ANALYSIS,
    SEARCH_TEACHER_UNIFORM_SELECTION_FRACTION,
    SEARCH_TEACHER_VALUE_DISAGREEMENT_WEIGHT,
    SEARCH_RESIGNATION_AREA_MARGIN_POINTS,
    SEARCH_ROOT_POLICY_TEMPERATURE,
    SEARCH_ROOT_SYMMETRY_GEOMETRIC_POLICY_WEIGHT,
)
from go_model.features import encode_moka_features
from go_model.model import MokaNestedNetwork
from go_model.search import MokaEvaluator, MokaSearchSession
from go_model.teacher import KataGoTeacher


@dataclass
class AnalysisTargets:
    policy: np.ndarray
    q_values: np.ndarray
    q_weights: np.ndarray
    value: float
    child_features: list[np.ndarray]
    child_values: list[float]
    child_weights: list[float]
    ownership: np.ndarray | None
    score: float | None


class KataGoAnalysisEngine:
    def __init__(
        self,
        executable_path: Path,
        config_path: Path,
        model_path: Path,
    ) -> None:
        override_config = ",".join(
            [
                "reportAnalysisWinratesAs=SIDETOMOVE",
                f"numAnalysisThreads={SEARCH_TEACHER_ANALYSIS_THREAD_COUNT}",
                (
                    "numSearchThreadsPerAnalysisThread="
                    f"{SEARCH_TEACHER_THREADS_PER_ANALYSIS}"
                ),
                f"nnMaxBatchSize={SEARCH_TEACHER_MAX_BATCH_SIZE}",
                f"nnCacheSizePowerOfTwo={SEARCH_TEACHER_CACHE_POWER}",
                f"maxBoardXSizeForNNBuffer={BOARD_SIZE}",
                f"maxBoardYSizeForNNBuffer={BOARD_SIZE}",
            ]
        )
        self.process = subprocess.Popen(
            [
                str(executable_path),
                "analysis",
                "-config",
                str(config_path),
                "-model",
                str(model_path),
                "-override-config",
                override_config,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if (
            self.process.stdin is None
            or self.process.stdout is None
            or self.process.stderr is None
        ):
            raise RuntimeError("Failed to open KataGo analysis pipes.")
        self.standard_input = self.process.stdin
        self.standard_output = self.process.stdout
        self.error_lines: list[str] = []
        self.error_thread = threading.Thread(
            target=self.capture_errors,
            daemon=True,
        )
        self.error_thread.start()

    def capture_errors(self) -> None:
        if self.process.stderr is None:
            return

        for line in self.process.stderr:
            self.error_lines.append(line.rstrip())

    def analyze(
        self,
        queries: list[dict],
        expected_response_count: int,
    ) -> list[dict]:
        for query in queries:
            self.standard_input.write(json.dumps(query, separators=(",", ":")) + "\n")
        self.standard_input.flush()
        responses: list[dict] = []

        while len(responses) < expected_response_count:
            line = self.standard_output.readline()

            if not line:
                error_tail = "\n".join(self.error_lines[-20:])
                raise RuntimeError(
                    f"KataGo analysis stopped unexpectedly.\n{error_tail}"
                )

            response = json.loads(line)

            if "error" in response:
                raise RuntimeError(response["error"])

            if "warning" in response or response.get("isDuringSearch"):
                continue

            responses.append(response)

        return responses

    def close(self) -> None:
        self.standard_input.close()
        self.process.wait(timeout=SEARCH_TEACHER_SHUTDOWN_TIMEOUT_SECONDS)
        self.error_thread.join(timeout=SEARCH_TEACHER_SHUTDOWN_TIMEOUT_SECONDS)


def move_to_coordinate(move: int) -> str:
    if move == BOARD_AREA:
        return "pass"

    row, column = divmod(move, BOARD_SIZE)
    return f"({column},{row})"


def coordinate_to_move(coordinate: str) -> int:
    if coordinate.lower() == "pass":
        return BOARD_AREA

    if coordinate.startswith("("):
        column, row = (
            int(value)
            for value in coordinate.strip("()").split(",")
        )
        return row * BOARD_SIZE + column

    column_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    column = column_letters.index(coordinate[0].upper())
    row = BOARD_SIZE - int(coordinate[1:])
    return row * BOARD_SIZE + column


def is_moka_turn(game_id: int, game_state: GameState) -> bool:
    is_moka_black = game_id % 2 == 0
    return (game_state.next_color == 1) == is_moka_black


def generate_rollout_games(
    checkpoint_path: Path,
    opponent_path: Path,
    game_count: int,
    batch_size: int,
    random_seed: int,
    use_deterministic_rollouts: bool,
    opening_offset: int | None,
    use_action_regret: bool,
    rollout_simulation_count: int,
) -> tuple[list[list[GameState]], list[list[int]], list[list[float]]]:
    random_generator = np.random.default_rng(random_seed)
    model = MokaNestedNetwork()
    model.load_weights(str(checkpoint_path))
    model.eval()
    opponent = KataGoTeacher(opponent_path)
    evaluator = MokaEvaluator(model)
    root_evaluator = MokaEvaluator(
        model,
        use_symmetry_ensemble=True,
        policy_temperature=SEARCH_ROOT_POLICY_TEMPERATURE,
        symmetry_geometric_policy_weight=(
            SEARCH_ROOT_SYMMETRY_GEOMETRIC_POLICY_WEIGHT
        ),
    )
    game_state_histories: list[list[GameState]] = [
        [] for _ in range(game_count)
    ]
    game_moves: list[list[int]] = [[] for _ in range(game_count)]
    game_surprises: list[list[float]] = [[] for _ in range(game_count)]
    active_game_ids = list(range(min(batch_size, game_count)))
    game_states = [
        (
            create_opening_game_state(game_id, opening_offset)
            if opening_offset is not None
            else GameState()
        )
        for game_id in active_game_ids
    ]
    search_sessions = [
        MokaSearchSession(
            evaluator,
            root_evaluator=root_evaluator,
        )
        for _ in active_game_ids
    ]
    for game_id, game_state in zip(active_game_ids, game_states, strict=True):
        game_moves[game_id] = game_state.move_history.copy()
    next_game_id = len(active_game_ids)
    completed_game_count = 0
    reported_completed_game_count = 0

    while game_states:
        moka_evaluations = evaluate_moka_batch(model, game_states, False)
        opponent_evaluations = opponent.evaluate_batch(game_states)

        for game_index in range(len(game_states) - 1, -1, -1):
            game_state = game_states[game_index]
            game_id = active_game_ids[game_index]
            game_state_histories[game_id].append(game_state.copy())
            moka_policy, moka_value = moka_evaluations[game_index]
            opponent_policy, opponent_value = opponent_evaluations[game_index]
            positive_policy_mask = opponent_policy > 0
            policy_surprise = float(
                np.sum(
                    opponent_policy[positive_policy_mask]
                    * (
                        np.log(opponent_policy[positive_policy_mask])
                        - np.log(
                            np.maximum(
                                moka_policy[positive_policy_mask],
                                np.finfo(np.float32).tiny,
                            )
                        )
                    )
                )
            )
            game_surprises[game_id].append(
                policy_surprise
                + SEARCH_TEACHER_VALUE_DISAGREEMENT_WEIGHT
                * abs(opponent_value - moka_value)
            )
            is_current_moka_turn = is_moka_turn(game_id, game_state)
            policy = (
                moka_policy
                if is_current_moka_turn
                else opponent_policy
            )
            move = (
                search_sessions[game_index].select_move(
                    game_state,
                    rollout_simulation_count,
                )
                if is_current_moka_turn and rollout_simulation_count > 0
                else (
                    select_greedy_rollout_move(game_state, policy)
                    if use_deterministic_rollouts
                    else sample_rollout_move(
                        game_state,
                        policy,
                        random_generator,
                    )
                )
            )
            game_moves[game_id].append(move)
            did_resign = (
                is_current_moka_turn
                and should_resign_selected_pass(
                    game_state,
                    move,
                    SEARCH_RESIGNATION_AREA_MARGIN_POINTS,
                )
            )
            next_state = (
                None
                if did_resign
                else play_move(game_state, move)
            )

            if next_state is not None and not is_game_over(next_state):
                game_states[game_index] = next_state
                continue

            completed_game_count += 1
            if next_game_id < game_count:
                game_states[game_index] = (
                    create_opening_game_state(next_game_id, opening_offset)
                    if opening_offset is not None
                    else GameState()
                )
                game_moves[next_game_id] = (
                    game_states[game_index].move_history.copy()
                )
                search_sessions[game_index] = MokaSearchSession(
                    evaluator,
                    root_evaluator=root_evaluator,
                )
                active_game_ids[game_index] = next_game_id
                next_game_id += 1
            else:
                del game_states[game_index]
                del active_game_ids[game_index]
                del search_sessions[game_index]

        if completed_game_count >= reported_completed_game_count + batch_size:
            print(f"generated {completed_game_count:,}/{game_count:,} games")
            reported_completed_game_count = completed_game_count

    if use_action_regret:
        game_surprises = calculate_action_regrets(
            opponent,
            game_state_histories,
            game_moves,
            batch_size,
        )

    return game_state_histories, game_moves, game_surprises


def calculate_action_regrets(
    opponent: KataGoTeacher,
    game_state_histories: list[list[GameState]],
    game_moves: list[list[int]],
    batch_size: int,
) -> list[list[float]]:
    parent_states: list[GameState] = []
    child_states: list[GameState] = []
    game_turns: list[tuple[int, int]] = []

    for game_id, game_state_history in enumerate(game_state_histories):
        prefix_length = len(game_moves[game_id]) - len(game_state_history)

        for turn_number, game_state in enumerate(game_state_history):
            move = game_moves[game_id][prefix_length + turn_number]
            child_state = play_move(game_state, move)

            if child_state is None:
                raise RuntimeError("Rollout selected an illegal move.")

            parent_states.append(game_state)
            child_states.append(child_state)
            game_turns.append((game_id, turn_number))

    regrets = [
        [0.0 for _ in game_state_history]
        for game_state_history in game_state_histories
    ]

    for batch_start in range(0, len(parent_states), batch_size):
        batch_end = batch_start + batch_size
        parent_evaluations = opponent.evaluate_batch(
            parent_states[batch_start:batch_end]
        )
        child_evaluations = opponent.evaluate_batch(
            child_states[batch_start:batch_end]
        )

        for batch_index, (
            parent_evaluation,
            child_evaluation,
        ) in enumerate(
            zip(parent_evaluations, child_evaluations, strict=True)
        ):
            game_id, turn_number = game_turns[batch_start + batch_index]
            regrets[game_id][turn_number] = max(
                0.0,
                parent_evaluation[1] + child_evaluation[1],
            )

    return regrets


def select_analysis_turns(
    surprises: list[float],
    selection_fraction: float,
    uniform_selection_fraction: float,
    random_generator: np.random.Generator,
    eligible_turns: list[int] | None = None,
) -> list[int]:
    all_turns = np.asarray(
        eligible_turns
        if eligible_turns is not None
        else list(range(len(surprises)))
    )

    if len(all_turns) == 0:
        return []

    selected_count = max(1, round(len(all_turns) * selection_fraction))
    uniform_count = round(selected_count * uniform_selection_fraction)
    uniform_turns = (
        random_generator.choice(
            all_turns,
            size=uniform_count,
            replace=False,
        ).tolist()
        if uniform_count > 0
        else []
    )
    uniform_turn_set = set(uniform_turns)
    remaining_turns = [
        turn
        for turn in all_turns
        if int(turn) not in uniform_turn_set
    ]
    surprise_count = selected_count - uniform_count
    surprising_turns = sorted(
        remaining_turns,
        key=lambda turn: surprises[int(turn)],
        reverse=True,
    )[:surprise_count]
    return sorted([int(turn) for turn in uniform_turns + surprising_turns])


def get_eligible_analysis_turns(
    game_state_history: list[GameState],
    game_id: int,
    use_middle_game_reanalysis: bool,
    use_moka_turns_only: bool,
) -> list[int] | None:
    if not use_middle_game_reanalysis and not use_moka_turns_only:
        return None

    return [
        turn_number
        for turn_number, game_state in enumerate(game_state_history)
        if (
            (
                not use_middle_game_reanalysis
                or (
                    OPENING_MOVE_COUNT
                    <= game_state.move_count
                    < MID_GAME_MOVE_COUNT
                )
            )
            and (
                not use_moka_turns_only
                or is_moka_turn(game_id, game_state)
            )
        )
    ]


def validate_reanalysis_modes(
    use_selective_reanalysis: bool,
    use_action_regret: bool,
    use_blunder_risk_reanalysis: bool,
) -> None:
    if use_blunder_risk_reanalysis and not use_selective_reanalysis:
        raise ValueError(
            "Blunder-risk reanalysis requires selective reanalysis."
        )

    if use_blunder_risk_reanalysis and use_action_regret:
        raise ValueError(
            "Blunder-risk and regret reanalysis cannot be combined."
        )


def create_analysis_query(
    game_id: int | str,
    moves: list[int],
    visit_count: int,
    analyze_turns: list[int] | None = None,
    include_auxiliary_targets: bool = False,
) -> dict:
    query = {
        "id": str(game_id),
        "moves": [
            [
                "B" if move_index % 2 == 0 else "W",
                move_to_coordinate(move),
            ]
            for move_index, move in enumerate(moves)
        ],
        "rules": KATAGO_SIMPLE_AREA_RULES,
        "komi": KOMI_POINTS,
        "boardXSize": BOARD_SIZE,
        "boardYSize": BOARD_SIZE,
        "analyzeTurns": (
            analyze_turns
            if analyze_turns is not None
            else list(range(len(moves)))
        ),
        "maxVisits": visit_count,
        "analysisPVLen": 1,
    }
    if include_auxiliary_targets:
        query["includeOwnership"] = True
    return query


def extract_auxiliary_targets(
    response: dict,
) -> tuple[np.ndarray, float]:
    ownership = np.asarray(response["ownership"], dtype=np.float32)
    if ownership.shape != (BOARD_AREA,):
        raise ValueError(
            f"Expected {BOARD_AREA} ownership values, received {ownership.shape}."
        )
    return (
        ownership.reshape(BOARD_SIZE, BOARD_SIZE),
        float(response["rootInfo"]["scoreLead"]),
    )


def extract_analysis_targets(
    response: dict,
    game_state: GameState,
    include_auxiliary_targets: bool,
) -> AnalysisTargets | None:
    policy = np.zeros(POLICY_MOVE_COUNT, dtype=np.float32)
    q_values = np.zeros(POLICY_MOVE_COUNT, dtype=np.float32)
    q_weights = np.zeros(POLICY_MOVE_COUNT, dtype=np.float32)
    child_features: list[np.ndarray] = []
    child_values: list[float] = []
    child_weights: list[float] = []

    for move_information in response["moveInfos"]:
        move = coordinate_to_move(move_information["move"])
        parent_value = 2 * float(move_information["winrate"]) - 1
        edge_visit_count = float(move_information["edgeVisits"])
        policy[move] = edge_visit_count
        q_values[move] = parent_value
        q_weights[move] = edge_visit_count
        child_state = play_move(game_state, move)

        if child_state is not None:
            child_features.append(encode_moka_features(child_state))
            child_values.append(convert_parent_value_to_child(parent_value))
            child_weights.append(edge_visit_count)

    policy_sum = float(np.sum(policy))
    if policy_sum == 0:
        return None
    policy /= policy_sum
    ownership, score = (
        extract_auxiliary_targets(response)
        if include_auxiliary_targets
        else (None, None)
    )
    return AnalysisTargets(
        policy=policy,
        q_values=q_values,
        q_weights=q_weights,
        value=2 * float(response["rootInfo"]["winrate"]) - 1,
        child_features=child_features,
        child_values=child_values,
        child_weights=child_weights,
        ownership=ownership,
        score=score,
    )


def convert_parent_value_to_child(value: float) -> float:
    return -value


def create_search_dataset(
    checkpoint_path: Path,
    opponent_path: Path,
    executable_path: Path,
    config_path: Path,
    model_path: Path,
    game_count: int,
    visit_count: int,
    batch_size: int,
    random_seed: int,
    use_selective_reanalysis: bool,
    use_counterfactual_reanalysis: bool,
    use_deterministic_rollouts: bool,
    opening_offset: int | None,
    use_action_regret: bool,
    use_middle_game_reanalysis: bool,
    rollout_simulation_count: int,
    use_moka_turns_only: bool,
    include_auxiliary_targets: bool,
    use_teacher_branches: bool,
    use_blunder_risk_reanalysis: bool,
) -> dict[str, np.ndarray]:
    validate_reanalysis_modes(
        use_selective_reanalysis,
        use_action_regret,
        use_blunder_risk_reanalysis,
    )

    if use_counterfactual_reanalysis and use_teacher_branches:
        raise ValueError(
            "Teacher branches cannot be combined with counterfactual reanalysis."
        )

    game_state_histories, game_moves, game_surprises = generate_rollout_games(
        checkpoint_path,
        opponent_path,
        game_count,
        batch_size,
        random_seed,
        use_deterministic_rollouts,
        opening_offset,
        use_action_regret,
        rollout_simulation_count,
    )
    random_generator = np.random.default_rng(random_seed)
    eligible_analysis_turns = [
        get_eligible_analysis_turns(
            game_state_histories[game_id],
            game_id,
            use_middle_game_reanalysis,
            use_moka_turns_only,
        )
        for game_id in range(game_count)
    ]
    if use_blunder_risk_reanalysis:
        game_surprises = calculate_game_blunder_risks(
            checkpoint_path,
            game_state_histories,
            eligible_analysis_turns,
            batch_size,
        )
    analysis_turns = [
        (
            select_analysis_turns(
                game_surprises[game_id],
                SEARCH_TEACHER_SELECTION_FRACTION,
                SEARCH_TEACHER_UNIFORM_SELECTION_FRACTION,
                random_generator,
                eligible_analysis_turns[game_id],
            )
            if use_selective_reanalysis
            else (
                eligible_analysis_turns[game_id]
                if eligible_analysis_turns[game_id] is not None
                else list(range(len(game_state_histories[game_id])))
            )
        )
        for game_id in range(game_count)
    ]
    prefix_lengths = [
        len(game_moves[game_id]) - len(game_state_histories[game_id])
        for game_id in range(game_count)
    ]
    absolute_analysis_turns = [
        [
            prefix_lengths[game_id] + turn_number
            for turn_number in analysis_turns[game_id]
        ]
        for game_id in range(game_count)
    ]
    engine = KataGoAnalysisEngine(
        executable_path,
        config_path,
        model_path,
    )
    features: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    q_values: list[np.ndarray] = []
    q_weights: list[np.ndarray] = []
    values: list[float] = []
    game_ids: list[int] = []
    selection_scores: list[float] = []
    child_features: list[np.ndarray] = []
    child_values: list[float] = []
    child_weights: list[float] = []
    child_game_ids: list[int] = []
    child_root_indexes: list[int] = []
    absolute_turn_numbers: list[int] = []
    rollout_moves: list[int] = []
    ownerships: list[np.ndarray] = []
    scores: list[float] = []
    branch_parent_indexes: list[int] = []
    branch_queries: list[dict] = []
    branch_states: dict[int, GameState] = {}
    branch_game_ids: dict[int, int] = {}

    try:
        for query_batch_start in range(
            0,
            game_count,
            SEARCH_TEACHER_QUERY_BATCH_SIZE,
        ):
            query_game_ids = list(
                range(
                    query_batch_start,
                    min(
                        query_batch_start + SEARCH_TEACHER_QUERY_BATCH_SIZE,
                        game_count,
                    ),
                )
            )
            queries = [
                create_analysis_query(
                    game_id,
                    game_moves[game_id],
                    visit_count,
                    absolute_analysis_turns[game_id],
                    include_auxiliary_targets,
                )
                for game_id in query_game_ids
            ]
            expected_response_count = sum(
                len(analysis_turns[game_id])
                for game_id in query_game_ids
            )
            responses = engine.analyze(queries, expected_response_count)

            for response in responses:
                game_id = int(response["id"])
                turn_number = (
                    int(response["turnNumber"])
                    - prefix_lengths[game_id]
                )
                game_state = game_state_histories[game_id][turn_number]
                targets = extract_analysis_targets(
                    response,
                    game_state,
                    include_auxiliary_targets,
                )
                if targets is None:
                    continue

                root_index = len(features)
                child_features.extend(targets.child_features)
                child_values.extend(targets.child_values)
                child_weights.extend(targets.child_weights)
                child_game_ids.extend([game_id] * len(targets.child_features))
                child_root_indexes.extend(
                    [root_index] * len(targets.child_features)
                )
                features.append(encode_moka_features(game_state))
                policies.append(targets.policy)
                q_values.append(targets.q_values)
                q_weights.append(targets.q_weights)
                values.append(targets.value)
                game_ids.append(game_id)
                selection_scores.append(
                    game_surprises[game_id][turn_number]
                )
                absolute_turn_number = int(response["turnNumber"])
                absolute_turn_numbers.append(absolute_turn_number)
                rollout_move = game_moves[game_id][absolute_turn_number]
                rollout_moves.append(rollout_move)
                branch_parent_indexes.append(-1)
                if include_auxiliary_targets:
                    if targets.ownership is None or targets.score is None:
                        raise RuntimeError("Missing requested auxiliary targets.")
                    ownerships.append(targets.ownership)
                    scores.append(targets.score)
                teacher_move = int(np.argmax(targets.policy))
                branch_state = play_move(game_state, teacher_move)
                if (
                    use_teacher_branches
                    and teacher_move != rollout_move
                    and branch_state is not None
                ):
                    branch_queries.append(
                        create_analysis_query(
                            f"branch-{root_index}",
                            game_moves[game_id][:absolute_turn_number]
                            + [teacher_move],
                            visit_count,
                            [absolute_turn_number + 1],
                            include_auxiliary_targets,
                        )
                    )
                    branch_states[root_index] = branch_state
                    branch_game_ids[root_index] = game_id

            print(
                f"analyzed {min(query_batch_start + len(queries), game_count):,}"
                f"/{game_count:,} games positions={len(features):,}"
            )

        for query_batch_start in range(
            0,
            len(branch_queries),
            SEARCH_TEACHER_QUERY_BATCH_SIZE,
        ):
            query_batch = branch_queries[
                query_batch_start : query_batch_start
                + SEARCH_TEACHER_QUERY_BATCH_SIZE
            ]
            responses = engine.analyze(query_batch, len(query_batch))

            for response in responses:
                parent_index = int(response["id"].removeprefix("branch-"))
                game_id = branch_game_ids[parent_index]
                game_state = branch_states[parent_index]
                targets = extract_analysis_targets(
                    response,
                    game_state,
                    include_auxiliary_targets,
                )
                if targets is None:
                    continue

                root_index = len(features)
                child_features.extend(targets.child_features)
                child_values.extend(targets.child_values)
                child_weights.extend(targets.child_weights)
                child_game_ids.extend([game_id] * len(targets.child_features))
                child_root_indexes.extend(
                    [root_index] * len(targets.child_features)
                )
                features.append(encode_moka_features(game_state))
                policies.append(targets.policy)
                q_values.append(targets.q_values)
                q_weights.append(targets.q_weights)
                values.append(targets.value)
                game_ids.append(game_id)
                selection_scores.append(selection_scores[parent_index])
                absolute_turn_numbers.append(-1)
                rollout_moves.append(int(np.argmax(targets.policy)))
                branch_parent_indexes.append(parent_index)
                if include_auxiliary_targets:
                    if targets.ownership is None or targets.score is None:
                        raise RuntimeError("Missing requested auxiliary targets.")
                    ownerships.append(targets.ownership)
                    scores.append(targets.score)

            print(
                "branches "
                f"{min(query_batch_start + len(query_batch), len(branch_queries)):,}"
                f"/{len(branch_queries):,} positions={len(features):,}"
            )

        counterfactual_values = np.full(len(features), np.nan, dtype=np.float32)
        moka_moves = np.full(len(features), BOARD_AREA, dtype=np.int16)

        if use_counterfactual_reanalysis and features:
            model = MokaNestedNetwork()
            model.load_weights(str(checkpoint_path))
            model.eval()
            policy_logit_batches: list[np.ndarray] = []

            for batch_start in range(0, len(features), batch_size):
                policy_logits, _ = model(
                    mx.array(
                        np.asarray(
                            features[batch_start : batch_start + batch_size],
                            dtype=np.float32,
                        )
                    )
                )
                mx.eval(policy_logits)
                policy_logit_batches.append(np.asarray(policy_logits))

            policy_logits = np.concatenate(policy_logit_batches)

            for row_index, (game_id, absolute_turn_number) in enumerate(
                zip(game_ids, absolute_turn_numbers, strict=True)
            ):
                game_state = game_state_histories[game_id][
                    absolute_turn_number - prefix_lengths[game_id]
                ]
                legal_moves = get_legal_moves(game_state)
                selectable_moves = (
                    [move for move in legal_moves if move != BOARD_AREA]
                    if game_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT
                    else legal_moves
                )
                moka_moves[row_index] = int(
                    selectable_moves[
                        int(
                            np.argmax(
                                policy_logits[row_index, selectable_moves]
                            )
                        )
                    ]
                )
            counterfactual_queries = []

            for row_index, (
                game_id,
                absolute_turn_number,
                moka_move,
            ) in enumerate(
                zip(
                    game_ids,
                    absolute_turn_numbers,
                    moka_moves,
                    strict=True,
                )
            ):
                if int(moka_move) == int(np.argmax(policies[row_index])):
                    continue

                game_state = game_state_histories[game_id][
                    absolute_turn_number - prefix_lengths[game_id]
                ]
                next_state = play_move(game_state, int(moka_move))

                if next_state is None:
                    continue

                counterfactual_moves = (
                    game_moves[game_id][:absolute_turn_number]
                    + [int(moka_move)]
                )
                counterfactual_queries.append(
                    create_analysis_query(
                        f"counterfactual-{row_index}",
                        counterfactual_moves,
                        visit_count,
                        [len(counterfactual_moves)],
                    )
                )

            for query_batch_start in range(
                0,
                len(counterfactual_queries),
                SEARCH_TEACHER_QUERY_BATCH_SIZE,
            ):
                query_batch = counterfactual_queries[
                    query_batch_start : (
                        query_batch_start + SEARCH_TEACHER_QUERY_BATCH_SIZE
                    )
                ]
                responses = engine.analyze(query_batch, len(query_batch))

                for response in responses:
                    row_index = int(response["id"].removeprefix("counterfactual-"))
                    child_value = (
                        2 * float(response["rootInfo"]["winrate"]) - 1
                    )
                    counterfactual_values[row_index] = (
                        convert_parent_value_to_child(child_value)
                    )

                print(
                    "counterfactual "
                    f"{min(query_batch_start + len(query_batch), len(counterfactual_queries)):,}"
                    f"/{len(counterfactual_queries):,}"
                )
    finally:
        engine.close()

    dataset = {
        "features": np.asarray(features, dtype=np.float16),
        "game_ids": np.asarray(game_ids, dtype=np.int32),
        "policies": np.asarray(policies, dtype=np.float16),
        "rollout_moves": np.asarray(rollout_moves, dtype=np.int16),
        "selection_scores": np.asarray(
            selection_scores,
            dtype=np.float16,
        ),
        "moka_moves": moka_moves,
        "counterfactual_values": counterfactual_values.astype(np.float16),
        "q_values": np.asarray(q_values, dtype=np.float16),
        "q_weights": np.asarray(q_weights, dtype=np.float16),
        "child_features": np.asarray(child_features, dtype=np.float16),
        "child_game_ids": np.asarray(child_game_ids, dtype=np.int32),
        "child_root_indexes": np.asarray(
            child_root_indexes,
            dtype=np.int32,
        ),
        "child_values": np.asarray(child_values, dtype=np.float16),
        "child_weights": np.asarray(child_weights, dtype=np.float16),
        "values": np.asarray(values, dtype=np.float16),
    }
    if include_auxiliary_targets:
        dataset["ownerships"] = np.asarray(ownerships, dtype=np.float16)
        dataset["scores"] = np.asarray(scores, dtype=np.float16)
    if use_teacher_branches:
        dataset["branch_parent_indexes"] = np.asarray(
            branch_parent_indexes,
            dtype=np.int32,
        )
    return dataset


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument(
        "--opponent",
        type=Path,
        default=Path("../public/models/katago-b6c96.onnx"),
    )
    argument_parser.add_argument(
        "--katago",
        type=Path,
        default=Path("/opt/homebrew/bin/katago"),
    )
    argument_parser.add_argument(
        "--config",
        type=Path,
        default=Path("teachers/katago-source/cpp/configs/analysis_example.cfg"),
    )
    argument_parser.add_argument(
        "--search-model",
        type=Path,
        default=Path(
            "teachers/kata9x9-native/kata9x9-b18c384nbt-20231025.bin"
        ),
    )
    argument_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/moka-search-distillation.npz"),
    )
    argument_parser.add_argument("--selective-reanalysis", action="store_true")
    argument_parser.add_argument("--counterfactual-reanalysis", action="store_true")
    argument_parser.add_argument("--deterministic-rollouts", action="store_true")
    argument_parser.add_argument("--diverse-openings", action="store_true")
    argument_parser.add_argument("--opening-offset", type=int)
    argument_parser.add_argument("--regret-reanalysis", action="store_true")
    argument_parser.add_argument(
        "--blunder-risk-reanalysis",
        action="store_true",
    )
    argument_parser.add_argument(
        "--middle-game-reanalysis",
        action="store_true",
    )
    argument_parser.add_argument("--moka-turns-only", action="store_true")
    argument_parser.add_argument("--auxiliary-targets", action="store_true")
    argument_parser.add_argument("--teacher-branches", action="store_true")
    argument_parser.add_argument(
        "--rollout-simulations",
        type=int,
        default=SEARCH_TEACHER_ROLLOUT_SIMULATION_COUNT,
    )
    argument_parser.add_argument(
        "--games",
        type=int,
        default=SEARCH_TEACHER_DEFAULT_GAME_COUNT,
    )
    argument_parser.add_argument(
        "--visits",
        type=int,
        default=SEARCH_TEACHER_DEFAULT_VISIT_COUNT,
    )
    argument_parser.add_argument(
        "--batch-size",
        type=int,
        default=ON_POLICY_BATCH_SIZE,
    )
    argument_parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    dataset = create_search_dataset(
        arguments.checkpoint,
        arguments.opponent,
        arguments.katago,
        arguments.config,
        arguments.search_model,
        arguments.games,
        arguments.visits,
        arguments.batch_size,
        arguments.seed,
        arguments.selective_reanalysis,
        arguments.counterfactual_reanalysis,
        arguments.deterministic_rollouts,
        (
            arguments.opening_offset
            if arguments.opening_offset is not None
            else DISTILLATION_OPENING_OFFSET
            if arguments.diverse_openings
            else None
        ),
        arguments.regret_reanalysis,
        arguments.middle_game_reanalysis,
        arguments.rollout_simulations,
        arguments.moka_turns_only,
        arguments.auxiliary_targets,
        arguments.teacher_branches,
        arguments.blunder_risk_reanalysis,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, **dataset)
    print(
        f"saved {arguments.output} "
        f"positions={len(dataset['features']):,} "
        f"bytes={arguments.output.stat().st_size:,}"
    )


if __name__ == "__main__":
    main()
