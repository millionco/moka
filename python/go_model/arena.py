import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from go_model.board import (
    GameState,
    get_area_score,
    get_legal_moves,
    is_game_over,
    play_move,
)
from go_model.config import (
    ARENA_DEFAULT_GAME_COUNT,
    ARENA_OPENING_INDEX_MULTIPLIER,
    ARENA_OPENING_MOVE_COUNT,
    ARENA_OPENING_MOVE_MULTIPLIER,
    ARENA_OPENING_PAIR_SIZE,
    BOARD_AREA,
    MAXIMUM_GAME_MOVE_COUNT,
    MINIMUM_TEACHER_PASS_MOVE_COUNT,
    SEARCH_AREA_VALUE_WEIGHT,
    SEARCH_ADAPTIVE_MAX_SIMULATION_COUNT,
    SEARCH_ADAPTIVE_VISIT_MARGIN_RATIO,
    SEARCH_FIRST_PLAY_URGENCY_REDUCTION,
    SEARCH_FIRST_PLAY_URGENCY_ROOT_ONLY,
    SEARCH_FIRST_PLAY_URGENCY_USE_PRIOR_MASS,
    SEARCH_LATE_SIMULATION_COUNT,
    SEARCH_LATE_SIMULATION_START_MOVE_COUNT,
    SEARCH_OPPONENT_BRANCH_COUNT,
    SEARCH_PUCT_EXPLORATION,
    SEARCH_PUCT_VALUE_WEIGHT,
    SEARCH_ROLLOUT_DEPTH,
    SEARCH_ROOT_BRANCH_COUNT,
    SEARCH_ROOT_POLICY_TEMPERATURE,
    SEARCH_SEQUENTIAL_HALVING_CANDIDATE_COUNT,
)
from go_model.features import (
    encode_moka_context_features,
    encode_moka_features,
)
from go_model.model import MokaNetwork, create_moka_network
from go_model.search import (
    MokaEvaluator,
    MokaSearchSession,
    MokaSequentialHalvingSearchSession,
    select_policy_value_move,
    select_rollout_move,
    select_search_move,
)
from go_model.teacher import KataGoTeacher


def create_opening_game_state(game_index: int, opening_offset: int) -> GameState:
    opening_index = opening_offset + game_index // ARENA_OPENING_PAIR_SIZE
    game_state = GameState()

    for opening_move_index in range(ARENA_OPENING_MOVE_COUNT):
        legal_moves = [
            move for move in get_legal_moves(game_state) if move < BOARD_AREA
        ]
        selected_move_index = (
            opening_index * ARENA_OPENING_INDEX_MULTIPLIER
            + opening_move_index * ARENA_OPENING_MOVE_MULTIPLIER
        ) % len(legal_moves)
        next_state = play_move(game_state, legal_moves[selected_move_index])

        if next_state is None:
            break

        game_state = next_state

    return game_state


def get_search_simulation_count(
    game_state: GameState,
    simulation_count: int,
    late_simulation_count: int,
    late_simulation_start_move_count: int,
) -> int:
    if (
        late_simulation_count > 0
        and game_state.move_count >= late_simulation_start_move_count
    ):
        return late_simulation_count

    return simulation_count


def select_moka_move(
    model: MokaNetwork,
    evaluator: MokaEvaluator,
    game_state: GameState,
    simulation_count: int,
    lookahead_candidate_count: int,
    use_context_features: bool,
    rollout_candidate_count: int,
    rollout_count: int,
    random_seed: int,
    search_session: MokaSearchSession | None,
) -> int:
    if (
        game_state.move_count >= MINIMUM_TEACHER_PASS_MOVE_COUNT
        and game_state.consecutive_pass_count == 1
    ):
        pass_state = play_move(game_state, BOARD_AREA)

        if pass_state is not None:
            did_black_win = get_area_score(pass_state) > 0
            did_current_player_win = did_black_win == (
                game_state.next_color == 1
            )

            if did_current_player_win:
                return BOARD_AREA

    if rollout_count > 0:
        return select_rollout_move(
            evaluator,
            game_state,
            rollout_candidate_count,
            rollout_count,
            random_seed,
        )

    if lookahead_candidate_count > 0:
        return select_policy_value_move(
            evaluator,
            game_state,
            lookahead_candidate_count,
        )

    if simulation_count > 0:
        if search_session is not None:
            return search_session.select_move(game_state, simulation_count)
        return select_search_move(evaluator, game_state, simulation_count)

    features = (
        encode_moka_context_features(game_state)
        if use_context_features
        else encode_moka_features(game_state)
    )
    policy_logits, _ = model(mx.array(features[None], dtype=mx.float32))
    mx.eval(policy_logits)
    logits = np.asarray(policy_logits)[0]
    legal_moves = get_legal_moves(game_state)
    selectable_moves = (
        [move for move in legal_moves if move != BOARD_AREA]
        if game_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT
        else legal_moves
    )
    return int(selectable_moves[int(np.argmax(logits[selectable_moves]))])


def select_teacher_move(teacher: KataGoTeacher, game_state: GameState) -> int:
    policy, _ = teacher.evaluate(game_state)
    legal_moves = get_legal_moves(game_state)
    selectable_moves = (
        [move for move in legal_moves if move != BOARD_AREA]
        if game_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT
        else legal_moves
    )
    return int(selectable_moves[int(np.argmax(policy[selectable_moves]))])


def run_arena(
    checkpoint_path: Path,
    teacher_path: Path,
    game_count: int,
    simulation_count: int,
    lookahead_candidate_count: int,
    opening_offset: int,
    use_nested_network: bool,
    use_spatial_network: bool,
    use_recurrent_network: bool,
    use_context_network: bool,
    use_wide_network: bool,
    rollout_candidate_count: int,
    rollout_count: int,
    search_exploration: float = SEARCH_PUCT_EXPLORATION,
    search_value_weight: float = SEARCH_PUCT_VALUE_WEIGHT,
    search_area_value_weight: float = SEARCH_AREA_VALUE_WEIGHT,
    search_rollout_depth: int = SEARCH_ROLLOUT_DEPTH,
    sequential_halving_candidate_count: int = (
        SEARCH_SEQUENTIAL_HALVING_CANDIDATE_COUNT
    ),
    use_symmetry_ensemble: bool = False,
    symmetry_rotation_count: int = 0,
    should_flip_symmetry: bool = False,
    adaptive_max_simulation_count: int = (
        SEARCH_ADAPTIVE_MAX_SIMULATION_COUNT
    ),
    adaptive_visit_margin_ratio: float = SEARCH_ADAPTIVE_VISIT_MARGIN_RATIO,
    use_global_pool_network: bool = False,
    opponent_branch_count: int = SEARCH_OPPONENT_BRANCH_COUNT,
    use_root_symmetry_ensemble: bool = False,
    use_descendant_symmetry_pair: bool = False,
    root_selection_visit_slack: int = -1,
    root_capture_prior_bonus: float = 0,
    root_self_atari_prior_penalty: float = 0,
    search_first_play_urgency_reduction: float = (
        SEARCH_FIRST_PLAY_URGENCY_REDUCTION
    ),
    use_first_play_urgency_prior_mass: bool = (
        SEARCH_FIRST_PLAY_URGENCY_USE_PRIOR_MASS
    ),
    use_first_play_urgency_at_root_only: bool = (
        SEARCH_FIRST_PLAY_URGENCY_ROOT_ONLY
    ),
    root_branch_count: int = SEARCH_ROOT_BRANCH_COUNT,
    root_policy_temperature: float = SEARCH_ROOT_POLICY_TEMPERATURE,
    late_simulation_count: int = SEARCH_LATE_SIMULATION_COUNT,
    late_simulation_start_move_count: int = (
        SEARCH_LATE_SIMULATION_START_MOVE_COUNT
    ),
) -> tuple[int, int, int]:
    model = create_moka_network(
        use_nested_network,
        use_spatial_network,
        use_recurrent_network,
        use_context_network,
        use_wide_network,
        use_global_pool_network,
    )
    model.load_weights(
        str(checkpoint_path),
        strict=not use_spatial_network,
    )
    model.eval()
    teacher = KataGoTeacher(teacher_path)
    evaluator = MokaEvaluator(
        model,
        use_symmetry_ensemble,
        symmetry_rotation_count,
        should_flip_symmetry,
        use_descendant_symmetry_pair,
    )
    root_evaluator = (
        MokaEvaluator(model, use_symmetry_ensemble=True)
        if use_root_symmetry_ensemble
        else None
    )
    moka_win_count = 0
    moka_black_win_count = 0
    moka_white_win_count = 0
    moka_move_cap_win_count = 0
    kata_go_win_count = 0
    move_cap_count = 0
    moka_pass_count = 0
    teacher_pass_count = 0
    capped_repeated_position_count = 0
    capped_unique_position_count = 0
    start_time = time.perf_counter()

    for game_index in range(game_count):
        game_state = create_opening_game_state(game_index, opening_offset)
        is_moka_black = game_index % ARENA_OPENING_PAIR_SIZE == 0
        search_session = (
            (
                MokaSequentialHalvingSearchSession(
                    evaluator=evaluator,
                    candidate_count=sequential_halving_candidate_count,
                    exploration=search_exploration,
                    value_weight=search_value_weight,
                    area_value_weight=search_area_value_weight,
                    rollout_depth=search_rollout_depth,
                    adaptive_max_simulation_count=(
                        adaptive_max_simulation_count
                    ),
                    adaptive_visit_margin_ratio=(
                        adaptive_visit_margin_ratio
                    ),
                    opponent_branch_count=opponent_branch_count,
                    root_evaluator=root_evaluator,
                    root_selection_visit_slack=(
                        root_selection_visit_slack
                    ),
                    root_capture_prior_bonus=root_capture_prior_bonus,
                    root_self_atari_prior_penalty=(
                        root_self_atari_prior_penalty
                    ),
                    first_play_urgency_reduction=(
                        search_first_play_urgency_reduction
                    ),
                    use_first_play_urgency_prior_mass=(
                        use_first_play_urgency_prior_mass
                    ),
                    use_first_play_urgency_at_root_only=(
                        use_first_play_urgency_at_root_only
                    ),
                    root_branch_count=root_branch_count,
                    root_policy_temperature=root_policy_temperature,
                )
                if sequential_halving_candidate_count > 0
                else MokaSearchSession(
                    evaluator=evaluator,
                    exploration=search_exploration,
                    value_weight=search_value_weight,
                    area_value_weight=search_area_value_weight,
                    rollout_depth=search_rollout_depth,
                    adaptive_max_simulation_count=(
                        adaptive_max_simulation_count
                    ),
                    adaptive_visit_margin_ratio=(
                        adaptive_visit_margin_ratio
                    ),
                    opponent_branch_count=opponent_branch_count,
                    root_evaluator=root_evaluator,
                    root_selection_visit_slack=(
                        root_selection_visit_slack
                    ),
                    root_capture_prior_bonus=root_capture_prior_bonus,
                    root_self_atari_prior_penalty=(
                        root_self_atari_prior_penalty
                    ),
                    first_play_urgency_reduction=(
                        search_first_play_urgency_reduction
                    ),
                    use_first_play_urgency_prior_mass=(
                        use_first_play_urgency_prior_mass
                    ),
                    use_first_play_urgency_at_root_only=(
                        use_first_play_urgency_at_root_only
                    ),
                    root_branch_count=root_branch_count,
                    root_policy_temperature=root_policy_temperature,
                )
            )
            if simulation_count > 0
            else None
        )
        seen_position_keys: set[tuple[bytes, int]] = set()
        repeated_position_count = 0

        while not is_game_over(game_state):
            position_key = (
                game_state.board.tobytes(),
                game_state.next_color,
            )
            repeated_position_count += int(
                position_key in seen_position_keys
            )
            seen_position_keys.add(position_key)
            is_moka_turn = (game_state.next_color == 1) == is_moka_black
            turn_simulation_count = get_search_simulation_count(
                game_state,
                simulation_count,
                late_simulation_count,
                late_simulation_start_move_count,
            )
            move = (
                select_moka_move(
                    model,
                    evaluator,
                    game_state,
                    turn_simulation_count,
                    lookahead_candidate_count,
                    use_context_network,
                    rollout_candidate_count,
                    rollout_count,
                    (game_index + 1) * ARENA_OPENING_MOVE_MULTIPLIER
                    + game_state.move_count,
                    search_session,
                )
                if is_moka_turn
                else select_teacher_move(teacher, game_state)
            )
            moka_pass_count += int(is_moka_turn and move == BOARD_AREA)
            teacher_pass_count += int(not is_moka_turn and move == BOARD_AREA)
            next_state = play_move(game_state, move)

            if next_state is None:
                raise RuntimeError("Arena engine selected an illegal move.")

            game_state = next_state

        did_black_win = get_area_score(game_state) > 0
        did_moka_win = did_black_win == is_moka_black
        moka_win_count += int(did_moka_win)
        moka_black_win_count += int(did_moka_win and is_moka_black)
        moka_white_win_count += int(did_moka_win and not is_moka_black)
        kata_go_win_count += int(not did_moka_win)
        did_reach_move_cap = game_state.move_count >= MAXIMUM_GAME_MOVE_COUNT
        move_cap_count += int(did_reach_move_cap)
        moka_move_cap_win_count += int(did_moka_win and did_reach_move_cap)

        if did_reach_move_cap:
            capped_repeated_position_count += repeated_position_count
            capped_unique_position_count += len(seen_position_keys)

    duration_seconds = time.perf_counter() - start_time
    print(
        f"Moka={moka_win_count} "
        f"black={moka_black_win_count} "
        f"white={moka_white_win_count} "
        f"KataGo={kata_go_win_count} "
        f"caps={move_cap_count} "
        f"MokaCapWins={moka_move_cap_win_count} "
        f"MokaPasses={moka_pass_count} "
        f"KataGoPasses={teacher_pass_count} "
        f"CapRepeats={capped_repeated_position_count} "
        f"CapUnique={capped_unique_position_count} "
        f"seconds={duration_seconds:.1f}"
    )
    return moka_win_count, kata_go_win_count, move_cap_count


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument(
        "--teacher",
        type=Path,
        default=Path("../public/models/katago-b6c96.onnx"),
    )
    argument_parser.add_argument(
        "--games",
        type=int,
        default=ARENA_DEFAULT_GAME_COUNT,
    )
    argument_parser.add_argument("--simulations", type=int, default=0)
    argument_parser.add_argument(
        "--late-simulations",
        type=int,
        default=SEARCH_LATE_SIMULATION_COUNT,
    )
    argument_parser.add_argument(
        "--late-simulation-start-move",
        type=int,
        default=SEARCH_LATE_SIMULATION_START_MOVE_COUNT,
    )
    argument_parser.add_argument(
        "--search-exploration",
        type=float,
        default=SEARCH_PUCT_EXPLORATION,
    )
    argument_parser.add_argument(
        "--search-value-weight",
        type=float,
        default=SEARCH_PUCT_VALUE_WEIGHT,
    )
    argument_parser.add_argument(
        "--search-fpu-reduction",
        type=float,
        default=SEARCH_FIRST_PLAY_URGENCY_REDUCTION,
    )
    argument_parser.add_argument(
        "--search-fpu-prior-mass",
        action="store_true",
    )
    argument_parser.add_argument(
        "--search-fpu-root-only",
        action="store_true",
    )
    argument_parser.add_argument(
        "--search-area-value-weight",
        type=float,
        default=SEARCH_AREA_VALUE_WEIGHT,
    )
    argument_parser.add_argument(
        "--search-rollout-depth",
        type=int,
        default=SEARCH_ROLLOUT_DEPTH,
    )
    argument_parser.add_argument(
        "--sequential-halving-candidates",
        type=int,
        default=SEARCH_SEQUENTIAL_HALVING_CANDIDATE_COUNT,
    )
    argument_parser.add_argument("--symmetry-ensemble", action="store_true")
    argument_parser.add_argument("--symmetry-rotation", type=int, default=0)
    argument_parser.add_argument("--symmetry-flip", action="store_true")
    argument_parser.add_argument(
        "--adaptive-max-simulations",
        type=int,
        default=SEARCH_ADAPTIVE_MAX_SIMULATION_COUNT,
    )
    argument_parser.add_argument(
        "--adaptive-visit-margin",
        type=float,
        default=SEARCH_ADAPTIVE_VISIT_MARGIN_RATIO,
    )
    argument_parser.add_argument(
        "--opponent-branches",
        type=int,
        default=SEARCH_OPPONENT_BRANCH_COUNT,
    )
    argument_parser.add_argument(
        "--root-symmetry-ensemble",
        action="store_true",
    )
    argument_parser.add_argument(
        "--root-branches",
        type=int,
        default=SEARCH_ROOT_BRANCH_COUNT,
    )
    argument_parser.add_argument(
        "--root-policy-temperature",
        type=float,
        default=SEARCH_ROOT_POLICY_TEMPERATURE,
    )
    argument_parser.add_argument(
        "--descendant-symmetry-pair",
        action="store_true",
    )
    argument_parser.add_argument(
        "--root-selection-visit-slack",
        type=int,
        default=-1,
    )
    argument_parser.add_argument(
        "--root-capture-prior-bonus",
        type=float,
        default=0,
    )
    argument_parser.add_argument(
        "--root-self-atari-prior-penalty",
        type=float,
        default=0,
    )
    argument_parser.add_argument("--lookahead-candidates", type=int, default=0)
    argument_parser.add_argument("--rollout-candidates", type=int, default=4)
    argument_parser.add_argument("--rollouts", type=int, default=0)
    argument_parser.add_argument("--opening-offset", type=int, default=0)
    argument_parser.add_argument("--nested", action="store_true")
    argument_parser.add_argument("--spatial", action="store_true")
    argument_parser.add_argument("--recurrent", action="store_true")
    argument_parser.add_argument("--context", action="store_true")
    argument_parser.add_argument("--wide", action="store_true")
    argument_parser.add_argument("--global-pool", action="store_true")
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    run_arena(
        arguments.checkpoint,
        arguments.teacher,
        arguments.games,
        arguments.simulations,
        arguments.lookahead_candidates,
        arguments.opening_offset,
        arguments.nested,
        arguments.spatial,
        arguments.recurrent,
        arguments.context,
        arguments.wide,
        arguments.rollout_candidates,
        arguments.rollouts,
        arguments.search_exploration,
        arguments.search_value_weight,
        arguments.search_area_value_weight,
        arguments.search_rollout_depth,
        arguments.sequential_halving_candidates,
        arguments.symmetry_ensemble,
        arguments.symmetry_rotation,
        arguments.symmetry_flip,
        arguments.adaptive_max_simulations,
        arguments.adaptive_visit_margin,
        arguments.global_pool,
        arguments.opponent_branches,
        arguments.root_symmetry_ensemble,
        arguments.descendant_symmetry_pair,
        arguments.root_selection_visit_slack,
        arguments.root_capture_prior_bonus,
        arguments.root_self_atari_prior_penalty,
        arguments.search_fpu_reduction,
        arguments.search_fpu_prior_mass,
        arguments.search_fpu_root_only,
        arguments.root_branches,
        arguments.root_policy_temperature,
        arguments.late_simulations,
        arguments.late_simulation_start_move,
    )


if __name__ == "__main__":
    main()
