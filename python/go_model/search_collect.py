import argparse
from pathlib import Path

import numpy as np

from go_model.arena import (
    create_opening_game_state,
    should_resign_selected_pass,
)
from go_model.board import get_legal_moves, is_game_over, play_move
from go_model.collect import select_greedy_rollout_move
from go_model.config import (
    SEARCH_DISTILLATION_DEFAULT_GAME_COUNT,
    SEARCH_DISTILLATION_DEFAULT_SIMULATION_COUNT,
    SEARCH_DISTILLATION_OPENING_OFFSET,
    SEARCH_DISTILLATION_PROGRESS_INTERVAL_GAMES,
    SEARCH_FIRST_PLAY_URGENCY_REDUCTION,
    SEARCH_OPPONENT_BRANCH_COUNT,
    SEARCH_PUCT_EXPLORATION,
    SEARCH_Q_POLICY_BLEND,
    SEARCH_Q_POLICY_TEMPERATURE,
    SEARCH_RESIGNATION_AREA_MARGIN_POINTS,
)
from go_model.features import encode_moka_features
from go_model.model import create_moka_network
from go_model.search import MokaEvaluator, MokaSearchSession, SearchNode
from go_model.teacher import KataGoTeacher


def create_search_q_policy(
    q_values: np.ndarray,
    q_weights: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Search Q policy temperature must be positive.")

    visited_mask = q_weights > 0
    q_policy = np.zeros_like(q_values)

    if not np.any(visited_mask):
        return q_policy

    visited_q_values = q_values[visited_mask]
    maximum_q_value = float(np.max(visited_q_values))
    visited_probabilities = np.exp(
        (visited_q_values - maximum_q_value) / temperature
    )
    visited_probabilities /= np.sum(visited_probabilities)
    q_policy[visited_mask] = visited_probabilities
    return q_policy


def get_visited_child_value_targets(
    root: SearchNode,
) -> list[tuple[np.ndarray, float, float]]:
    return [
        (
            encode_moka_features(child.game_state),
            child.mean_value,
            float(child.visit_count),
        )
        for child in root.children.values()
        if child.visit_count > 0
    ]


def collect_search_distillation_dataset(
    checkpoint_path: Path,
    teacher_path: Path,
    game_count: int,
    simulation_count: int,
    opening_offset: int,
    opponent_branch_count: int,
    use_teacher_policy_targets: bool,
    use_root_symmetry_ensemble: bool,
    search_policy_blend: float,
    search_sample_weight: float,
    use_moka_turns_only: bool,
    search_q_policy_blend: float = SEARCH_Q_POLICY_BLEND,
    search_q_policy_temperature: float = SEARCH_Q_POLICY_TEMPERATURE,
    search_exploration: float = SEARCH_PUCT_EXPLORATION,
    search_first_play_urgency_reduction: float = (
        SEARCH_FIRST_PLAY_URGENCY_REDUCTION
    ),
    resignation_area_margin_points: float = (
        SEARCH_RESIGNATION_AREA_MARGIN_POINTS
    ),
    should_collect_child_value_targets: bool = False,
) -> dict[str, np.ndarray]:
    if not 0 <= search_policy_blend <= 1:
        raise ValueError("Search policy blend must be between zero and one.")
    if search_sample_weight <= 0:
        raise ValueError("Search sample weight must be positive.")
    if not 0 <= search_q_policy_blend <= 1:
        raise ValueError("Search Q policy blend must be between zero and one.")
    if search_q_policy_temperature <= 0:
        raise ValueError("Search Q policy temperature must be positive.")

    model = create_moka_network(True, False, False, False, False)
    model.load_weights(str(checkpoint_path))
    model.eval()
    evaluator = MokaEvaluator(model)
    teacher = KataGoTeacher(teacher_path)
    features: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values: list[float] = []
    game_ids: list[int] = []
    sample_weights: list[float] = []
    search_q_values: list[np.ndarray] = []
    search_q_weights: list[np.ndarray] = []
    child_features: list[np.ndarray] = []
    child_values: list[float] = []
    child_weights: list[float] = []
    child_game_ids: list[int] = []
    child_root_indexes: list[int] = []
    root_evaluator = (
        MokaEvaluator(model, use_symmetry_ensemble=True)
        if use_root_symmetry_ensemble
        else None
    )

    for game_id in range(game_count):
        game_state = create_opening_game_state(game_id, opening_offset)
        is_moka_black = game_id % 2 == 0
        search_session = MokaSearchSession(
            evaluator,
            exploration=search_exploration,
            opponent_branch_count=opponent_branch_count,
            root_evaluator=root_evaluator,
            first_play_urgency_reduction=(
                search_first_play_urgency_reduction
            ),
        )

        while not is_game_over(game_state):
            teacher_policy, teacher_value = teacher.evaluate(game_state)
            is_moka_turn = (game_state.next_color == 1) == is_moka_black

            if is_moka_turn:
                searched_root = search_session.align_root(game_state)
                root_policy = (
                    root_evaluator.evaluate(game_state)[0]
                    if root_evaluator is not None
                    else evaluator.evaluate(game_state)[0]
                )
                (
                    move,
                    target_policy,
                    root_q_values,
                    root_q_weights,
                ) = search_session.select_move_with_search_targets(
                    game_state,
                    simulation_count,
                )
                if use_teacher_policy_targets:
                    target_policy = teacher_policy
                else:
                    if search_q_policy_blend > 0:
                        q_policy = create_search_q_policy(
                            root_q_values,
                            root_q_weights,
                            search_q_policy_temperature,
                        )
                        target_policy = (
                            search_q_policy_blend * q_policy
                            + (1 - search_q_policy_blend) * target_policy
                        )
                if (
                    not use_teacher_policy_targets
                    and search_policy_blend < 1
                ):
                    legal_moves = get_legal_moves(game_state)
                    legal_root_policy = np.zeros_like(root_policy)
                    legal_root_policy[legal_moves] = root_policy[legal_moves]
                    legal_root_policy /= np.sum(legal_root_policy)
                    target_policy = (
                        search_policy_blend * target_policy
                        + (1 - search_policy_blend) * legal_root_policy
                    )
                if should_collect_child_value_targets:
                    root_index = len(features)

                    for (
                        child_feature,
                        child_value,
                        child_weight,
                    ) in get_visited_child_value_targets(searched_root):
                        child_features.append(child_feature)
                        child_values.append(child_value)
                        child_weights.append(child_weight)
                        child_game_ids.append(game_id)
                        child_root_indexes.append(root_index)
            else:
                move = select_greedy_rollout_move(game_state, teacher_policy)
                target_policy = teacher_policy
                root_q_values = np.zeros_like(teacher_policy)
                root_q_weights = np.zeros_like(teacher_policy)

            if is_moka_turn or not use_moka_turns_only:
                features.append(encode_moka_features(game_state))
                policies.append(target_policy)
                values.append(teacher_value)
                game_ids.append(game_id)
                sample_weights.append(
                    search_sample_weight if is_moka_turn else 1
                )
                search_q_values.append(root_q_values)
                search_q_weights.append(root_q_weights)

            if is_moka_turn and should_resign_selected_pass(
                game_state,
                move,
                resignation_area_margin_points,
            ):
                break

            next_state = play_move(game_state, move)

            if next_state is None:
                raise RuntimeError("Search distillation selected an illegal move.")

            game_state = next_state

        completed_game_count = game_id + 1

        if (
            completed_game_count % SEARCH_DISTILLATION_PROGRESS_INTERVAL_GAMES == 0
            or completed_game_count == game_count
        ):
            print(
                f"completed {completed_game_count:,}/{game_count:,} games "
                f"positions={len(features):,}"
            )

    dataset = {
        "features": np.asarray(features, dtype=np.float16),
        "game_ids": np.asarray(game_ids, dtype=np.int32),
        "policies": np.asarray(policies, dtype=np.float16),
        "sample_weights": np.asarray(sample_weights, dtype=np.float16),
        "search_q_values": np.asarray(search_q_values, dtype=np.float16),
        "search_q_weights": np.asarray(search_q_weights, dtype=np.float16),
        "values": np.asarray(values, dtype=np.float16),
    }
    if should_collect_child_value_targets:
        dataset.update(
            {
                "child_features": np.asarray(
                    child_features,
                    dtype=np.float16,
                ),
                "child_game_ids": np.asarray(
                    child_game_ids,
                    dtype=np.int32,
                ),
                "child_root_indexes": np.asarray(
                    child_root_indexes,
                    dtype=np.int32,
                ),
                "child_values": np.asarray(
                    child_values,
                    dtype=np.float16,
                ),
                "child_weights": np.asarray(
                    child_weights,
                    dtype=np.float16,
                ),
            }
        )
    return dataset


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument(
        "--teacher",
        type=Path,
        default=Path("../public/models/katago-b6c96.onnx"),
    )
    argument_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/moka-search-distillation.npz"),
    )
    argument_parser.add_argument(
        "--games",
        type=int,
        default=SEARCH_DISTILLATION_DEFAULT_GAME_COUNT,
    )
    argument_parser.add_argument(
        "--simulations",
        type=int,
        default=SEARCH_DISTILLATION_DEFAULT_SIMULATION_COUNT,
    )
    argument_parser.add_argument(
        "--opening-offset",
        type=int,
        default=SEARCH_DISTILLATION_OPENING_OFFSET,
    )
    argument_parser.add_argument(
        "--opponent-branches",
        type=int,
        default=SEARCH_OPPONENT_BRANCH_COUNT,
    )
    argument_parser.add_argument(
        "--teacher-policy-targets",
        action="store_true",
    )
    argument_parser.add_argument(
        "--root-symmetry-ensemble",
        action="store_true",
    )
    argument_parser.add_argument(
        "--search-policy-blend",
        type=float,
        default=1,
    )
    argument_parser.add_argument(
        "--search-sample-weight",
        type=float,
        default=1,
    )
    argument_parser.add_argument(
        "--moka-turns-only",
        action="store_true",
    )
    argument_parser.add_argument(
        "--search-q-policy-blend",
        type=float,
        default=SEARCH_Q_POLICY_BLEND,
    )
    argument_parser.add_argument(
        "--search-q-policy-temperature",
        type=float,
        default=SEARCH_Q_POLICY_TEMPERATURE,
    )
    argument_parser.add_argument(
        "--search-exploration",
        type=float,
        default=SEARCH_PUCT_EXPLORATION,
    )
    argument_parser.add_argument(
        "--search-fpu-reduction",
        type=float,
        default=SEARCH_FIRST_PLAY_URGENCY_REDUCTION,
    )
    argument_parser.add_argument(
        "--resignation-area-margin",
        type=float,
        default=SEARCH_RESIGNATION_AREA_MARGIN_POINTS,
    )
    argument_parser.add_argument(
        "--child-value-targets",
        action="store_true",
    )
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    dataset = collect_search_distillation_dataset(
        arguments.checkpoint,
        arguments.teacher,
        arguments.games,
        arguments.simulations,
        arguments.opening_offset,
        arguments.opponent_branches,
        arguments.teacher_policy_targets,
        arguments.root_symmetry_ensemble,
        arguments.search_policy_blend,
        arguments.search_sample_weight,
        arguments.moka_turns_only,
        arguments.search_q_policy_blend,
        arguments.search_q_policy_temperature,
        arguments.search_exploration,
        arguments.search_fpu_reduction,
        arguments.resignation_area_margin,
        arguments.child_value_targets,
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
