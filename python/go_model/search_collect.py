import argparse
from pathlib import Path

import numpy as np

from go_model.arena import create_opening_game_state
from go_model.board import is_game_over, play_move
from go_model.collect import select_greedy_rollout_move
from go_model.config import (
    SEARCH_DISTILLATION_DEFAULT_GAME_COUNT,
    SEARCH_DISTILLATION_DEFAULT_SIMULATION_COUNT,
    SEARCH_DISTILLATION_OPENING_OFFSET,
    SEARCH_DISTILLATION_PROGRESS_INTERVAL_GAMES,
    SEARCH_OPPONENT_BRANCH_COUNT,
)
from go_model.features import encode_moka_features
from go_model.model import create_moka_network
from go_model.search import MokaEvaluator, MokaSearchSession
from go_model.teacher import KataGoTeacher


def collect_search_distillation_dataset(
    checkpoint_path: Path,
    teacher_path: Path,
    game_count: int,
    simulation_count: int,
    opening_offset: int,
    opponent_branch_count: int,
    use_teacher_policy_targets: bool,
) -> dict[str, np.ndarray]:
    model = create_moka_network(True, False, False, False, False)
    model.load_weights(str(checkpoint_path))
    model.eval()
    evaluator = MokaEvaluator(model)
    teacher = KataGoTeacher(teacher_path)
    features: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values: list[float] = []
    game_ids: list[int] = []

    for game_id in range(game_count):
        game_state = create_opening_game_state(game_id, opening_offset)
        is_moka_black = game_id % 2 == 0
        search_session = MokaSearchSession(
            evaluator,
            opponent_branch_count=opponent_branch_count,
        )

        while not is_game_over(game_state):
            teacher_policy, teacher_value = teacher.evaluate(game_state)
            is_moka_turn = (game_state.next_color == 1) == is_moka_black

            if is_moka_turn:
                move, target_policy = search_session.select_move_with_policy(
                    game_state,
                    simulation_count,
                )
                if use_teacher_policy_targets:
                    target_policy = teacher_policy
            else:
                move = select_greedy_rollout_move(game_state, teacher_policy)
                target_policy = teacher_policy

            features.append(encode_moka_features(game_state))
            policies.append(target_policy)
            values.append(teacher_value)
            game_ids.append(game_id)
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

    return {
        "features": np.asarray(features, dtype=np.float16),
        "game_ids": np.asarray(game_ids, dtype=np.int32),
        "policies": np.asarray(policies, dtype=np.float16),
        "values": np.asarray(values, dtype=np.float16),
    }


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
