import argparse
from pathlib import Path

import numpy as np

from go_model.board import GameState
from go_model.config import (
    DEFAULT_RANDOM_SEED,
    DISTILLATION_OPENING_OFFSET,
    ON_POLICY_BATCH_SIZE,
    SEARCH_TEACHER_DEFAULT_GAME_COUNT,
    SEARCH_TEACHER_ROLLOUT_SIMULATION_COUNT,
)
from go_model.features import encode_moka_features
from go_model.search_generate import (
    evaluate_optimistic_policies,
    flatten_move_histories,
    generate_rollout_games,
    is_moka_turn,
)


def select_moka_game_states(
    game_state_histories: list[list[GameState]],
) -> tuple[list[GameState], np.ndarray]:
    game_states: list[GameState] = []
    game_ids: list[int] = []

    for game_id, game_state_history in enumerate(game_state_histories):
        for game_state in game_state_history:
            if is_moka_turn(game_id, game_state):
                game_states.append(game_state)
                game_ids.append(game_id)

    return game_states, np.asarray(game_ids, dtype=np.int32)


def create_optimistic_policy_dataset(
    checkpoint_path: Path,
    opponent_path: Path,
    teacher_checkpoint_path: Path,
    teacher_source_path: Path,
    game_count: int,
    batch_size: int,
    random_seed: int,
    opening_offset: int,
    rollout_simulation_count: int,
) -> dict[str, np.ndarray]:
    game_state_histories, _, _ = generate_rollout_games(
        checkpoint_path,
        opponent_path,
        game_count,
        batch_size,
        random_seed,
        False,
        opening_offset,
        False,
        rollout_simulation_count,
    )
    game_states, game_ids = select_moka_game_states(game_state_histories)
    optimistic_policies = evaluate_optimistic_policies(
        game_states,
        teacher_checkpoint_path,
        teacher_source_path,
    )
    root_moves, root_move_offsets = flatten_move_histories(game_states)
    return {
        "features": np.asarray(
            [encode_moka_features(game_state) for game_state in game_states],
            dtype=np.float16,
        ),
        "game_ids": game_ids,
        "optimistic_policies": optimistic_policies.astype(np.float16),
        "root_moves": root_moves,
        "root_move_offsets": root_move_offsets,
    }


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument("--opponent", required=True, type=Path)
    argument_parser.add_argument(
        "--optimistic-policy-checkpoint",
        required=True,
        type=Path,
    )
    argument_parser.add_argument(
        "--katago-source",
        required=True,
        type=Path,
    )
    argument_parser.add_argument("--output", required=True, type=Path)
    argument_parser.add_argument(
        "--games",
        type=int,
        default=SEARCH_TEACHER_DEFAULT_GAME_COUNT,
    )
    argument_parser.add_argument(
        "--batch-size",
        type=int,
        default=ON_POLICY_BATCH_SIZE,
    )
    argument_parser.add_argument(
        "--rollout-simulations",
        type=int,
        default=SEARCH_TEACHER_ROLLOUT_SIMULATION_COUNT,
    )
    argument_parser.add_argument(
        "--opening-offset",
        type=int,
        default=DISTILLATION_OPENING_OFFSET,
    )
    argument_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    dataset = create_optimistic_policy_dataset(
        arguments.checkpoint,
        arguments.opponent,
        arguments.optimistic_policy_checkpoint,
        arguments.katago_source,
        arguments.games,
        arguments.batch_size,
        arguments.seed,
        arguments.opening_offset,
        arguments.rollout_simulations,
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
