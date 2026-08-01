import argparse
from pathlib import Path

import numpy as np

from go_model.features import encode_moka_features
from go_model.model import create_moka_network_for_checkpoint
from go_model.search import MokaEvaluator, MokaSearchSession
from go_model.search_generate import reconstruct_game_states
from go_model.strong_teacher import StrongKataGoTeacher


class RecordingMokaEvaluator(MokaEvaluator):
    def __init__(self, model) -> None:
        super().__init__(model)
        self.recorded_game_states = {}
        self.recorded_game_ids = {}
        self.recorded_policies = {}
        self.source_game_id = 0

    def evaluate_batch(self, game_states):
        evaluations = super().evaluate_batch(game_states)
        for game_state, evaluation in zip(game_states, evaluations):
            cache_key = self.get_cache_key(game_state)
            if cache_key not in self.recorded_game_states:
                self.recorded_game_states[cache_key] = game_state
                self.recorded_game_ids[cache_key] = self.source_game_id
                self.recorded_policies[cache_key] = evaluation[0]
        return evaluations


def select_stratified_root_indexes(
    game_ids: np.ndarray,
    roots_per_game: int,
) -> list[int]:
    selected_indexes = []
    for game_id in np.unique(game_ids):
        game_indexes = np.flatnonzero(game_ids == game_id)
        quantiles = (
            np.arange(roots_per_game, dtype=np.float64) + 1
        ) / (roots_per_game + 1)
        relative_indexes = np.rint(
            quantiles * (len(game_indexes) - 1)
        ).astype(np.int64)
        selected_indexes.extend(game_indexes[relative_indexes].tolist())
    return selected_indexes


def collect_search_leaf_dataset(
    source_path: Path,
    checkpoint_path: Path,
    teacher_checkpoint_path: Path,
    teacher_source_path: Path,
    output_path: Path,
    roots_per_game: int,
    simulation_count: int,
    teacher_batch_size: int,
) -> None:
    source = np.load(source_path)
    source_game_states = reconstruct_game_states(
        source["root_moves"],
        source["root_move_offsets"],
    )
    selected_indexes = select_stratified_root_indexes(
        source["game_ids"],
        roots_per_game,
    )
    model = create_moka_network_for_checkpoint(str(checkpoint_path))
    model.load_weights(str(checkpoint_path))
    model.eval()
    evaluator = RecordingMokaEvaluator(model)

    for root_number, source_index in enumerate(selected_indexes, start=1):
        root_game_state = source_game_states[source_index]
        evaluator.source_game_id = int(source["game_ids"][source_index])
        search_session = MokaSearchSession(evaluator)
        search_session.select_move(root_game_state, simulation_count)
        if root_number % roots_per_game == 0:
            print(
                f"roots={root_number:,}/{len(selected_indexes):,} "
                f"unique_leaves={len(evaluator.recorded_game_states):,}"
            )

    leaf_game_states = list(evaluator.recorded_game_states.values())
    leaf_keys = list(evaluator.recorded_game_states)
    leaf_game_ids = [
        evaluator.recorded_game_ids[cache_key]
        for cache_key in leaf_keys
    ]

    teacher = StrongKataGoTeacher(
        teacher_checkpoint_path,
        teacher_source_path,
    )
    teacher_values = []
    for batch_start in range(0, len(leaf_game_states), teacher_batch_size):
        batch_game_states = leaf_game_states[
            batch_start : batch_start + teacher_batch_size
        ]
        teacher_values.extend(
            evaluation[1]
            for evaluation in teacher.evaluate_batch(batch_game_states)
        )
        print(
            f"labeled={min(batch_start + teacher_batch_size, len(leaf_game_states)):,}/"
            f"{len(leaf_game_states):,}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=np.asarray(
            [encode_moka_features(game_state) for game_state in leaf_game_states],
            dtype=np.float16,
        ),
        policies=np.asarray(
            [evaluator.recorded_policies[cache_key] for cache_key in leaf_keys],
            dtype=np.float16,
        ),
        values=np.asarray(teacher_values, dtype=np.float32),
        game_ids=np.asarray(leaf_game_ids, dtype=np.int32),
        sample_weights=np.ones(len(leaf_game_states), dtype=np.float32),
    )
    print(f"saved={output_path} positions={len(leaf_game_states):,}")


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--source", required=True, type=Path)
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument(
        "--teacher-checkpoint",
        required=True,
        type=Path,
    )
    argument_parser.add_argument("--teacher-source", required=True, type=Path)
    argument_parser.add_argument("--output", required=True, type=Path)
    argument_parser.add_argument("--roots-per-game", required=True, type=int)
    argument_parser.add_argument("--simulations", required=True, type=int)
    argument_parser.add_argument("--teacher-batch-size", default=512, type=int)
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    collect_search_leaf_dataset(
        arguments.source,
        arguments.checkpoint,
        arguments.teacher_checkpoint,
        arguments.teacher_source,
        arguments.output,
        arguments.roots_per_game,
        arguments.simulations,
        arguments.teacher_batch_size,
    )


if __name__ == "__main__":
    main()
