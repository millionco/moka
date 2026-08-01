import argparse
from pathlib import Path

import numpy as np

from go_model.search_generate import reconstruct_game_states
from go_model.strong_teacher import StrongKataGoTeacher


def collect_strong_value_dataset(
    source_path: Path,
    teacher_checkpoint_path: Path,
    teacher_source_path: Path,
    output_path: Path,
    teacher_batch_size: int,
    sample_weight: float,
) -> None:
    if sample_weight <= 0:
        raise ValueError("Sample weight must be positive.")
    source = np.load(source_path)
    game_states = reconstruct_game_states(
        source["root_moves"],
        source["root_move_offsets"],
    )
    teacher = StrongKataGoTeacher(
        teacher_checkpoint_path,
        teacher_source_path,
    )
    policies = []
    values = []

    for batch_start in range(0, len(game_states), teacher_batch_size):
        evaluations = teacher.evaluate_batch(
            game_states[batch_start : batch_start + teacher_batch_size]
        )
        policies.extend(evaluation[0] for evaluation in evaluations)
        values.extend(evaluation[1] for evaluation in evaluations)
        print(
            f"labeled={min(batch_start + teacher_batch_size, len(game_states)):,}/"
            f"{len(game_states):,}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=source["features"],
        policies=np.asarray(policies, dtype=np.float16),
        values=np.asarray(values, dtype=np.float32),
        game_ids=source["game_ids"],
        sample_weights=np.full(
            len(game_states),
            sample_weight,
            dtype=np.float32,
        ),
    )
    print(f"saved={output_path} positions={len(game_states):,}")


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--source", required=True, type=Path)
    argument_parser.add_argument(
        "--teacher-checkpoint",
        required=True,
        type=Path,
    )
    argument_parser.add_argument("--teacher-source", required=True, type=Path)
    argument_parser.add_argument("--output", required=True, type=Path)
    argument_parser.add_argument("--teacher-batch-size", default=512, type=int)
    argument_parser.add_argument("--sample-weight", default=1, type=float)
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    collect_strong_value_dataset(
        arguments.source,
        arguments.teacher_checkpoint,
        arguments.teacher_source,
        arguments.output,
        arguments.teacher_batch_size,
        arguments.sample_weight,
    )


if __name__ == "__main__":
    main()
