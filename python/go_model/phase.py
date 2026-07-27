import argparse
from pathlib import Path

import numpy as np

from go_model.config import (
    MID_GAME_MOVE_COUNT,
    MID_GAME_SAMPLE_WEIGHT,
    OPENING_MOVE_COUNT,
    OPENING_SAMPLE_WEIGHT,
)


def calculate_move_numbers(game_ids: np.ndarray) -> np.ndarray:
    game_move_counts: dict[int, int] = {}
    move_numbers = np.zeros(len(game_ids), dtype=np.int32)

    for position_index, game_id_value in enumerate(game_ids):
        game_id = int(game_id_value)
        move_numbers[position_index] = game_move_counts.get(game_id, 0)
        game_move_counts[game_id] = move_numbers[position_index] + 1

    return move_numbers


def reweight_game_phases(dataset_path: Path, output_path: Path) -> None:
    dataset = np.load(dataset_path)
    move_numbers = calculate_move_numbers(dataset["game_ids"])
    phase_weights = np.where(
        move_numbers < OPENING_MOVE_COUNT,
        OPENING_SAMPLE_WEIGHT,
        np.where(
            move_numbers < MID_GAME_MOVE_COUNT,
            MID_GAME_SAMPLE_WEIGHT,
            1,
        ),
    ).astype(np.float32)
    existing_weights = (
        dataset["sample_weights"].astype(np.float32)
        if "sample_weights" in dataset
        else np.ones(len(move_numbers), dtype=np.float32)
    )
    weights = existing_weights * phase_weights
    weights /= np.mean(weights)
    output_values = {name: dataset[name] for name in dataset.files}
    output_values["sample_weights"] = weights.astype(np.float16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output_values)
    print(
        f"saved {output_path} "
        f"mean={float(np.mean(weights)):.3f} "
        f"opening={float(np.mean(weights[move_numbers < OPENING_MOVE_COUNT])):.3f} "
        f"late={float(np.mean(weights[move_numbers >= MID_GAME_MOVE_COUNT])):.3f}"
    )


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data", required=True, type=Path)
    argument_parser.add_argument("--output", required=True, type=Path)
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    reweight_game_phases(arguments.data, arguments.output)


if __name__ == "__main__":
    main()
