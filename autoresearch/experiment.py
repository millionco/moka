import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENT_EPOCH_COUNT = 2
EXPERIMENT_LEARNING_RATE = 0.00001
SEARCH_REPLAY_COUNT = 8


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--seed", required=True, type=int)
    argument_parser.add_argument("--output", required=True, type=Path)
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    search_dataset_path = "data/moka-search-selective-512x128-offset50k.npz"
    supplemental_dataset_arguments = [
        "--supplemental-data",
        "data/moka-opponent-dagger-50k.npz",
    ]

    for _ in range(SEARCH_REPLAY_COUNT):
        supplemental_dataset_arguments.extend(
            ["--supplemental-data", search_dataset_path]
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "go_model.train",
            "--data",
            "data/strong-teacher-batched-100k.npz",
            "--checkpoint",
            str(arguments.output),
            "--epochs",
            str(EXPERIMENT_EPOCH_COUNT),
            "--learning-rate",
            str(EXPERIMENT_LEARNING_RATE),
            "--initial-checkpoint",
            "checkpoints/moka-dgrpo-prefix-v1.safetensors",
            "--nested",
            "--freeze-trunk",
            "--selection-metric",
            "move",
            "--seed",
            str(arguments.seed),
            *supplemental_dataset_arguments,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
