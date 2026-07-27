import argparse
from pathlib import Path

import mlx.core as mx


def average_checkpoints(
    first_checkpoint_path: Path,
    second_checkpoint_path: Path,
    second_checkpoint_weight: float,
    output_path: Path,
) -> None:
    first_parameters = mx.load(str(first_checkpoint_path))
    second_parameters = mx.load(str(second_checkpoint_path))

    if first_parameters.keys() != second_parameters.keys():
        raise ValueError("Checkpoint parameters do not match.")

    averaged_parameters = {
        name: (1 - second_checkpoint_weight) * first_parameters[name]
        + second_checkpoint_weight * second_parameters[name]
        for name in first_parameters
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output_path), averaged_parameters)
    print(
        f"saved {output_path} "
        f"first={1 - second_checkpoint_weight:.2f} "
        f"second={second_checkpoint_weight:.2f}"
    )


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--first", required=True, type=Path)
    argument_parser.add_argument("--second", required=True, type=Path)
    argument_parser.add_argument("--second-weight", required=True, type=float)
    argument_parser.add_argument("--output", required=True, type=Path)
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    average_checkpoints(
        arguments.first,
        arguments.second,
        arguments.second_weight,
        arguments.output,
    )


if __name__ == "__main__":
    main()
