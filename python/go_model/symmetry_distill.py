import argparse
from pathlib import Path

import numpy as np

from go_model.blunder import evaluate_aligned_symmetry_outputs
from go_model.config import (
    DEFAULT_BATCH_SIZE,
    SEARCH_ROOT_SYMMETRY_GEOMETRIC_POLICY_WEIGHT,
)
from go_model.model import MokaNestedNetwork
from go_model.symmetry import aggregate_symmetry_policies


def create_symmetry_consensus_targets(
    aligned_policies: np.ndarray,
    symmetry_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    consensus_policies = np.stack(
        [
            aggregate_symmetry_policies(
                list(sample_policies),
                SEARCH_ROOT_SYMMETRY_GEOMETRIC_POLICY_WEIGHT,
            )
            for sample_policies in aligned_policies
        ]
    )
    consensus_values = np.mean(symmetry_values, axis=1)
    return consensus_policies, consensus_values


def create_symmetry_consensus_dataset(
    dataset_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int,
) -> None:
    dataset = np.load(dataset_path)
    features = dataset["features"].astype(np.float32)
    model = MokaNestedNetwork()
    model.load_weights(str(checkpoint_path))
    model.eval()
    aligned_policies, symmetry_values = (
        evaluate_aligned_symmetry_outputs(
            model,
            features,
            batch_size,
        )
    )
    consensus_policies, consensus_values = (
        create_symmetry_consensus_targets(
            aligned_policies,
            symmetry_values,
        )
    )
    output_values = {
        name: dataset[name]
        for name in dataset.files
        if name not in ("policies", "values")
    }
    output_values["policies"] = consensus_policies.astype(np.float16)
    output_values["values"] = consensus_values.astype(np.float16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output_values)
    print(
        f"saved {output_path} "
        f"positions={len(features):,} "
        f"bytes={output_path.stat().st_size:,}"
    )


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data", required=True, type=Path)
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument("--output", required=True, type=Path)
    argument_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    create_symmetry_consensus_dataset(
        arguments.data,
        arguments.checkpoint,
        arguments.output,
        arguments.batch_size,
    )


if __name__ == "__main__":
    main()
