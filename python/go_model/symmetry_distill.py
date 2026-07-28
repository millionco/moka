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


def mix_source_and_consensus_targets(
    source_policies: np.ndarray,
    source_values: np.ndarray,
    consensus_policies: np.ndarray,
    consensus_values: np.ndarray,
    source_target_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= source_target_weight <= 1:
        raise ValueError("Source target weight must be between zero and one.")
    consensus_target_weight = 1 - source_target_weight
    mixed_policies = (
        source_target_weight * source_policies
        + consensus_target_weight * consensus_policies
    )
    mixed_policies /= np.sum(mixed_policies, axis=1, keepdims=True)
    mixed_values = (
        source_target_weight * source_values
        + consensus_target_weight * consensus_values
    )
    return mixed_policies, mixed_values


def create_symmetry_consensus_dataset(
    dataset_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int,
    source_target_weight: float,
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
    output_policies, output_target_values = (
        mix_source_and_consensus_targets(
            dataset["policies"].astype(np.float32),
            dataset["values"].astype(np.float32),
            consensus_policies,
            consensus_values,
            source_target_weight,
        )
    )
    output_values = {
        name: dataset[name]
        for name in dataset.files
        if name not in ("policies", "values")
    }
    output_values["policies"] = output_policies.astype(np.float16)
    output_values["values"] = output_target_values.astype(np.float16)
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
    argument_parser.add_argument(
        "--source-target-weight",
        type=float,
        default=0,
    )
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    create_symmetry_consensus_dataset(
        arguments.data,
        arguments.checkpoint,
        arguments.output,
        arguments.batch_size,
        arguments.source_target_weight,
    )


if __name__ == "__main__":
    main()
