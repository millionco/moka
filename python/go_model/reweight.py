import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from go_model.config import (
    DEFAULT_BATCH_SIZE,
    POLICY_SURPRISE_MAXIMUM_WEIGHT,
    POLICY_SURPRISE_UNIFORM_WEIGHT,
)
from go_model.model import StudentNetwork


def calculate_policy_surprise_weights(
    model: StudentNetwork,
    features: np.ndarray,
    policies: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    policy_surprises: list[np.ndarray] = []

    for batch_start in range(0, len(features), batch_size):
        batch_end = batch_start + batch_size
        feature_batch = mx.array(features[batch_start:batch_end], dtype=mx.float32)
        policy_logits, _ = model(feature_batch)
        log_probabilities = policy_logits - mx.logsumexp(
            policy_logits,
            axis=1,
            keepdims=True,
        )
        student_log_probabilities = np.asarray(log_probabilities)
        teacher_probabilities = policies[batch_start:batch_end]
        teacher_log_probabilities = np.log(np.maximum(teacher_probabilities, 1e-8))
        policy_surprises.append(
            np.sum(
                teacher_probabilities
                * (teacher_log_probabilities - student_log_probabilities),
                axis=1,
            )
        )

    surprises = np.maximum(np.concatenate(policy_surprises), 0)
    proportional_weights = surprises / max(float(np.mean(surprises)), 1e-8)
    weights = (
        POLICY_SURPRISE_UNIFORM_WEIGHT
        + (1 - POLICY_SURPRISE_UNIFORM_WEIGHT) * proportional_weights
    )
    return np.minimum(weights, POLICY_SURPRISE_MAXIMUM_WEIGHT).astype(np.float32)


def reweight_dataset(
    dataset_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int,
) -> None:
    dataset = np.load(dataset_path)
    features = dataset["features"].astype(np.float32)
    policies = dataset["policies"].astype(np.float32)
    model = StudentNetwork()
    model.load_weights(str(checkpoint_path))
    model.eval()
    weights = calculate_policy_surprise_weights(
        model,
        features,
        policies,
        batch_size,
    )
    output_values = {name: dataset[name] for name in dataset.files}
    output_values["sample_weights"] = weights.astype(np.float16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output_values)
    print(
        f"saved {output_path} "
        f"mean={float(np.mean(weights)):.3f} "
        f"p95={float(np.percentile(weights, 95)):.3f} "
        f"max={float(np.max(weights)):.3f}"
    )


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data", required=True, type=Path)
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument("--output", required=True, type=Path)
    argument_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    reweight_dataset(
        arguments.data,
        arguments.checkpoint,
        arguments.output,
        arguments.batch_size,
    )


if __name__ == "__main__":
    main()
