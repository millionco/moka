import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from go_model.config import (
    ARENA_OPENING_PAIR_SIZE,
    DEFAULT_BATCH_SIZE,
    SEARCH_UNCERTAINTY_TARGET_EPSILON,
    TEST_BUCKET_INDEX,
    VALIDATION_BUCKET_INDEX,
)
from go_model.model import (
    MokaUncertaintyNetwork,
    create_moka_network_for_checkpoint,
)
from go_model.split import create_game_split_buckets


def load_uncertainty_dataset(
    dataset_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(dataset_path) as dataset:
        if "short_winrate_errors" not in dataset:
            raise ValueError(
                f"Short-term uncertainty targets are missing from {dataset_path}."
            )
        features = dataset["features"].astype(np.float32)
        game_ids = dataset["game_ids"].astype(np.int64)
        uncertainty_targets = dataset["short_winrate_errors"].astype(
            np.float64
        )
    if not np.all(np.isfinite(uncertainty_targets)):
        raise ValueError(f"Uncertainty targets are invalid in {dataset_path}.")
    return features, game_ids, uncertainty_targets


def extract_uncertainty_features(
    model: MokaUncertaintyNetwork,
    features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    feature_batches: list[np.ndarray] = []

    for batch_start in range(0, len(features), batch_size):
        batch_end = batch_start + batch_size
        trunk_values = model.get_trunk_values(
            mx.array(features[batch_start:batch_end], dtype=mx.float32)
        )
        mx.eval(trunk_values)
        trunk_array = np.asarray(trunk_values, dtype=np.float64)
        feature_batches.append(
            np.concatenate(
                [
                    np.mean(trunk_array, axis=(1, 2)),
                    np.max(trunk_array, axis=(1, 2)),
                ],
                axis=1,
            )
        )

    return np.concatenate(feature_batches)


def fit_uncertainty_weights(
    features: np.ndarray,
    uncertainty_targets: np.ndarray,
) -> tuple[np.ndarray, float]:
    log_targets = np.log(
        uncertainty_targets + SEARCH_UNCERTAINTY_TARGET_EPSILON
    )
    feature_means = np.mean(features, axis=0)
    feature_scales = np.std(features, axis=0) + 1e-6
    standardized_features = (features - feature_means) / feature_scales
    design = np.concatenate(
        [
            standardized_features,
            np.ones((len(features), 1), dtype=np.float64),
        ],
        axis=1,
    )
    ridge = np.eye(design.shape[1], dtype=np.float64)
    standardized_weights = np.linalg.solve(
        design.T @ design + ridge,
        design.T @ log_targets,
    )
    weights = standardized_weights[:-1] / feature_scales
    bias = float(
        standardized_weights[-1]
        - np.sum(standardized_weights[:-1] * feature_means / feature_scales)
    )
    return weights.astype(np.float32), bias


def calculate_uncertainty_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    baseline_log_target: float,
) -> tuple[float, float, float]:
    log_targets = np.log(targets + SEARCH_UNCERTAINTY_TARGET_EPSILON)
    mean_squared_error = float(np.mean(np.square(predictions - log_targets)))
    baseline_error = float(
        np.mean(np.square(baseline_log_target - log_targets))
    )
    explained_error = 1 - mean_squared_error / baseline_error
    correlation = float(np.corrcoef(predictions, log_targets)[0, 1])
    return mean_squared_error, explained_error, correlation


def save_uncertainty_checkpoint(
    initial_checkpoint_path: Path,
    checkpoint_path: Path,
    weights: np.ndarray,
    bias: float,
) -> None:
    model = create_moka_network_for_checkpoint(
        str(initial_checkpoint_path),
        use_uncertainty_network=True,
    )
    if not isinstance(model, MokaUncertaintyNetwork):
        raise ValueError("Uncertainty checkpoint requires a compatible model.")
    model.load_weights(str(initial_checkpoint_path), strict=False)
    model.uncertainty_output.weight = mx.array(weights[None, :])
    model.uncertainty_output.bias = mx.array([bias])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(checkpoint_path))


def fit_uncertainty_head(
    dataset_path: Path,
    test_dataset_path: Path,
    initial_checkpoint_path: Path,
    checkpoint_path: Path,
    batch_size: int,
) -> None:
    features, game_ids, uncertainty_targets = load_uncertainty_dataset(
        dataset_path
    )
    test_features, _, test_uncertainty_targets = load_uncertainty_dataset(
        test_dataset_path
    )
    model = create_moka_network_for_checkpoint(
        str(initial_checkpoint_path),
        use_uncertainty_network=True,
    )
    if not isinstance(model, MokaUncertaintyNetwork):
        raise ValueError("Uncertainty fitting requires a compatible model.")
    model.load_weights(str(initial_checkpoint_path), strict=False)
    pooled_features = extract_uncertainty_features(model, features, batch_size)
    test_pooled_features = extract_uncertainty_features(
        model,
        test_features,
        batch_size,
    )
    buckets = create_game_split_buckets(game_ids, ARENA_OPENING_PAIR_SIZE)
    training_indexes = np.flatnonzero(
        (buckets != VALIDATION_BUCKET_INDEX)
        & (buckets != TEST_BUCKET_INDEX)
    )
    validation_indexes = np.flatnonzero(
        buckets == VALIDATION_BUCKET_INDEX
    )
    weights, bias = fit_uncertainty_weights(
        pooled_features[training_indexes],
        uncertainty_targets[training_indexes],
    )
    baseline_log_target = float(
        np.mean(
            np.log(
                uncertainty_targets[training_indexes]
                + SEARCH_UNCERTAINTY_TARGET_EPSILON
            )
        )
    )

    for label, metric_features, metric_targets in (
        (
            "validation",
            pooled_features[validation_indexes],
            uncertainty_targets[validation_indexes],
        ),
        ("external", test_pooled_features, test_uncertainty_targets),
    ):
        predictions = metric_features @ weights + bias
        mean_squared_error, explained_error, correlation = (
            calculate_uncertainty_metrics(
                predictions,
                metric_targets,
                baseline_log_target,
            )
        )
        print(
            f"{label} mse={mean_squared_error:.5f} "
            f"explained={explained_error:.3%} correlation={correlation:.4f}"
        )

    save_uncertainty_checkpoint(
        initial_checkpoint_path,
        checkpoint_path,
        weights,
        bias,
    )
    print(
        f"saved {checkpoint_path} bytes={checkpoint_path.stat().st_size:,}"
    )


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data", required=True, type=Path)
    argument_parser.add_argument("--test-data", required=True, type=Path)
    argument_parser.add_argument(
        "--initial-checkpoint",
        required=True,
        type=Path,
    )
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    fit_uncertainty_head(
        arguments.data,
        arguments.test_data,
        arguments.initial_checkpoint,
        arguments.checkpoint,
        arguments.batch_size,
    )


if __name__ == "__main__":
    main()
