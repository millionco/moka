import argparse
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from go_model.config import (
    ACTION_VALUE_PRIOR_MINIMUM_Q_GAP,
    ACTION_VALUE_PRIOR_MINIMUM_VISIT_COUNT,
    ACTION_VALUE_PRIOR_RIDGE_WEIGHT,
    BOARD_AREA,
    POLICY_MOVE_COUNT,
    SPLIT_BUCKET_COUNT,
)
from go_model.model import (
    MokaActionValueNetwork,
    MokaNetwork,
    create_moka_network_for_checkpoint,
)
from go_model.split import create_game_split_buckets


@dataclass
class ActionValueDataset:
    path: Path
    representations: np.ndarray
    q_values: np.ndarray
    q_weights: np.ndarray
    policies: np.ndarray
    split_buckets: np.ndarray


def evaluate_trunk_values(
    model: MokaNetwork,
    features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    trunk_batches: list[np.ndarray] = []
    for batch_start in range(0, len(features), batch_size):
        trunk_values = model.get_trunk_values(
            mx.array(
                features[batch_start : batch_start + batch_size],
                dtype=mx.float32,
            )
        )
        mx.eval(trunk_values)
        trunk_batches.append(np.asarray(trunk_values, dtype=np.float32))
    return np.concatenate(trunk_batches)


def create_action_representations(
    trunk_values: np.ndarray,
    policies: np.ndarray,
) -> np.ndarray:
    trunk_channel_count = trunk_values.shape[-1]
    representations = np.zeros(
        (
            len(trunk_values),
            POLICY_MOVE_COUNT,
            trunk_channel_count + 2,
        ),
        dtype=np.float32,
    )
    representations[:, :BOARD_AREA, :trunk_channel_count] = (
        trunk_values.reshape(len(trunk_values), BOARD_AREA, trunk_channel_count)
    )
    representations[:, BOARD_AREA, :trunk_channel_count] = np.mean(
        trunk_values,
        axis=(1, 2),
    )
    representations[:, BOARD_AREA, trunk_channel_count] = 1
    representations[:, :, trunk_channel_count + 1] = np.log(
        np.maximum(policies, np.finfo(np.float32).tiny)
    )
    return representations


def load_action_value_dataset(
    path: Path,
    model: MokaNetwork,
    batch_size: int,
) -> ActionValueDataset:
    with np.load(path) as archive:
        required_keys = {
            "features",
            "game_ids",
            "policies",
            "q_values",
            "q_weights",
        }
        missing_keys = required_keys - set(archive.files)
        if missing_keys:
            raise ValueError(
                f"{path} is missing: {', '.join(sorted(missing_keys))}"
            )
        features = archive["features"].astype(np.float32)
        policies = archive["policies"].astype(np.float32)
        q_values = archive["q_values"].astype(np.float32)
        q_weights = archive["q_weights"].astype(np.float32)
        game_ids = archive["game_ids"].astype(np.int64)
    trunk_values = evaluate_trunk_values(model, features, batch_size)
    return ActionValueDataset(
        path=path,
        representations=create_action_representations(
            trunk_values,
            policies,
        ),
        q_values=q_values,
        q_weights=q_weights,
        policies=policies,
        split_buckets=create_game_split_buckets(game_ids, 2),
    )


def create_rank_pairs(
    dataset: ActionValueDataset,
    selected_buckets: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_features: list[np.ndarray] = []
    pair_targets: list[float] = []
    policy_orders: list[bool] = []
    selected_rows = np.flatnonzero(
        np.isin(dataset.split_buckets, list(selected_buckets))
    )
    for row_index in selected_rows:
        visited_moves = np.flatnonzero(
            dataset.q_weights[row_index]
            >= ACTION_VALUE_PRIOR_MINIMUM_VISIT_COUNT
        )
        for first_index, first_move in enumerate(visited_moves):
            for second_move in visited_moves[first_index + 1 :]:
                q_gap = float(
                    dataset.q_values[row_index, first_move]
                    - dataset.q_values[row_index, second_move]
                )
                if abs(q_gap) < ACTION_VALUE_PRIOR_MINIMUM_Q_GAP:
                    continue
                better_move, worse_move = (
                    (first_move, second_move)
                    if q_gap > 0
                    else (second_move, first_move)
                )
                pair_features.append(
                    dataset.representations[row_index, better_move]
                    - dataset.representations[row_index, worse_move]
                )
                pair_targets.append(abs(q_gap))
                policy_orders.append(
                    dataset.policies[row_index, better_move]
                    > dataset.policies[row_index, worse_move]
                )
    if not pair_features:
        feature_count = dataset.representations.shape[-1]
        return (
            np.empty((0, feature_count), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.bool_),
        )
    return (
        np.asarray(pair_features, dtype=np.float32),
        np.asarray(pair_targets, dtype=np.float32),
        np.asarray(policy_orders, dtype=np.bool_),
    )


def fit_action_value_prior(
    datasets: list[ActionValueDataset],
) -> np.ndarray:
    training_buckets = set(range(2, SPLIT_BUCKET_COUNT))
    pair_sets = [
        create_rank_pairs(dataset, training_buckets) for dataset in datasets
    ]
    pair_features = np.concatenate([pair_set[0] for pair_set in pair_sets])
    pair_targets = np.concatenate([pair_set[1] for pair_set in pair_sets])
    feature_scales = np.std(pair_features, axis=0)
    feature_scales = np.where(feature_scales > 1e-6, feature_scales, 1)
    scaled_features = pair_features / feature_scales
    regularizer = np.eye(scaled_features.shape[1], dtype=np.float64)
    scaled_weights = np.linalg.solve(
        scaled_features.T @ scaled_features
        + ACTION_VALUE_PRIOR_RIDGE_WEIGHT * regularizer,
        scaled_features.T @ pair_targets,
    )
    print(f"training pairs={len(pair_targets):,}")
    return np.asarray(scaled_weights / feature_scales, dtype=np.float32)


def report_pair_metrics(
    datasets: list[ActionValueDataset],
    weights: np.ndarray,
    bucket: int,
    label: str,
) -> None:
    for dataset in datasets:
        pair_features, pair_targets, policy_orders = create_rank_pairs(
            dataset,
            {bucket},
        )
        predicted_gaps = pair_features @ weights
        rank_accuracy = float(np.mean(predicted_gaps > 0))
        policy_accuracy = float(np.mean(policy_orders))
        mean_squared_error = float(
            np.mean(np.square(predicted_gaps - pair_targets))
        )
        print(
            f"{label} {dataset.path.name} pairs={len(pair_targets):,} "
            f"rank={rank_accuracy:.4%} policy={policy_accuracy:.4%} "
            f"delta={rank_accuracy - policy_accuracy:+.4%} "
            f"mse={mean_squared_error:.6f}"
        )


def save_action_value_checkpoint(
    base_checkpoint_path: Path,
    output_path: Path,
    weights: np.ndarray,
) -> None:
    model = create_moka_network_for_checkpoint(
        str(base_checkpoint_path),
        use_action_value_network=True,
    )
    if not isinstance(model, MokaActionValueNetwork):
        raise ValueError("Action-value checkpoint requires a compatible model.")
    model.load_weights(str(base_checkpoint_path), strict=False)
    trunk_channel_count = model.action_value_spatial.weight.shape[1]
    model.action_value_spatial.weight = mx.array(
        weights[:trunk_channel_count][None, :],
        dtype=mx.float32,
    )
    model.action_value_pass_bias = mx.array(
        weights[trunk_channel_count : trunk_channel_count + 1],
        dtype=mx.float32,
    )
    model.action_value_policy_scale = mx.array(
        weights[trunk_channel_count + 1 : trunk_channel_count + 2],
        dtype=mx.float32,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(output_path))
    print(f"saved {output_path} bytes={output_path.stat().st_size:,}")


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument(
        "--data",
        required=True,
        action="append",
        type=Path,
    )
    argument_parser.add_argument("--output", required=True, type=Path)
    argument_parser.add_argument("--batch-size", type=int, default=256)
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    model = create_moka_network_for_checkpoint(str(arguments.checkpoint))
    model.load_weights(str(arguments.checkpoint))
    model.eval()
    datasets = [
        load_action_value_dataset(path, model, arguments.batch_size)
        for path in arguments.data
    ]
    weights = fit_action_value_prior(datasets)
    report_pair_metrics(datasets, weights, 0, "validation")
    report_pair_metrics(datasets, weights, 1, "test")
    save_action_value_checkpoint(
        arguments.checkpoint,
        arguments.output,
        weights,
    )


if __name__ == "__main__":
    main()
