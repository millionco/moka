import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from go_model.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCH_COUNT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_RANDOM_SEED,
    PAIRWISE_VALUE_MINIMUM_GAP,
    SPLIT_BUCKET_COUNT,
    TEST_BUCKET_INDEX,
    VALIDATION_BUCKET_INDEX,
)
from go_model.model import (
    MokaNestedNetwork,
    create_moka_network_for_checkpoint,
)


def calculate_value_loss(
    model: MokaNestedNetwork,
    features: mx.array,
    targets: mx.array,
    sample_weights: mx.array,
    reference_policy_logits: mx.array,
    policy_preservation_weight: float,
) -> mx.array:
    policy_logits, values = model(features)
    value_loss = mx.sum(sample_weights * mx.square(values - targets)) / mx.sum(
        sample_weights
    )
    policy_preservation_loss = mx.mean(
        mx.square(policy_logits - reference_policy_logits)
    )
    return (
        value_loss
        + policy_preservation_weight * policy_preservation_loss
    )


def calculate_ranked_value_loss(
    model: MokaNestedNetwork,
    features: mx.array,
    targets: mx.array,
    sample_weights: mx.array,
    reference_policy_logits: mx.array,
    policy_preservation_weight: float,
    preferred_features: mx.array,
    alternative_features: mx.array,
    target_value_gaps: mx.array,
    pair_weights: mx.array,
    pairwise_ranking_weight: float,
) -> mx.array:
    pointwise_loss = calculate_value_loss(
        model,
        features,
        targets,
        sample_weights,
        reference_policy_logits,
        policy_preservation_weight,
    )
    _, preferred_values = model(preferred_features)
    _, alternative_values = model(alternative_features)
    predicted_value_gaps = alternative_values - preferred_values
    pairwise_loss = mx.sum(
        pair_weights * mx.square(predicted_value_gaps - target_value_gaps)
    ) / mx.sum(pair_weights)
    return pointwise_loss + pairwise_ranking_weight * pairwise_loss


def create_child_value_pairs(
    dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    child_values = dataset["child_values"].astype(np.float32)
    child_weights = dataset["child_weights"].astype(np.float32)
    child_game_ids = dataset["child_game_ids"]
    root_value_key = (
        "q_values"
        if "q_values" in dataset
        else "search_q_values"
    )
    child_root_indexes = (
        dataset["child_root_indexes"]
        if "child_root_indexes" in dataset
        else np.repeat(
            np.arange(len(dataset[root_value_key])),
            np.count_nonzero(dataset[root_value_key], axis=1),
        )
    )

    if len(child_root_indexes) != len(child_values):
        raise ValueError("Child states do not align with root Q targets.")
    if np.any(np.diff(child_root_indexes) < 0):
        raise ValueError("Child root indexes must be sorted.")

    preferred_indexes: list[int] = []
    alternative_indexes: list[int] = []
    target_value_gaps: list[float] = []
    pair_weights: list[float] = []
    pair_game_ids: list[int] = []
    for root_index in range(len(dataset[root_value_key])):
        child_start = int(
            np.searchsorted(child_root_indexes, root_index, side="left")
        )
        child_end = int(
            np.searchsorted(child_root_indexes, root_index, side="right")
        )
        root_indexes = np.arange(child_start, child_end)
        valid_indexes = root_indexes[child_weights[root_indexes] > 0]

        if len(valid_indexes) >= 2:
            preferred_index = int(
                valid_indexes[
                    np.argmin(child_values[valid_indexes])
                ]
            )

            for alternative_index in valid_indexes:
                value_gap = (
                    child_values[alternative_index]
                    - child_values[preferred_index]
                )

                if value_gap < PAIRWISE_VALUE_MINIMUM_GAP:
                    continue

                preferred_indexes.append(preferred_index)
                alternative_indexes.append(int(alternative_index))
                target_value_gaps.append(float(value_gap))
                pair_weights.append(
                    float(
                        np.sqrt(
                            child_weights[preferred_index]
                            * child_weights[alternative_index]
                        )
                    )
                )
                pair_game_ids.append(int(child_game_ids[preferred_index]))

    return (
        np.asarray(preferred_indexes, dtype=np.int64),
        np.asarray(alternative_indexes, dtype=np.int64),
        np.asarray(target_value_gaps, dtype=np.float32),
        np.asarray(pair_weights, dtype=np.float32),
        np.asarray(pair_game_ids, dtype=np.int32),
    )


def evaluate_value(
    model: MokaNestedNetwork,
    features: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray,
    batch_size: int,
) -> float:
    absolute_error_sum = 0.0
    weight_sum = 0.0

    for batch_start in range(0, len(features), batch_size):
        batch_end = batch_start + batch_size
        _, values = model(
            mx.array(features[batch_start:batch_end], dtype=mx.float32)
        )
        mx.eval(values)
        batch_weights = sample_weights[batch_start:batch_end]
        absolute_error_sum += float(
            np.sum(
                batch_weights
                * np.abs(
                    np.asarray(values) - targets[batch_start:batch_end]
                )
            )
        )
        weight_sum += float(np.sum(batch_weights))

    return absolute_error_sum / weight_sum


def evaluate_pairwise_value(
    model: MokaNestedNetwork,
    features: np.ndarray,
    preferred_indexes: np.ndarray,
    alternative_indexes: np.ndarray,
    target_value_gaps: np.ndarray,
    pair_weights: np.ndarray,
    pair_indexes: np.ndarray,
    batch_size: int,
) -> float:
    weighted_error_sum = 0.0
    weight_sum = 0.0

    for batch_start in range(0, len(pair_indexes), batch_size):
        batch_indexes = pair_indexes[batch_start : batch_start + batch_size]
        _, preferred_values = model(
            mx.array(
                features[preferred_indexes[batch_indexes]],
                dtype=mx.float32,
            )
        )
        _, alternative_values = model(
            mx.array(
                features[alternative_indexes[batch_indexes]],
                dtype=mx.float32,
            )
        )
        predicted_gaps = alternative_values - preferred_values
        target_gaps = mx.array(
            target_value_gaps[batch_indexes],
            dtype=mx.float32,
        )
        weights = mx.array(pair_weights[batch_indexes], dtype=mx.float32)
        error_sum = mx.sum(weights * mx.square(predicted_gaps - target_gaps))
        mx.eval(error_sum)
        weighted_error_sum += float(error_sum.item())
        weight_sum += float(np.sum(pair_weights[batch_indexes]))

    return weighted_error_sum / weight_sum


def train_value_head(
    dataset_path: Path,
    initial_checkpoint_path: Path,
    checkpoint_path: Path,
    epoch_count: int,
    batch_size: int,
    learning_rate: float,
    random_seed: int,
    use_child_targets: bool,
    should_unfreeze_trunk: bool,
    policy_preservation_weight: float,
    pairwise_ranking_weight: float,
) -> None:
    dataset = np.load(dataset_path)
    key_prefix = "child_" if use_child_targets else ""
    features = dataset[f"{key_prefix}features"].astype(np.float32)
    game_ids = dataset[f"{key_prefix}game_ids"]
    values = dataset[f"{key_prefix}values"].astype(np.float32)
    sample_weights = (
        dataset[f"{key_prefix}weights"].astype(np.float32)
        if f"{key_prefix}weights" in dataset
        else np.ones(len(features), dtype=np.float32)
    )
    game_buckets = game_ids % SPLIT_BUCKET_COUNT
    validation_indexes = np.flatnonzero(game_buckets == VALIDATION_BUCKET_INDEX)
    test_indexes = np.flatnonzero(game_buckets == TEST_BUCKET_INDEX)
    training_indexes = np.flatnonzero(
        (game_buckets != VALIDATION_BUCKET_INDEX)
        & (game_buckets != TEST_BUCKET_INDEX)
    )
    random_generator = np.random.default_rng(random_seed)
    mx.random.seed(random_seed)
    model = create_moka_network_for_checkpoint(
        str(initial_checkpoint_path)
    )
    model.load_weights(str(initial_checkpoint_path))
    reference_model = create_moka_network_for_checkpoint(
        str(initial_checkpoint_path)
    )
    reference_model.load_weights(str(initial_checkpoint_path))
    reference_model.freeze()
    reference_model.eval()
    model.freeze()

    if should_unfreeze_trunk:
        model.unfreeze()
        model.policy_convolution.freeze()
        model.policy_linear.freeze()
    else:
        model.value_convolution.unfreeze()
        model.value_hidden.unfreeze()
        model.value_output.unfreeze()
    optimizer = optim.AdamW(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, calculate_value_loss)
    ranked_loss_and_grad = nn.value_and_grad(
        model,
        calculate_ranked_value_loss,
    )
    if pairwise_ranking_weight > 0:
        (
            preferred_indexes,
            alternative_indexes,
            target_value_gaps,
            pair_weights,
            pair_game_ids,
        ) = create_child_value_pairs(dataset)
        pair_training_indexes = np.flatnonzero(
            ~np.isin(
                pair_game_ids % SPLIT_BUCKET_COUNT,
                [VALIDATION_BUCKET_INDEX, TEST_BUCKET_INDEX],
            )
        )
        print(
            f"pairs={len(preferred_indexes):,} "
            f"training_pairs={len(pair_training_indexes):,}"
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_error = float("inf")
    best_validation_ranking_error = float("inf")

    for epoch_index in range(epoch_count):
        shuffled_indexes = random_generator.permutation(training_indexes)
        training_losses: list[float] = []

        for batch_start in range(0, len(shuffled_indexes), batch_size):
            batch_indexes = shuffled_indexes[batch_start : batch_start + batch_size]
            feature_batch = mx.array(features[batch_indexes], dtype=mx.float32)
            target_batch = mx.array(values[batch_indexes], dtype=mx.float32)
            weight_batch = mx.array(
                sample_weights[batch_indexes],
                dtype=mx.float32,
            )
            reference_policy_logits, _ = reference_model(feature_batch)
            mx.eval(reference_policy_logits)
            if pairwise_ranking_weight > 0:
                pair_batch_indexes = random_generator.choice(
                    pair_training_indexes,
                    size=len(batch_indexes),
                    replace=len(pair_training_indexes) < len(batch_indexes),
                )
                preferred_feature_batch = mx.array(
                    features[preferred_indexes[pair_batch_indexes]],
                    dtype=mx.float32,
                )
                alternative_feature_batch = mx.array(
                    features[alternative_indexes[pair_batch_indexes]],
                    dtype=mx.float32,
                )
                target_value_gap_batch = mx.array(
                    target_value_gaps[pair_batch_indexes],
                    dtype=mx.float32,
                )
                pair_weight_batch = mx.array(
                    pair_weights[pair_batch_indexes],
                    dtype=mx.float32,
                )
                loss, gradients = ranked_loss_and_grad(
                    model,
                    feature_batch,
                    target_batch,
                    weight_batch,
                    reference_policy_logits,
                    policy_preservation_weight,
                    preferred_feature_batch,
                    alternative_feature_batch,
                    target_value_gap_batch,
                    pair_weight_batch,
                    pairwise_ranking_weight,
                )
            else:
                loss, gradients = loss_and_grad(
                    model,
                    feature_batch,
                    target_batch,
                    weight_batch,
                    reference_policy_logits,
                    policy_preservation_weight,
                )
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            training_losses.append(float(loss.item()))

        validation_error = evaluate_value(
            model,
            features[validation_indexes],
            values[validation_indexes],
            sample_weights[validation_indexes],
            batch_size,
        )
        progress = (
            f"epoch {epoch_index + 1:02d} "
            f"train={np.mean(training_losses):.4f} "
            f"validation_mae={validation_error:.4f}"
        )
        if pairwise_ranking_weight > 0:
            pair_validation_indexes = np.flatnonzero(
                pair_game_ids % SPLIT_BUCKET_COUNT == VALIDATION_BUCKET_INDEX
            )
            validation_ranking_error = evaluate_pairwise_value(
                model,
                features,
                preferred_indexes,
                alternative_indexes,
                target_value_gaps,
                pair_weights,
                pair_validation_indexes,
                batch_size,
            )
            progress += f" validation_pair_mse={validation_ranking_error:.4f}"
        else:
            validation_ranking_error = float("inf")
        print(progress)
        did_improve = (
            validation_ranking_error < best_validation_ranking_error
            if pairwise_ranking_weight > 0
            else validation_error < best_validation_error
        )

        if did_improve:
            best_validation_error = validation_error
            best_validation_ranking_error = validation_ranking_error
            model.save_weights(str(checkpoint_path))

    model.load_weights(str(checkpoint_path))
    test_error = evaluate_value(
        model,
        features[test_indexes],
        values[test_indexes],
        sample_weights[test_indexes],
        batch_size,
    )
    print(f"test_mae={test_error:.4f}")
    if pairwise_ranking_weight > 0:
        pair_test_indexes = np.flatnonzero(
            pair_game_ids % SPLIT_BUCKET_COUNT == TEST_BUCKET_INDEX
        )
        test_ranking_error = evaluate_pairwise_value(
            model,
            features,
            preferred_indexes,
            alternative_indexes,
            target_value_gaps,
            pair_weights,
            pair_test_indexes,
            batch_size,
        )
        print(f"test_pair_mse={test_ranking_error:.4f}")


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data", required=True, type=Path)
    argument_parser.add_argument("--initial-checkpoint", required=True, type=Path)
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCH_COUNT)
    argument_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    argument_parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    argument_parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    argument_parser.add_argument("--child-targets", action="store_true")
    argument_parser.add_argument("--unfreeze-trunk", action="store_true")
    argument_parser.add_argument(
        "--policy-preservation-weight",
        type=float,
        default=0,
    )
    argument_parser.add_argument(
        "--pairwise-ranking-weight",
        type=float,
        default=0,
    )
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    train_value_head(
        arguments.data,
        arguments.initial_checkpoint,
        arguments.checkpoint,
        arguments.epochs,
        arguments.batch_size,
        arguments.learning_rate,
        arguments.seed,
        arguments.child_targets,
        arguments.unfreeze_trunk,
        arguments.policy_preservation_weight,
        arguments.pairwise_ranking_weight,
    )


if __name__ == "__main__":
    main()
