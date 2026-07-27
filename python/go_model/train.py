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
    SPLIT_BUCKET_COUNT,
    SYMMETRY_FLIP_OPTION_COUNT,
    SYMMETRY_ROTATION_COUNT,
    TEST_BUCKET_INDEX,
    VALIDATION_BUCKET_INDEX,
    VALUE_LOSS_WEIGHT,
)
from go_model.model import StudentNetwork
from go_model.symmetry import apply_board_symmetry


def calculate_loss(
    model: StudentNetwork,
    features: mx.array,
    policy_targets: mx.array,
    value_targets: mx.array,
    sample_weights: mx.array,
) -> tuple[mx.array, tuple[mx.array, mx.array]]:
    policy_logits, values = model(features)
    log_probabilities = policy_logits - mx.logsumexp(
        policy_logits,
        axis=1,
        keepdims=True,
    )
    normalized_weights = sample_weights / mx.mean(sample_weights)
    policy_loss = -mx.mean(
        mx.sum(policy_targets * log_probabilities, axis=1) * normalized_weights
    )
    value_loss = mx.mean(mx.square(values - value_targets) * normalized_weights)
    return policy_loss + VALUE_LOSS_WEIGHT * value_loss, (policy_loss, value_loss)


def evaluate(
    model: StudentNetwork,
    features: np.ndarray,
    policy_targets: np.ndarray,
    value_targets: np.ndarray,
    batch_size: int,
) -> tuple[float, float, float]:
    losses: list[float] = []
    correct_move_count = 0
    value_absolute_error_sum = 0.0

    for batch_start in range(0, len(features), batch_size):
        batch_end = batch_start + batch_size
        feature_batch = mx.array(features[batch_start:batch_end], dtype=mx.float32)
        policy_batch = mx.array(policy_targets[batch_start:batch_end], dtype=mx.float32)
        value_batch = mx.array(value_targets[batch_start:batch_end], dtype=mx.float32)
        sample_weights = mx.ones(len(feature_batch), dtype=mx.float32)
        loss, _ = calculate_loss(
            model,
            feature_batch,
            policy_batch,
            value_batch,
            sample_weights,
        )
        policy_logits, values = model(feature_batch)
        mx.eval(loss, policy_logits, values)
        losses.append(float(loss.item()))
        predicted_moves = np.asarray(mx.argmax(policy_logits, axis=1))
        target_moves = np.argmax(policy_targets[batch_start:batch_end], axis=1)
        correct_move_count += int(np.sum(predicted_moves == target_moves))
        predicted_values = np.asarray(values)
        value_absolute_error_sum += float(
            np.sum(np.abs(predicted_values - value_targets[batch_start:batch_end]))
        )

    return (
        float(np.mean(losses)),
        correct_move_count / len(features),
        value_absolute_error_sum / len(features),
    )


def train(
    dataset_path: Path,
    checkpoint_path: Path,
    epoch_count: int,
    batch_size: int,
    learning_rate: float,
    random_seed: int,
    supplemental_dataset_paths: list[Path],
    initial_checkpoint_path: Path | None,
    hard_policy_weight: float,
) -> None:
    dataset = np.load(dataset_path)
    features = dataset["features"].astype(np.float32)
    game_ids = dataset["game_ids"] if "game_ids" in dataset else np.arange(len(features))
    policies = dataset["policies"].astype(np.float32)
    sample_weights = (
        dataset["sample_weights"].astype(np.float32)
        if "sample_weights" in dataset
        else np.ones(len(features), dtype=np.float32)
    )
    values = dataset["values"].astype(np.float32)
    random_generator = np.random.default_rng(random_seed)
    mx.random.seed(random_seed)
    game_buckets = game_ids % SPLIT_BUCKET_COUNT
    validation_mask = game_buckets == VALIDATION_BUCKET_INDEX
    test_mask = game_buckets == TEST_BUCKET_INDEX
    validation_indexes = np.flatnonzero(validation_mask)
    test_indexes = np.flatnonzero(test_mask)
    training_indexes = np.flatnonzero(~validation_mask & ~test_mask)
    training_features = features[training_indexes]
    training_policies = policies[training_indexes]
    training_sample_weights = sample_weights[training_indexes]
    training_values = values[training_indexes]

    for supplemental_dataset_path in supplemental_dataset_paths:
        supplemental_dataset = np.load(supplemental_dataset_path)
        supplemental_features = supplemental_dataset["features"].astype(np.float32)
        supplemental_game_ids = supplemental_dataset["game_ids"]
        supplemental_training_mask = ~np.isin(
            supplemental_game_ids % SPLIT_BUCKET_COUNT,
            [VALIDATION_BUCKET_INDEX, TEST_BUCKET_INDEX],
        )
        supplemental_indexes = np.flatnonzero(supplemental_training_mask)
        supplemental_sample_weights = (
            supplemental_dataset["sample_weights"].astype(np.float32)
            if "sample_weights" in supplemental_dataset
            else np.ones(len(supplemental_features), dtype=np.float32)
        )
        training_features = np.concatenate(
            [training_features, supplemental_features[supplemental_indexes]]
        )
        training_policies = np.concatenate(
            [
                training_policies,
                supplemental_dataset["policies"][supplemental_indexes].astype(
                    np.float32
                ),
            ]
        )
        training_sample_weights = np.concatenate(
            [
                training_sample_weights,
                supplemental_sample_weights[supplemental_indexes],
            ]
        )
        training_values = np.concatenate(
            [
                training_values,
                supplemental_dataset["values"][supplemental_indexes].astype(
                    np.float32
                ),
            ]
        )

    print(
        f"training={len(training_features):,} "
        f"validation={len(validation_indexes):,} "
        f"test={len(test_indexes):,}"
    )
    model = StudentNetwork()

    if initial_checkpoint_path:
        model.load_weights(str(initial_checkpoint_path))

    optimizer = optim.AdamW(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, calculate_loss)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    for epoch_index in range(epoch_count):
        shuffled_indexes = random_generator.permutation(len(training_features))
        training_losses: list[float] = []

        for batch_start in range(0, len(shuffled_indexes), batch_size):
            batch_indexes = shuffled_indexes[batch_start : batch_start + batch_size]
            augmented_features = training_features[batch_indexes].copy()
            augmented_policies = training_policies[batch_indexes].copy()

            for sample_index in range(len(batch_indexes)):
                rotation_count = int(
                    random_generator.integers(SYMMETRY_ROTATION_COUNT)
                )
                should_flip = bool(
                    random_generator.integers(SYMMETRY_FLIP_OPTION_COUNT)
                )
                (
                    augmented_features[sample_index],
                    augmented_policies[sample_index],
                ) = apply_board_symmetry(
                    augmented_features[sample_index],
                    augmented_policies[sample_index],
                    rotation_count,
                    should_flip,
                )

            feature_batch = mx.array(augmented_features, dtype=mx.float32)
            if hard_policy_weight > 0:
                hard_policies = np.zeros_like(augmented_policies)
                hard_policies[
                    np.arange(len(augmented_policies)),
                    np.argmax(augmented_policies, axis=1),
                ] = 1
                augmented_policies = (
                    (1 - hard_policy_weight) * augmented_policies
                    + hard_policy_weight * hard_policies
                )

            policy_batch = mx.array(augmented_policies, dtype=mx.float32)
            value_batch = mx.array(training_values[batch_indexes], dtype=mx.float32)
            weight_batch = mx.array(
                training_sample_weights[batch_indexes],
                dtype=mx.float32,
            )
            (loss, _), gradients = loss_and_grad(
                model,
                feature_batch,
                policy_batch,
                value_batch,
                weight_batch,
            )
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            training_losses.append(float(loss.item()))

        validation_loss, move_agreement, value_error = evaluate(
            model,
            features[validation_indexes],
            policies[validation_indexes],
            values[validation_indexes],
            batch_size,
        )
        print(
            f"epoch {epoch_index + 1:02d} "
            f"train={np.mean(training_losses):.4f} "
            f"validation={validation_loss:.4f} "
            f"move={move_agreement:.1%} "
            f"value_mae={value_error:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            model.save_weights(str(checkpoint_path))

    model.load_weights(str(checkpoint_path))
    test_loss, test_move_agreement, test_value_error = evaluate(
        model,
        features[test_indexes],
        policies[test_indexes],
        values[test_indexes],
        batch_size,
    )
    print(
        f"test={test_loss:.4f} "
        f"move={test_move_agreement:.1%} "
        f"value_mae={test_value_error:.4f}"
    )


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/katago-distillation.npz"),
    )
    argument_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/go-model.safetensors"),
    )
    argument_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCH_COUNT)
    argument_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    argument_parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    argument_parser.add_argument("--initial-checkpoint", type=Path)
    argument_parser.add_argument("--hard-policy-weight", type=float, default=0)
    argument_parser.add_argument(
        "--supplemental-data",
        action="append",
        default=[],
        type=Path,
    )
    argument_parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    train(
        arguments.data,
        arguments.checkpoint,
        arguments.epochs,
        arguments.batch_size,
        arguments.learning_rate,
        arguments.seed,
        arguments.supplemental_data,
        arguments.initial_checkpoint,
        arguments.hard_policy_weight,
    )


if __name__ == "__main__":
    main()
