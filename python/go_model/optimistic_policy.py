import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from go_model.config import (
    ARENA_OPENING_PAIR_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCH_COUNT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_RANDOM_SEED,
    OPTIMISTIC_POLICY_RANK_LOSS_MARGIN,
    OPTIMISTIC_POLICY_RANK_LOSS_WEIGHT,
    POLICY_MOVE_COUNT,
    SYMMETRY_FLIP_OPTION_COUNT,
    SYMMETRY_ROTATION_COUNT,
    TEST_BUCKET_INDEX,
    VALIDATION_BUCKET_INDEX,
)
from go_model.model import (
    MokaOptimisticPolicyNetwork,
    create_moka_network_for_checkpoint,
    get_checkpoint_global_residual_block_interval,
)
from go_model.split import create_game_split_buckets
from go_model.symmetry import apply_batch_board_symmetry


def load_optimistic_policy_datasets(
    dataset_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_batches: list[np.ndarray] = []
    game_id_batches: list[np.ndarray] = []
    policy_batches: list[np.ndarray] = []
    game_id_offset = 0

    for dataset_path in dataset_paths:
        with np.load(dataset_path) as dataset:
            if "optimistic_policies" not in dataset:
                raise ValueError(
                    f"Optimistic-policy targets are missing from {dataset_path}."
                )
            features = dataset["features"].astype(np.float32)
            game_ids = dataset["game_ids"].astype(np.int64)
            policies = dataset["optimistic_policies"].astype(np.float32)
        if policies.shape != (len(features), POLICY_MOVE_COUNT):
            raise ValueError(
                f"Unexpected optimistic-policy shape in {dataset_path}: "
                f"{policies.shape}."
            )
        if not np.all(np.isfinite(policies)) or np.any(policies < 0):
            raise ValueError(
                f"Optimistic-policy targets are invalid in {dataset_path}."
            )
        policy_sums = np.sum(policies, axis=1, keepdims=True)
        if np.any(policy_sums <= 0):
            raise ValueError(
                f"Optimistic-policy targets are empty in {dataset_path}."
            )
        policies /= policy_sums
        feature_batches.append(features)
        game_id_batches.append(game_ids + game_id_offset)
        policy_batches.append(policies)
        game_id_offset += int(np.max(game_ids)) + 1

    return (
        np.concatenate(feature_batches),
        np.concatenate(game_id_batches),
        np.concatenate(policy_batches),
    )


def calculate_optimistic_policy_loss(
    model: MokaOptimisticPolicyNetwork,
    features: mx.array,
    policy_targets: mx.array,
) -> mx.array:
    _, _, optimistic_policy_logits = model.get_optimistic_policy_outputs(
        features
    )
    masked_logits = mx.where(
        policy_targets > 0,
        optimistic_policy_logits,
        mx.array(-1e9, dtype=optimistic_policy_logits.dtype),
    )
    log_probabilities = masked_logits - mx.logsumexp(
        masked_logits,
        axis=1,
        keepdims=True,
    )
    cross_entropy = -mx.mean(
        mx.sum(policy_targets * log_probabilities, axis=1)
    )
    ranking_loss = calculate_policy_ranking_loss(
        optimistic_policy_logits,
        policy_targets,
    )
    return cross_entropy + OPTIMISTIC_POLICY_RANK_LOSS_WEIGHT * ranking_loss


def calculate_policy_ranking_loss(
    policy_logits: mx.array,
    policy_targets: mx.array,
) -> mx.array:
    top_moves = mx.argmax(policy_targets, axis=1, keepdims=True)
    top_move_logits = mx.take_along_axis(
        policy_logits,
        top_moves,
        axis=1,
    ).squeeze(1)
    move_indexes = mx.arange(policy_logits.shape[1])[None, :]
    alternative_logits = mx.where(
        (move_indexes != top_moves) & (policy_targets > 0),
        policy_logits,
        mx.array(-1e9, dtype=policy_logits.dtype),
    )
    highest_alternative_logits = mx.max(alternative_logits, axis=1)
    return mx.mean(
        mx.maximum(
            OPTIMISTIC_POLICY_RANK_LOSS_MARGIN
            - top_move_logits
            + highest_alternative_logits,
            0,
        )
    )


def evaluate_policy_logits(
    policy_logits: np.ndarray,
    policy_targets: np.ndarray,
) -> tuple[float, float]:
    masked_logits = np.where(policy_targets > 0, policy_logits, -1e9)
    maximum_logits = np.max(masked_logits, axis=1, keepdims=True)
    probabilities = np.exp(masked_logits - maximum_logits)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    cross_entropy = float(
        -np.mean(
            np.sum(
                policy_targets
                * np.log(np.maximum(probabilities, np.finfo(np.float32).tiny)),
                axis=1,
            )
        )
    )
    top_move_agreement = float(
        np.mean(
            np.argmax(probabilities, axis=1)
            == np.argmax(policy_targets, axis=1)
        )
    )
    return cross_entropy, top_move_agreement


def evaluate_optimistic_policy_model(
    model: MokaOptimisticPolicyNetwork,
    features: np.ndarray,
    policy_targets: np.ndarray,
    batch_size: int,
) -> tuple[float, float]:
    logit_batches: list[np.ndarray] = []

    for batch_start in range(0, len(features), batch_size):
        batch_end = batch_start + batch_size
        _, _, optimistic_policy_logits = model.get_optimistic_policy_outputs(
            mx.array(features[batch_start:batch_end], dtype=mx.float32)
        )
        mx.eval(optimistic_policy_logits)
        logit_batches.append(np.asarray(optimistic_policy_logits))

    return evaluate_policy_logits(
        np.concatenate(logit_batches),
        policy_targets,
    )


def evaluate_normal_policy_model(
    model: MokaOptimisticPolicyNetwork,
    features: np.ndarray,
    policy_targets: np.ndarray,
    batch_size: int,
) -> tuple[float, float]:
    logit_batches: list[np.ndarray] = []

    for batch_start in range(0, len(features), batch_size):
        batch_end = batch_start + batch_size
        policy_logits, _ = model(
            mx.array(features[batch_start:batch_end], dtype=mx.float32)
        )
        mx.eval(policy_logits)
        logit_batches.append(np.asarray(policy_logits))

    return evaluate_policy_logits(
        np.concatenate(logit_batches),
        policy_targets,
    )


def train_optimistic_policy_head(
    dataset_paths: list[Path],
    test_dataset_path: Path,
    initial_checkpoint_path: Path,
    checkpoint_path: Path,
    epoch_count: int,
    batch_size: int,
    learning_rate: float,
    random_seed: int,
) -> None:
    features, game_ids, policy_targets = load_optimistic_policy_datasets(
        dataset_paths
    )
    test_features, _, test_policy_targets = load_optimistic_policy_datasets(
        [test_dataset_path]
    )
    buckets = create_game_split_buckets(
        game_ids,
        ARENA_OPENING_PAIR_SIZE,
    )
    training_indexes = np.flatnonzero(
        (buckets != VALIDATION_BUCKET_INDEX)
        & (buckets != TEST_BUCKET_INDEX)
    )
    validation_indexes = np.flatnonzero(
        buckets == VALIDATION_BUCKET_INDEX
    )
    random_generator = np.random.default_rng(random_seed)
    mx.random.seed(random_seed)
    global_residual_block_interval = (
        get_checkpoint_global_residual_block_interval(
            str(initial_checkpoint_path)
        )
    )
    if global_residual_block_interval == 0:
        raise ValueError(
            "Optimistic-policy training requires a global-residual checkpoint."
        )
    model = create_moka_network_for_checkpoint(
        str(initial_checkpoint_path),
        use_optimistic_policy_network=True,
    )
    if not isinstance(model, MokaOptimisticPolicyNetwork):
        raise ValueError("Optimistic-policy checkpoint requires a compatible model.")
    model.load_weights(str(initial_checkpoint_path), strict=False)
    model.freeze()
    model.optimistic_policy_convolution.unfreeze()
    model.optimistic_policy_pass.unfreeze()
    model.optimistic_policy_hidden.unfreeze()
    model.optimistic_policy_output.unfreeze()
    model.unfreeze(
        recurse=False,
        keys="optimistic_policy_scale",
        strict=True,
    )
    loss_and_grad = nn.value_and_grad(
        model,
        calculate_optimistic_policy_loss,
    )
    optimizer = optim.AdamW(learning_rate=learning_rate)
    best_validation_cross_entropy = float("inf")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    normal_cross_entropy, normal_top_move_agreement = (
        evaluate_normal_policy_model(
            model,
            test_features,
            test_policy_targets,
            batch_size,
        )
    )
    print(
        f"normal_test_cross_entropy={normal_cross_entropy:.5f} "
        f"normal_test_top={normal_top_move_agreement:.3%}"
    )

    for epoch_index in range(epoch_count):
        shuffled_indexes = random_generator.permutation(training_indexes)
        training_losses: list[float] = []

        for batch_start in range(0, len(shuffled_indexes), batch_size):
            batch_indexes = shuffled_indexes[
                batch_start : batch_start + batch_size
            ]
            rotation_counts = random_generator.integers(
                0,
                SYMMETRY_ROTATION_COUNT,
                len(batch_indexes),
            )
            should_flip = random_generator.integers(
                0,
                SYMMETRY_FLIP_OPTION_COUNT,
                len(batch_indexes),
                dtype=np.int8,
            ).astype(bool)
            augmented_features, augmented_policies = (
                apply_batch_board_symmetry(
                    features[batch_indexes],
                    policy_targets[batch_indexes],
                    rotation_counts,
                    should_flip,
                )
            )
            loss, gradients = loss_and_grad(
                model,
                mx.array(augmented_features, dtype=mx.float32),
                mx.array(augmented_policies, dtype=mx.float32),
            )
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            training_losses.append(float(loss.item()))

        validation_cross_entropy, validation_top_move_agreement = (
            evaluate_optimistic_policy_model(
                model,
                features[validation_indexes],
                policy_targets[validation_indexes],
                batch_size,
            )
        )
        print(
            f"epoch {epoch_index + 1:02d} "
            f"train={np.mean(training_losses):.5f} "
            f"validation_cross_entropy={validation_cross_entropy:.5f} "
            f"validation_top={validation_top_move_agreement:.3%}"
        )
        if validation_cross_entropy < best_validation_cross_entropy:
            best_validation_cross_entropy = validation_cross_entropy
            model.save_weights(str(checkpoint_path))

    model.load_weights(str(checkpoint_path))
    test_cross_entropy, test_top_move_agreement = (
        evaluate_optimistic_policy_model(
            model,
            test_features,
            test_policy_targets,
            batch_size,
        )
    )
    print(
        f"optimistic_test_cross_entropy={test_cross_entropy:.5f} "
        f"optimistic_test_top={test_top_move_agreement:.3%} "
        f"bytes={checkpoint_path.stat().st_size:,}"
    )


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--data",
        action="append",
        required=True,
        type=Path,
    )
    argument_parser.add_argument("--test-data", required=True, type=Path)
    argument_parser.add_argument(
        "--initial-checkpoint",
        required=True,
        type=Path,
    )
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCH_COUNT,
    )
    argument_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    argument_parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    argument_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    train_optimistic_policy_head(
        arguments.data,
        arguments.test_data,
        arguments.initial_checkpoint,
        arguments.checkpoint,
        arguments.epochs,
        arguments.batch_size,
        arguments.learning_rate,
        arguments.seed,
    )


if __name__ == "__main__":
    main()
