import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from go_model.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCH_COUNT,
    DEFAULT_GAME_PAIR_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_RANDOM_SEED,
    SCORE_NORMALIZATION_POINTS,
    SPLIT_BUCKET_COUNT,
    TEST_BUCKET_INDEX,
    VALIDATION_BUCKET_INDEX,
)
from go_model.model import (
    MokaGlobalScoreNetwork,
    get_checkpoint_global_residual_block_interval,
)
from go_model.split import create_game_split_buckets


def load_score_datasets(
    dataset_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_batches: list[np.ndarray] = []
    game_id_batches: list[np.ndarray] = []
    score_batches: list[np.ndarray] = []
    game_id_offset = 0

    for dataset_path in dataset_paths:
        dataset = np.load(dataset_path)
        if "scores" not in dataset:
            raise ValueError(f"Score targets are missing from {dataset_path}.")
        features = dataset["features"].astype(np.float32)
        game_ids = dataset["game_ids"].astype(np.int64)
        scores = np.clip(
            dataset["scores"].astype(np.float32)
            / SCORE_NORMALIZATION_POINTS,
            -1,
            1,
        )
        feature_batches.append(features)
        game_id_batches.append(game_ids + game_id_offset)
        score_batches.append(scores)
        game_id_offset += int(np.max(game_ids)) + 1

    return (
        np.concatenate(feature_batches),
        np.concatenate(game_id_batches),
        np.concatenate(score_batches),
    )


def calculate_score_loss(
    model: MokaGlobalScoreNetwork,
    features: mx.array,
    score_targets: mx.array,
) -> mx.array:
    _, _, score_predictions = model.get_search_outputs(features)
    return mx.mean(mx.square(score_predictions - score_targets))


def evaluate_score_model(
    model: MokaGlobalScoreNetwork,
    features: np.ndarray,
    score_targets: np.ndarray,
    batch_size: int,
) -> tuple[float, float]:
    predictions: list[np.ndarray] = []

    for batch_start in range(0, len(features), batch_size):
        batch_end = batch_start + batch_size
        _, _, batch_predictions = model.get_search_outputs(
            mx.array(features[batch_start:batch_end], dtype=mx.float32)
        )
        mx.eval(batch_predictions)
        predictions.append(np.asarray(batch_predictions))

    score_predictions = np.concatenate(predictions)
    mean_absolute_error = float(
        np.mean(np.abs(score_predictions - score_targets))
    )
    sign_agreement = float(
        np.mean(np.sign(score_predictions) == np.sign(score_targets))
    )
    return mean_absolute_error, sign_agreement


def train_score_head(
    dataset_paths: list[Path],
    test_dataset_path: Path,
    initial_checkpoint_path: Path,
    checkpoint_path: Path,
    epoch_count: int,
    batch_size: int,
    learning_rate: float,
    random_seed: int,
    game_pair_size: int,
) -> None:
    features, game_ids, score_targets = load_score_datasets(dataset_paths)
    test_features, _, test_score_targets = load_score_datasets(
        [test_dataset_path]
    )
    buckets = create_game_split_buckets(game_ids, game_pair_size)
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
            "Global score training requires a global-residual checkpoint."
        )
    model = MokaGlobalScoreNetwork(global_residual_block_interval)
    model.load_weights(str(initial_checkpoint_path), strict=False)
    model.freeze()
    model.global_score_convolution.unfreeze()
    model.global_score_output.unfreeze()
    loss_and_grad = nn.value_and_grad(
        model,
        calculate_score_loss,
    )
    optimizer = optim.AdamW(learning_rate=learning_rate)
    best_validation_error = float("inf")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch_index in range(epoch_count):
        shuffled_indexes = random_generator.permutation(training_indexes)
        training_losses: list[float] = []

        for batch_start in range(0, len(shuffled_indexes), batch_size):
            batch_indexes = shuffled_indexes[
                batch_start : batch_start + batch_size
            ]
            loss, gradients = loss_and_grad(
                model,
                mx.array(features[batch_indexes], dtype=mx.float32),
                mx.array(score_targets[batch_indexes], dtype=mx.float32),
            )
            optimizer.update(model, gradients)
            mx.eval(
                model.parameters(),
                optimizer.state,
                loss,
            )
            training_losses.append(float(loss.item()))

        validation_error, validation_sign_agreement = (
            evaluate_score_model(
                model,
                features[validation_indexes],
                score_targets[validation_indexes],
                batch_size,
            )
        )
        print(
            f"epoch {epoch_index + 1:02d} "
            f"train={np.mean(training_losses):.4f} "
            f"validation_mae={validation_error:.4f} "
            f"validation_sign={validation_sign_agreement:.3f}"
        )
        if validation_error < best_validation_error:
            best_validation_error = validation_error
            model.save_weights(str(checkpoint_path))

    model.load_weights(str(checkpoint_path))
    test_error, test_sign_agreement = evaluate_score_model(
        model,
        test_features,
        test_score_targets,
        batch_size,
    )
    print(
        f"test_mae={test_error:.4f} "
        f"test_sign={test_sign_agreement:.3f}"
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
    argument_parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
    )
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
    argument_parser.add_argument(
        "--game-pair-size",
        type=int,
        default=DEFAULT_GAME_PAIR_SIZE,
    )
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    train_score_head(
        arguments.data,
        arguments.test_data,
        arguments.initial_checkpoint,
        arguments.checkpoint,
        arguments.epochs,
        arguments.batch_size,
        arguments.learning_rate,
        arguments.seed,
        arguments.game_pair_size,
    )


if __name__ == "__main__":
    main()
