import argparse
import glob
import math
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from go_model.config import BOARD_AREA, BOARD_SIZE, GLOBAL_RESIDUAL_BLOCK_INTERVAL
from go_model.model import (
    MokaGlobalResidualNetwork,
    create_moka_network_for_checkpoint,
)

SCORE_NORMALIZATION_POINTS = 20.0
VALIDATION_BUCKET = 0
TEST_BUCKET = 1


class ScaledTrainingNetwork(nn.Module):
    def __init__(self, network: MokaGlobalResidualNetwork) -> None:
        super().__init__()
        self.network = network
        trunk_channel_count = int(network.stem.weight.shape[0])
        self.ownership_head = nn.Conv2d(trunk_channel_count, 1, kernel_size=1)
        self.score_head = nn.Linear(trunk_channel_count * 2, 1)

    def __call__(self, inputs: mx.array):
        network = self.network
        trunk_values = network.get_trunk_values(inputs)
        policy_values = nn.relu(network.policy_convolution(trunk_values))
        policy_logits = network.policy_linear(
            mx.flatten(policy_values, start_axis=1)
        )
        value_values = nn.relu(network.value_convolution(trunk_values))
        value_hidden = nn.relu(
            network.value_hidden(mx.flatten(value_values, start_axis=1))
        )
        value = mx.tanh(network.value_output(value_hidden)).squeeze(-1)
        ownership = mx.tanh(self.ownership_head(trunk_values)).squeeze(-1)
        pooled_trunk = mx.concatenate(
            [mx.mean(trunk_values, axis=(1, 2)), mx.max(trunk_values, axis=(1, 2))],
            axis=1,
        )
        score = self.score_head(pooled_trunk).squeeze(-1)
        return policy_logits, value, ownership, score


def load_corpus(patterns: list[str]) -> dict[str, np.ndarray]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError(f"No shards match {patterns}.")
    shards = [np.load(path) for path in paths]
    keys = [
        "features",
        "search_policies",
        "raw_policies",
        "values",
        "outcomes",
        "scores",
        "ownerships",
        "game_ids",
    ]
    corpus = {key: np.concatenate([shard[key] for shard in shards]) for key in keys}
    print(f"loaded {len(paths)} shards positions={len(corpus['features']):,}")
    return corpus


def create_targets(
    corpus: dict[str, np.ndarray],
    search_policy_weight: float,
    outcome_value_weight: float,
) -> dict[str, np.ndarray]:
    policies = (
        search_policy_weight * corpus["search_policies"].astype(np.float32)
        + (1 - search_policy_weight) * corpus["raw_policies"].astype(np.float32)
    )
    policies /= np.maximum(policies.sum(axis=1, keepdims=True), 1e-8)
    values = (
        outcome_value_weight * corpus["outcomes"].astype(np.float32)
        + (1 - outcome_value_weight) * corpus["values"].astype(np.float32)
    )
    return {
        "features": corpus["features"],
        "policies": policies.astype(np.float16),
        "values": values.astype(np.float32),
        "ownerships": corpus["ownerships"],
        "scores": np.clip(
            corpus["scores"].astype(np.float32) / SCORE_NORMALIZATION_POINTS,
            -3,
            3,
        ),
    }


def apply_symmetry(
    batch: dict[str, np.ndarray],
    rotation_count: int,
    should_flip: bool,
) -> dict[str, np.ndarray]:
    def transform(values: np.ndarray) -> np.ndarray:
        values = np.rot90(values, rotation_count, axes=(1, 2))
        return np.flip(values, axis=2) if should_flip else values

    board_policies = transform(
        batch["policies"][:, :BOARD_AREA].reshape(-1, BOARD_SIZE, BOARD_SIZE)
    ).reshape(-1, BOARD_AREA)
    return {
        "features": np.ascontiguousarray(transform(batch["features"])),
        "policies": np.concatenate(
            [board_policies, batch["policies"][:, BOARD_AREA:]], axis=1
        ),
        "values": batch["values"],
        "ownerships": np.ascontiguousarray(transform(batch["ownerships"])),
        "scores": batch["scores"],
    }


def compute_losses(
    model: ScaledTrainingNetwork,
    features: mx.array,
    policies: mx.array,
    values: mx.array,
    ownerships: mx.array,
    scores: mx.array,
):
    policy_logits, predicted_values, predicted_ownerships, predicted_scores = model(
        features
    )
    policy_loss = mx.mean(
        -mx.sum(policies * nn.log_softmax(policy_logits, axis=1), axis=1)
    )
    value_loss = mx.mean(mx.square(predicted_values - values))
    ownership_loss = mx.mean(mx.square(predicted_ownerships - ownerships))
    score_loss = mx.mean(nn.losses.huber_loss(predicted_scores, scores))
    return policy_loss, value_loss, ownership_loss, score_loss, policy_logits, predicted_values


def evaluate(
    model: ScaledTrainingNetwork,
    targets: dict[str, np.ndarray],
    indexes: np.ndarray,
    batch_size: int = 2048,
) -> dict[str, float]:
    totals = np.zeros(5)
    for start in range(0, len(indexes), batch_size):
        batch_indexes = indexes[start : start + batch_size]
        batch = {key: values[batch_indexes] for key, values in targets.items()}
        arrays = [
            mx.array(batch[key].astype(np.float32))
            for key in ["features", "policies", "values", "ownerships", "scores"]
        ]
        policy_loss, value_loss, ownership_loss, _, logits, predicted_values = (
            compute_losses(model, *arrays)
        )
        top_move_agreement = mx.mean(
            mx.argmax(logits, axis=1) == mx.argmax(arrays[1], axis=1)
        )
        value_error = mx.mean(mx.abs(predicted_values - arrays[2]))
        mx.eval(policy_loss, value_loss, ownership_loss, top_move_agreement, value_error)
        count = len(batch_indexes)
        totals += count * np.array(
            [
                policy_loss.item(),
                value_loss.item(),
                ownership_loss.item(),
                top_move_agreement.item(),
                value_error.item(),
            ]
        )
    totals /= len(indexes)
    return dict(
        zip(
            ["policy_loss", "value_mse", "ownership_mse", "top_move", "value_mae"],
            totals.tolist(),
        )
    )


def train(arguments: argparse.Namespace) -> None:
    mx.random.seed(arguments.seed)
    random_generator = np.random.default_rng(arguments.seed)
    corpus = load_corpus(arguments.data)
    targets = create_targets(
        corpus, arguments.search_policy_weight, arguments.outcome_value_weight
    )
    buckets = corpus["game_ids"] % 10
    train_indexes = np.flatnonzero((buckets != VALIDATION_BUCKET) & (buckets != TEST_BUCKET))
    validation_indexes = np.flatnonzero(buckets == VALIDATION_BUCKET)
    test_indexes = np.flatnonzero(buckets == TEST_BUCKET)
    print(
        f"train={len(train_indexes):,} validation={len(validation_indexes):,} "
        f"test={len(test_indexes):,}"
    )

    if arguments.initial_checkpoint:
        network = create_moka_network_for_checkpoint(str(arguments.initial_checkpoint))
        network.load_weights(str(arguments.initial_checkpoint))
    else:
        network = MokaGlobalResidualNetwork(
            GLOBAL_RESIDUAL_BLOCK_INTERVAL,
            arguments.trunk_channels,
            arguments.bottleneck_channels,
            arguments.blocks,
        )
    model = ScaledTrainingNetwork(network)
    mx.eval(model.parameters())
    parameter_count = sum(
        values.size for _, values in tree_flatten(network.parameters())
    )
    print(f"network parameters={parameter_count:,}")

    steps_per_epoch = len(train_indexes) // arguments.batch_size
    total_steps = steps_per_epoch * arguments.epochs
    warmup_steps = min(1000, total_steps // 20)
    schedule = optim.join_schedules(
        [
            optim.linear_schedule(1e-6, arguments.learning_rate, warmup_steps),
            optim.cosine_decay(
                arguments.learning_rate,
                max(total_steps - warmup_steps, 1),
                arguments.learning_rate * 0.02,
            ),
        ],
        [warmup_steps],
    )
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=arguments.weight_decay)

    def loss_function(model, features, policies, values, ownerships, scores):
        policy_loss, value_loss, ownership_loss, score_loss, _, _ = compute_losses(
            model, features, policies, values, ownerships, scores
        )
        return (
            policy_loss
            + arguments.value_weight * value_loss
            + arguments.ownership_weight * ownership_loss
            + arguments.score_weight * score_loss
        )

    loss_and_gradient = nn.value_and_grad(model, loss_function)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(features, policies, values, ownerships, scores):
        loss, gradients = loss_and_gradient(
            model, features, policies, values, ownerships, scores
        )
        gradients, _ = optim.clip_grad_norm(gradients, 5.0)
        optimizer.update(model, gradients)
        return loss

    best_validation_loss = math.inf
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(arguments.epochs):
        start_time = time.time()
        permutation = random_generator.permutation(train_indexes)
        running_loss = 0.0
        for step in range(steps_per_epoch):
            batch_indexes = np.sort(
                permutation[step * arguments.batch_size : (step + 1) * arguments.batch_size]
            )
            batch = apply_symmetry(
                {key: values[batch_indexes] for key, values in targets.items()},
                int(random_generator.integers(4)),
                bool(random_generator.integers(2)),
            )
            loss = train_step(
                *[
                    mx.array(batch[key].astype(np.float32))
                    for key in ["features", "policies", "values", "ownerships", "scores"]
                ]
            )
            mx.eval(state)
            running_loss += loss.item()
            if (step + 1) % 500 == 0:
                print(
                    f"epoch={epoch + 1} step={step + 1}/{steps_per_epoch} "
                    f"loss={running_loss / 500:.4f}",
                    flush=True,
                )
                running_loss = 0.0
        metrics = evaluate(model, targets, validation_indexes)
        validation_loss = metrics["policy_loss"] + arguments.value_weight * metrics["value_mse"]
        print(
            f"epoch={epoch + 1} seconds={time.time() - start_time:.0f} "
            + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()),
            flush=True,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            network.save_weights(str(arguments.checkpoint))
            print(f"saved {arguments.checkpoint}", flush=True)

    network.load_weights(str(arguments.checkpoint))
    test_metrics = evaluate(model, targets, test_indexes)
    print("test " + " ".join(f"{key}={value:.4f}" for key, value in test_metrics.items()))


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data", nargs="+", default=["data/selfplay/*.npz"])
    argument_parser.add_argument("--checkpoint", type=Path, required=True)
    argument_parser.add_argument("--initial-checkpoint", type=Path)
    argument_parser.add_argument("--trunk-channels", type=int, default=48)
    argument_parser.add_argument("--bottleneck-channels", type=int, default=24)
    argument_parser.add_argument("--blocks", type=int, default=16)
    argument_parser.add_argument("--epochs", type=int, default=8)
    argument_parser.add_argument("--batch-size", type=int, default=512)
    argument_parser.add_argument("--learning-rate", type=float, default=2e-3)
    argument_parser.add_argument("--weight-decay", type=float, default=1e-4)
    argument_parser.add_argument("--search-policy-weight", type=float, default=0.5)
    argument_parser.add_argument("--outcome-value-weight", type=float, default=0.3)
    argument_parser.add_argument("--value-weight", type=float, default=1.0)
    argument_parser.add_argument("--ownership-weight", type=float, default=0.5)
    argument_parser.add_argument("--score-weight", type=float, default=0.1)
    argument_parser.add_argument("--seed", type=int, default=0)
    return argument_parser


def main() -> None:
    train(create_argument_parser().parse_args())


if __name__ == "__main__":
    main()
