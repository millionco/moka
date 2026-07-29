import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from go_model.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_RANDOM_SEED,
    DISTILLED_GRPO_POLICY_WEIGHT,
    GRPO_CONTRIBUTION_WEIGHT,
    GRPO_FAILED_DISTILLATION_MULTIPLIER,
    GRPO_KL_WEIGHT,
    GRPO_ROUTED_DISTILLATION_MULTIPLIER,
    GRPO_SUCCESSFUL_DISTILLATION_MULTIPLIER,
    GRPO_TEACHER_ACTION_ADVANTAGE_CLIP,
    GRPO_TEACHER_ACTION_ADVANTAGE_WEIGHT,
    N_DISTILL_CORRECTION_WEIGHT,
    ON_POLICY_MOKA_TEMPERATURE,
    PPO_CLIP_RANGE,
    PPO_ENTROPY_WEIGHT,
    PPO_ILLEGAL_MOVE_LOGIT,
    PPO_PROBABILITY_EPSILON,
    SPLIT_BUCKET_COUNT,
    TEST_BUCKET_INDEX,
    VALIDATION_BUCKET_INDEX,
)
from go_model.model import (
    MokaNetwork,
    create_moka_network_for_checkpoint,
)


def calculate_policy_optimization_loss(
    model: MokaNetwork,
    features: mx.array,
    actions: mx.array,
    old_log_probabilities: mx.array,
    advantages: mx.array,
    legal_masks: mx.array,
    kl_weight: float,
    teacher_policies: mx.array,
    distillation_weight: float,
) -> mx.array:
    policy_logits, _ = model(features)
    tempered_logits = policy_logits / ON_POLICY_MOKA_TEMPERATURE
    masked_logits = mx.where(
        legal_masks,
        tempered_logits,
        PPO_ILLEGAL_MOVE_LOGIT,
    )
    log_probabilities = masked_logits - mx.logsumexp(
        masked_logits,
        axis=1,
        keepdims=True,
    )
    selected_log_probabilities = mx.take_along_axis(
        log_probabilities,
        actions[:, None],
        axis=1,
    ).squeeze(1)
    probability_ratios = mx.exp(
        selected_log_probabilities - old_log_probabilities
    )
    clipped_ratios = mx.clip(
        probability_ratios,
        1 - PPO_CLIP_RANGE,
        1 + PPO_CLIP_RANGE,
    )
    policy_loss = -mx.mean(
        mx.minimum(
            probability_ratios * advantages,
            clipped_ratios * advantages,
        )
    )
    probabilities = mx.exp(log_probabilities)
    entropy = -mx.mean(mx.sum(probabilities * log_probabilities, axis=1))
    log_probability_difference = (
        old_log_probabilities - selected_log_probabilities
    )
    sampled_kl = (
        mx.exp(log_probability_difference)
        - log_probability_difference
        - 1
    )
    distillation_loss = -mx.mean(
        mx.sum(teacher_policies * log_probabilities, axis=1)
    )
    return (
        policy_loss
        - PPO_ENTROPY_WEIGHT * entropy
        + kl_weight * mx.mean(sampled_kl)
        + distillation_weight * distillation_loss
    )


def evaluate_policy_loss(
    model: MokaNetwork,
    features: np.ndarray,
    actions: np.ndarray,
    old_log_probabilities: np.ndarray,
    advantages: np.ndarray,
    legal_masks: np.ndarray,
    batch_size: int,
    kl_weight: float,
    teacher_policies: np.ndarray,
    distillation_weight: float,
) -> float:
    losses: list[float] = []

    for batch_start in range(0, len(features), batch_size):
        batch_end = batch_start + batch_size
        loss = calculate_policy_optimization_loss(
            model,
            mx.array(features[batch_start:batch_end], dtype=mx.float32),
            mx.array(actions[batch_start:batch_end], dtype=mx.int32),
            mx.array(
                old_log_probabilities[batch_start:batch_end],
                dtype=mx.float32,
            ),
            mx.array(advantages[batch_start:batch_end], dtype=mx.float32),
            mx.array(legal_masks[batch_start:batch_end], dtype=mx.bool_),
            kl_weight,
            mx.array(
                teacher_policies[batch_start:batch_end],
                dtype=mx.float32,
            ),
            distillation_weight,
        )
        mx.eval(loss)
        losses.append(float(loss.item()))

    return float(np.mean(losses))


def calculate_group_relative_advantages(
    game_ids: np.ndarray,
    rewards: np.ndarray,
    group_size: int,
) -> np.ndarray:
    unique_game_ids, first_game_indexes = np.unique(
        game_ids,
        return_index=True,
    )
    game_rewards = rewards[first_game_indexes]
    group_ids = (
        unique_game_ids // (group_size * 2) * 2
        + unique_game_ids % 2
    )
    game_advantages: dict[int, float] = {}

    for group_id in np.unique(group_ids):
        group_mask = group_ids == group_id
        group_rewards = game_rewards[group_mask]
        normalized_rewards = (
            group_rewards - np.mean(group_rewards)
        ) / max(float(np.std(group_rewards)), PPO_PROBABILITY_EPSILON)
        for game_id, advantage in zip(
            unique_game_ids[group_mask],
            normalized_rewards,
            strict=True,
        ):
            game_advantages[int(game_id)] = float(advantage)

    return np.asarray(
        [game_advantages[int(game_id)] for game_id in game_ids],
        dtype=np.float32,
    )


def calculate_n_distill_advantages(
    model: MokaNetwork,
    features: np.ndarray,
    game_ids: np.ndarray,
    legal_masks: np.ndarray,
    moka_action_masks: np.ndarray,
    teacher_policies: np.ndarray,
    group_size: int,
    batch_size: int,
) -> np.ndarray:
    cross_entropies: list[np.ndarray] = []

    for batch_start in range(0, len(features), batch_size):
        batch_end = batch_start + batch_size
        policy_logits, _ = model(
            mx.array(features[batch_start:batch_end], dtype=mx.float32)
        )
        tempered_logits = policy_logits / ON_POLICY_MOKA_TEMPERATURE
        masked_logits = mx.where(
            mx.array(legal_masks[batch_start:batch_end], dtype=mx.bool_),
            tempered_logits,
            PPO_ILLEGAL_MOVE_LOGIT,
        )
        log_probabilities = masked_logits - mx.logsumexp(
            masked_logits,
            axis=1,
            keepdims=True,
        )
        batch_cross_entropies = -mx.sum(
            mx.array(
                teacher_policies[batch_start:batch_end],
                dtype=mx.float32,
            )
            * log_probabilities,
            axis=1,
        )
        mx.eval(batch_cross_entropies)
        cross_entropies.append(np.asarray(batch_cross_entropies))

    position_cross_entropies = np.concatenate(cross_entropies)
    game_corrections: dict[int, float] = {}

    for game_id in np.unique(game_ids):
        game_indexes = np.flatnonzero(game_ids == game_id)
        game_action_indexes = game_indexes[
            moka_action_masks[game_indexes]
        ]
        next_position_indexes = [
            int(game_indexes[np.searchsorted(game_indexes, action_index) + 1])
            for action_index in game_action_indexes
            if np.searchsorted(game_indexes, action_index) + 1 < len(game_indexes)
        ]
        game_corrections[int(game_id)] = -float(
            np.sum(position_cross_entropies[next_position_indexes])
        )

    correction_rewards = np.asarray(
        [game_corrections[int(game_id)] for game_id in game_ids],
        dtype=np.float32,
    )
    return calculate_group_relative_advantages(
        game_ids,
        correction_rewards,
        group_size,
    )


def calculate_contribution_weighted_advantages(
    advantages: np.ndarray,
    action_advantages: np.ndarray,
    contribution_weight: float,
) -> np.ndarray:
    clipped_action_advantages = np.clip(
        action_advantages,
        -GRPO_TEACHER_ACTION_ADVANTAGE_CLIP,
        GRPO_TEACHER_ACTION_ADVANTAGE_CLIP,
    )
    contribution_multipliers = (
        1
        + contribution_weight
        * np.sign(advantages)
        * clipped_action_advantages
    )
    return advantages * contribution_multipliers


def calculate_sample_routed_teacher_policies(
    teacher_policies: np.ndarray,
    advantages: np.ndarray,
) -> np.ndarray:
    distillation_multipliers = np.where(
        advantages < 0,
        GRPO_FAILED_DISTILLATION_MULTIPLIER,
        GRPO_SUCCESSFUL_DISTILLATION_MULTIPLIER,
    )
    return teacher_policies * distillation_multipliers[:, None]


def calculate_strict_sample_routing(
    teacher_policies: np.ndarray,
    advantages: np.ndarray,
    successful_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    failed_masks = ~successful_masks
    routed_advantages = np.where(failed_masks, 0, advantages)
    routed_teacher_policies = (
        teacher_policies
        * failed_masks[:, None]
        * GRPO_ROUTED_DISTILLATION_MULTIPLIER
    )
    return routed_advantages, routed_teacher_policies


def optimize_policy(
    dataset_path: Path,
    initial_checkpoint_path: Path,
    checkpoint_path: Path,
    epoch_count: int,
    batch_size: int,
    learning_rate: float,
    random_seed: int,
    grpo_group_size: int,
    distillation_weight: float,
    teacher_action_advantage_weight: float,
    contribution_weight: float,
    use_sample_routing: bool,
    use_strict_sample_routing: bool,
    use_wide_network: bool,
    use_n_distill: bool,
    freeze_trunk: bool,
) -> None:
    dataset = np.load(dataset_path)
    all_features = dataset["features"].astype(np.float32)
    all_game_ids = dataset["game_ids"]
    all_legal_masks = dataset["legal_masks"]
    all_moka_action_masks = dataset["moka_action_masks"]
    moka_action_indexes = np.flatnonzero(all_moka_action_masks)
    features = all_features[moka_action_indexes]
    actions = dataset["actions"][moka_action_indexes].astype(np.int32)
    game_ids = all_game_ids[moka_action_indexes]
    old_log_probabilities = dataset["old_log_probabilities"][
        moka_action_indexes
    ].astype(np.float32)
    legal_masks = all_legal_masks[moka_action_indexes]
    rewards = dataset["values"][moka_action_indexes].astype(np.float32)
    if distillation_weight > 0 and "teacher_policies" not in dataset:
        raise ValueError(
            "Distillation requires teacher_policies in the outcome dataset."
        )
    teacher_policies = (
        dataset["teacher_policies"][moka_action_indexes].astype(np.float32)
        if "teacher_policies" in dataset
        else np.zeros(
            (len(moka_action_indexes), legal_masks.shape[1]),
            dtype=np.float32,
        )
    )
    model = create_moka_network_for_checkpoint(
        str(initial_checkpoint_path),
        not use_wide_network,
        False,
        False,
        False,
        use_wide_network,
    )
    model.load_weights(str(initial_checkpoint_path))
    model.eval()
    if grpo_group_size > 0:
        advantages = calculate_group_relative_advantages(
            game_ids,
            rewards,
            grpo_group_size,
        )
    else:
        advantages = (
            rewards
            - dataset["baselines"][moka_action_indexes].astype(np.float32)
        )
        advantages = (advantages - np.mean(advantages)) / max(
            float(np.std(advantages)),
            PPO_PROBABILITY_EPSILON,
        )
    if use_n_distill:
        if grpo_group_size == 0 or "teacher_policies" not in dataset:
            raise ValueError("N-distill requires grouped teacher-policy trajectories.")
        correction_advantages = calculate_n_distill_advantages(
            model,
            all_features,
            all_game_ids,
            all_legal_masks,
            all_moka_action_masks,
            dataset["teacher_policies"].astype(np.float32),
            grpo_group_size,
            batch_size,
        )
        advantages += (
            N_DISTILL_CORRECTION_WEIGHT
            * correction_advantages[moka_action_indexes]
        )
    if use_sample_routing or use_strict_sample_routing:
        if distillation_weight == 0:
            raise ValueError("Sample routing requires policy distillation.")
        if use_strict_sample_routing:
            advantages, teacher_policies = calculate_strict_sample_routing(
                teacher_policies,
                advantages,
                rewards > 0,
            )
        else:
            teacher_policies = calculate_sample_routed_teacher_policies(
                teacher_policies,
                advantages,
            )
    if teacher_action_advantage_weight > 0 or contribution_weight > 0:
        if "teacher_action_advantages" not in dataset:
            raise ValueError(
                "Teacher action advantages are missing from the outcome dataset."
            )
        action_advantages = dataset["teacher_action_advantages"][
            moka_action_indexes
        ].astype(np.float32)
        if teacher_action_advantage_weight > 0:
            advantages += teacher_action_advantage_weight * np.clip(
                action_advantages,
                -GRPO_TEACHER_ACTION_ADVANTAGE_CLIP,
                GRPO_TEACHER_ACTION_ADVANTAGE_CLIP,
            )
        if contribution_weight > 0:
            advantages = calculate_contribution_weighted_advantages(
                advantages,
                action_advantages,
                contribution_weight,
            )
    split_ids = (
        game_ids // (grpo_group_size * 2) * 2 + game_ids % 2
        if grpo_group_size > 0
        else game_ids
    )
    game_buckets = split_ids % SPLIT_BUCKET_COUNT
    validation_indexes = np.flatnonzero(game_buckets == VALIDATION_BUCKET_INDEX)
    test_indexes = np.flatnonzero(game_buckets == TEST_BUCKET_INDEX)
    training_indexes = np.flatnonzero(
        (game_buckets != VALIDATION_BUCKET_INDEX)
        & (game_buckets != TEST_BUCKET_INDEX)
    )
    random_generator = np.random.default_rng(random_seed)
    mx.random.seed(random_seed)
    if grpo_group_size == 0 or freeze_trunk:
        model.freeze()
        model.policy_convolution.unfreeze()
        model.policy_linear.unfreeze()
    optimizer = optim.AdamW(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(
        model,
        calculate_policy_optimization_loss,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    for epoch_index in range(epoch_count):
        shuffled_indexes = random_generator.permutation(training_indexes)
        training_losses: list[float] = []

        for batch_start in range(0, len(shuffled_indexes), batch_size):
            batch_indexes = shuffled_indexes[batch_start : batch_start + batch_size]
            loss, gradients = loss_and_grad(
                model,
                mx.array(features[batch_indexes], dtype=mx.float32),
                mx.array(actions[batch_indexes], dtype=mx.int32),
                mx.array(old_log_probabilities[batch_indexes], dtype=mx.float32),
                mx.array(advantages[batch_indexes], dtype=mx.float32),
                mx.array(legal_masks[batch_indexes], dtype=mx.bool_),
                GRPO_KL_WEIGHT if grpo_group_size > 0 else 0,
                mx.array(
                    teacher_policies[batch_indexes],
                    dtype=mx.float32,
                ),
                distillation_weight,
            )
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            training_losses.append(float(loss.item()))

        validation_loss = evaluate_policy_loss(
            model,
            features[validation_indexes],
            actions[validation_indexes],
            old_log_probabilities[validation_indexes],
            advantages[validation_indexes],
            legal_masks[validation_indexes],
            batch_size,
            GRPO_KL_WEIGHT if grpo_group_size > 0 else 0,
            teacher_policies[validation_indexes],
            distillation_weight,
        )
        print(
            f"epoch {epoch_index + 1:02d} "
            f"train={np.mean(training_losses):.4f} "
            f"validation={validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            model.save_weights(str(checkpoint_path))

    model.load_weights(str(checkpoint_path))
    test_loss = evaluate_policy_loss(
        model,
        features[test_indexes],
        actions[test_indexes],
        old_log_probabilities[test_indexes],
        advantages[test_indexes],
        legal_masks[test_indexes],
        batch_size,
        GRPO_KL_WEIGHT if grpo_group_size > 0 else 0,
        teacher_policies[test_indexes],
        distillation_weight,
    )
    print(f"test={test_loss:.4f}")


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data", required=True, type=Path)
    argument_parser.add_argument("--initial-checkpoint", required=True, type=Path)
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument("--epochs", type=int, default=1)
    argument_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    argument_parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    argument_parser.add_argument(
        "--grpo-group-size",
        type=int,
        default=0,
    )
    argument_parser.add_argument(
        "--distillation-weight",
        type=float,
        default=0,
    )
    argument_parser.add_argument(
        "--distilled-grpo",
        action="store_true",
    )
    argument_parser.add_argument(
        "--teacher-action-advantage-weight",
        type=float,
        default=0,
    )
    argument_parser.add_argument(
        "--teacher-action-advantages",
        action="store_true",
    )
    argument_parser.add_argument(
        "--contribution-weight",
        type=float,
        default=0,
    )
    argument_parser.add_argument(
        "--contribution-weighted-grpo",
        action="store_true",
    )
    argument_parser.add_argument(
        "--sample-routed-grpo",
        action="store_true",
    )
    argument_parser.add_argument(
        "--strict-sample-routed-grpo",
        action="store_true",
    )
    argument_parser.add_argument("--wide", action="store_true")
    argument_parser.add_argument("--n-distill", action="store_true")
    argument_parser.add_argument("--freeze-trunk", action="store_true")
    argument_parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    optimize_policy(
        arguments.data,
        arguments.initial_checkpoint,
        arguments.checkpoint,
        arguments.epochs,
        arguments.batch_size,
        arguments.learning_rate,
        arguments.seed,
        arguments.grpo_group_size,
        (
            DISTILLED_GRPO_POLICY_WEIGHT
            if arguments.distilled_grpo
            else arguments.distillation_weight
        ),
        (
            GRPO_TEACHER_ACTION_ADVANTAGE_WEIGHT
            if arguments.teacher_action_advantages
            else arguments.teacher_action_advantage_weight
        ),
        (
            GRPO_CONTRIBUTION_WEIGHT
            if arguments.contribution_weighted_grpo
            else arguments.contribution_weight
        ),
        arguments.sample_routed_grpo,
        arguments.strict_sample_routed_grpo,
        arguments.wide,
        arguments.n_distill,
        arguments.freeze_trunk,
    )


if __name__ == "__main__":
    main()
