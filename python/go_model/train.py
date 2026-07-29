import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from go_model.config import (
    ATTENTION_TRANSFER_LOSS_WEIGHT,
    BOARD_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCH_COUNT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_RANDOM_SEED,
    GLOBAL_RESIDUAL_BLOCK_INTERVAL,
    LISTWISE_POLICY_RANK_COUNT,
    OWNERSHIP_LOSS_WEIGHT,
    PPO_ILLEGAL_MOVE_LOGIT,
    Q_RANK_MINIMUM_ADVANTAGE,
    Q_RANK_MINIMUM_VISIT_COUNT,
    Q_VALUE_LOSS_WEIGHT,
    REVERSE_KL_EPSILON,
    SCORE_LOSS_WEIGHT,
    SCORE_NORMALIZATION_POINTS,
    SHORT_VALUE_TARGET_WEIGHT,
    SOFT_POLICY_LOSS_WEIGHT,
    SOFT_POLICY_TEMPERATURE,
    SPLIT_BUCKET_COUNT,
    SYMMETRY_FLIP_OPTION_COUNT,
    SYMMETRY_ROTATION_COUNT,
    TEST_BUCKET_INDEX,
    TRAIN_SELECTION_LOSS,
    TRAIN_SELECTION_METRICS,
    VALIDATION_BUCKET_INDEX,
    VALUE_LOSS_WEIGHT,
)
from go_model.model import (
    GlobalNestedBottleneckBlock,
    MokaAttentionAuxiliaryNetwork,
    MokaAuxiliaryNetwork,
    MokaNestedNetwork,
    MokaNetwork,
    MokaQAuxiliaryNetwork,
    MokaSoftPolicyNetwork,
    create_moka_network,
    get_checkpoint_global_residual_block_interval,
)
from go_model.quantization import fake_quantize_int8_parameters
from go_model.symmetry import apply_batch_board_symmetry, apply_batch_spatial_symmetry


def normalize_sample_weights(sample_weights: mx.array) -> mx.array:
    return sample_weights / mx.maximum(
        mx.mean(sample_weights),
        REVERSE_KL_EPSILON,
    )


def normalize_score_targets(scores: np.ndarray) -> np.ndarray:
    return np.clip(
        scores / SCORE_NORMALIZATION_POINTS,
        -1,
        1,
    )


def calculate_listwise_policy_loss(
    policy_logits: mx.array,
    policy_targets: mx.array,
    sample_weights: mx.array,
) -> mx.array:
    teacher_rankings = mx.argsort(-policy_targets, axis=1)[
        :, :LISTWISE_POLICY_RANK_COUNT
    ]
    remaining_logits = policy_logits
    sample_losses = mx.zeros(policy_logits.shape[0])
    move_indexes = mx.arange(policy_logits.shape[1])[None, :]

    for rank_index in range(LISTWISE_POLICY_RANK_COUNT):
        ranked_moves = teacher_rankings[:, rank_index]
        selected_logits = mx.take_along_axis(
            remaining_logits,
            ranked_moves[:, None],
            axis=1,
        ).squeeze(1)
        sample_losses += mx.logsumexp(remaining_logits, axis=1) - selected_logits
        remaining_logits = mx.where(
            move_indexes == ranked_moves[:, None],
            PPO_ILLEGAL_MOVE_LOGIT,
            remaining_logits,
        )

    normalized_weights = normalize_sample_weights(sample_weights)
    return mx.mean(sample_losses * normalized_weights) / LISTWISE_POLICY_RANK_COUNT


def calculate_q_rank_policy_loss(
    policy_logits: mx.array,
    policy_targets: mx.array,
    q_value_targets: mx.array,
    q_value_weights: mx.array,
    sample_weights: mx.array,
) -> mx.array:
    teacher_moves = mx.argmax(policy_targets, axis=1)
    teacher_q_values = mx.take_along_axis(
        q_value_targets,
        teacher_moves[:, None],
        axis=1,
    ).squeeze(1)
    teacher_q_weights = mx.take_along_axis(
        q_value_weights,
        teacher_moves[:, None],
        axis=1,
    ).squeeze(1)
    teacher_logits = mx.take_along_axis(
        policy_logits,
        teacher_moves[:, None],
        axis=1,
    ).squeeze(1)
    advantages = teacher_q_values[:, None] - q_value_targets
    pair_weights = mx.where(
        (teacher_q_weights[:, None] >= Q_RANK_MINIMUM_VISIT_COUNT)
        & (q_value_weights >= Q_RANK_MINIMUM_VISIT_COUNT)
        & (advantages >= Q_RANK_MINIMUM_ADVANTAGE),
        advantages,
        0,
    )
    pair_weight_sums = mx.sum(pair_weights, axis=1)
    pair_losses = mx.logaddexp(
        mx.zeros_like(policy_logits),
        policy_logits - teacher_logits[:, None],
    )
    sample_losses = mx.sum(pair_losses * pair_weights, axis=1) / mx.maximum(
        pair_weight_sums,
        REVERSE_KL_EPSILON,
    )
    active_samples = (pair_weight_sums > 0).astype(mx.float32)
    normalized_weights = normalize_sample_weights(sample_weights)
    active_weights = active_samples * normalized_weights
    return mx.sum(sample_losses * active_weights) / mx.maximum(
        mx.sum(active_weights),
        REVERSE_KL_EPSILON,
    )


def calculate_loss(
    model: MokaNetwork,
    features: mx.array,
    policy_targets: mx.array,
    value_targets: mx.array,
    sample_weights: mx.array,
    ownership_targets: mx.array,
    ownership_weights: mx.array,
    score_targets: mx.array,
    score_weights: mx.array,
    q_value_targets: mx.array,
    q_value_weights: mx.array,
    attention_targets: mx.array,
    attention_weights: mx.array,
    ownership_loss_weight: float,
    reverse_kl_weight: float,
    soft_policy_loss_weight: float,
    use_auxiliary_network: bool,
    use_q_auxiliary: bool,
    use_attention_auxiliary: bool,
    listwise_policy_weight: float,
    q_rank_policy_weight: float,
    reference_policy_logits: mx.array,
    policy_preservation_weight: float,
) -> tuple[
    mx.array,
    tuple[
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
    ],
]:
    training_outputs = model.get_training_outputs(features)
    policy_logits, values, ownership_predictions = training_outputs[:3]
    soft_policy_logits = (
        training_outputs[3]
        if soft_policy_loss_weight > 0
        else policy_logits
    )
    score_predictions = (
        training_outputs[3]
        if use_auxiliary_network
        else mx.zeros_like(value_targets)
    )
    q_value_predictions = (
        training_outputs[3]
        if use_q_auxiliary
        else mx.zeros_like(q_value_targets)
    )
    attention_predictions = (
        training_outputs[3]
        if use_attention_auxiliary
        else mx.zeros_like(attention_targets)
    )
    log_probabilities = policy_logits - mx.logsumexp(
        policy_logits,
        axis=1,
        keepdims=True,
    )
    normalized_weights = normalize_sample_weights(sample_weights)
    forward_policy_loss = -mx.mean(
        mx.sum(policy_targets * log_probabilities, axis=1) * normalized_weights
    )
    probabilities = mx.exp(log_probabilities)
    teacher_log_probabilities = mx.log(
        mx.maximum(policy_targets, REVERSE_KL_EPSILON)
    )
    reverse_policy_loss = mx.mean(
        mx.sum(
            probabilities * (log_probabilities - teacher_log_probabilities),
            axis=1,
        )
        * normalized_weights
    )
    policy_loss = (
        (1 - reverse_kl_weight) * forward_policy_loss
        + reverse_kl_weight * reverse_policy_loss
    )
    listwise_policy_loss = calculate_listwise_policy_loss(
        policy_logits,
        policy_targets,
        sample_weights,
    )
    q_rank_policy_loss = calculate_q_rank_policy_loss(
        policy_logits,
        policy_targets,
        q_value_targets,
        q_value_weights,
        sample_weights,
    )
    policy_preservation_loss = mx.mean(
        mx.square(policy_logits - reference_policy_logits)
    )
    soft_policy_targets = mx.power(
        policy_targets,
        1 / SOFT_POLICY_TEMPERATURE,
    )
    soft_policy_targets /= mx.sum(
        soft_policy_targets,
        axis=1,
        keepdims=True,
    )
    soft_log_probabilities = soft_policy_logits - mx.logsumexp(
        soft_policy_logits,
        axis=1,
        keepdims=True,
    )
    soft_policy_loss = -mx.mean(
        mx.sum(
            soft_policy_targets * soft_log_probabilities,
            axis=1,
        )
        * normalized_weights
    )
    value_loss = mx.mean(mx.square(values - value_targets) * normalized_weights)
    ownership_sample_losses = mx.mean(
        mx.square(ownership_predictions - ownership_targets),
        axis=(1, 2),
    )
    ownership_loss = mx.sum(ownership_sample_losses * ownership_weights) / mx.maximum(
        mx.sum(ownership_weights),
        1,
    )
    score_sample_losses = mx.square(score_predictions - score_targets)
    score_loss = mx.sum(score_sample_losses * score_weights) / mx.maximum(
        mx.sum(score_weights),
        1,
    )
    q_value_loss = mx.sum(
        mx.square(q_value_predictions - q_value_targets) * q_value_weights
    ) / mx.maximum(
        mx.sum(q_value_weights),
        1,
    )
    attention_sample_losses = mx.sum(
        mx.square(attention_predictions - attention_targets),
        axis=(1, 2),
    )
    attention_loss = mx.sum(
        attention_sample_losses * attention_weights
    ) / mx.maximum(
        mx.sum(attention_weights),
        1,
    )
    return (
        policy_loss
        + soft_policy_loss_weight * soft_policy_loss
        + VALUE_LOSS_WEIGHT * value_loss
        + ownership_loss_weight * ownership_loss
        + SCORE_LOSS_WEIGHT * score_loss
        + Q_VALUE_LOSS_WEIGHT * q_value_loss
        + ATTENTION_TRANSFER_LOSS_WEIGHT * attention_loss
        + listwise_policy_weight * listwise_policy_loss
        + q_rank_policy_weight * q_rank_policy_loss
        + policy_preservation_weight * policy_preservation_loss,
        (
            policy_loss,
            value_loss,
            ownership_loss,
            score_loss,
            q_value_loss,
            attention_loss,
            soft_policy_loss,
            listwise_policy_loss,
            q_rank_policy_loss,
        ),
    )


def evaluate(
    model: MokaNetwork,
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
        ownership_targets = mx.zeros(
            (len(feature_batch), BOARD_SIZE, BOARD_SIZE),
            dtype=mx.float32,
        )
        ownership_weights = mx.zeros(len(feature_batch), dtype=mx.float32)
        score_targets = mx.zeros(len(feature_batch), dtype=mx.float32)
        score_weights = mx.zeros(len(feature_batch), dtype=mx.float32)
        q_value_targets = mx.zeros_like(policy_batch)
        q_value_weights = mx.zeros_like(policy_batch)
        attention_targets = mx.zeros(
            (len(feature_batch), BOARD_SIZE, BOARD_SIZE),
            dtype=mx.float32,
        )
        attention_weights = mx.zeros(len(feature_batch), dtype=mx.float32)
        loss, _ = calculate_loss(
            model,
            feature_batch,
            policy_batch,
            value_batch,
            sample_weights,
            ownership_targets,
            ownership_weights,
            score_targets,
            score_weights,
            q_value_targets,
            q_value_weights,
            attention_targets,
            attention_weights,
            OWNERSHIP_LOSS_WEIGHT,
            0,
            0,
            False,
            False,
            False,
            0,
            0,
            mx.zeros_like(policy_batch),
            0,
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
    use_nested_network: bool,
    use_spatial_network: bool,
    use_recurrent_network: bool,
    use_context_network: bool,
    use_wide_network: bool,
    use_global_pool_network: bool,
    selection_metric: str,
    policy_top_k: int,
    reverse_kl_weight: float,
    soft_policy_loss_weight: float,
    use_auxiliary_network: bool,
    use_score_auxiliary: bool,
    ownership_loss_weight: float,
    use_q_auxiliary: bool,
    use_attention_auxiliary: bool,
    listwise_policy_weight: float,
    q_rank_policy_weight: float,
    auxiliary_checkpoint_path: Path | None,
    freeze_trunk: bool,
    freeze_policy_convolution: bool,
    policy_preservation_weight: float,
    train_global_adapter_only: bool,
    use_int8_quantization_aware_training: bool,
    use_global_residual_network: bool,
    global_residual_block_interval: int | None,
    train_global_residual_adapter_only: bool,
    train_new_global_residual_adapters_only: bool,
) -> None:
    checkpoint_global_residual_block_interval = (
        get_checkpoint_global_residual_block_interval(
            str(initial_checkpoint_path)
        )
        if initial_checkpoint_path is not None
        else 0
    )
    if checkpoint_global_residual_block_interval > 0:
        use_global_residual_network = True
    global_residual_block_interval = (
        global_residual_block_interval
        or checkpoint_global_residual_block_interval
        or GLOBAL_RESIDUAL_BLOCK_INTERVAL
    )
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
    ownerships = (
        dataset["ownerships"].astype(np.float32)
        if "ownerships" in dataset
        else np.zeros(
            (len(features), BOARD_SIZE, BOARD_SIZE),
            dtype=np.float32,
        )
    )
    ownership_weights = (
        np.ones(len(features), dtype=np.float32)
        if "ownerships" in dataset
        else np.zeros(len(features), dtype=np.float32)
    )
    scores = (
        normalize_score_targets(dataset["scores"].astype(np.float32))
        if "scores" in dataset
        else np.zeros(len(features), dtype=np.float32)
    )
    score_weights = (
        np.ones(len(features), dtype=np.float32)
        if "scores" in dataset and use_score_auxiliary
        else np.zeros(len(features), dtype=np.float32)
    )
    q_values = (
        dataset["q_values"].astype(np.float32)
        if "q_values" in dataset
        else np.zeros_like(policies)
    )
    q_weights = (
        dataset["q_weights"].astype(np.float32)
        if "q_weights" in dataset
        else np.zeros_like(policies)
    )
    attentions = (
        dataset["attentions"].astype(np.float32)
        if "attentions" in dataset
        else np.zeros(
            (len(features), BOARD_SIZE, BOARD_SIZE),
            dtype=np.float32,
        )
    )
    attention_weights = (
        np.ones(len(features), dtype=np.float32)
        if "attentions" in dataset
        else np.zeros(len(features), dtype=np.float32)
    )
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
    training_ownerships = ownerships[training_indexes]
    training_ownership_weights = ownership_weights[training_indexes]
    training_scores = scores[training_indexes]
    training_score_weights = score_weights[training_indexes]
    training_q_values = q_values[training_indexes]
    training_q_weights = q_weights[training_indexes]
    training_attentions = attentions[training_indexes]
    training_attention_weights = attention_weights[training_indexes]

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
        supplemental_ownerships = (
            supplemental_dataset["ownerships"].astype(np.float32)
            if "ownerships" in supplemental_dataset
            else np.zeros(
                (len(supplemental_features), BOARD_SIZE, BOARD_SIZE),
                dtype=np.float32,
            )
        )
        supplemental_ownership_weights = (
            np.ones(len(supplemental_features), dtype=np.float32)
            if "ownerships" in supplemental_dataset
            else np.zeros(len(supplemental_features), dtype=np.float32)
        )
        supplemental_scores = (
            normalize_score_targets(
                supplemental_dataset["scores"].astype(np.float32)
            )
            if "scores" in supplemental_dataset
            else np.zeros(len(supplemental_features), dtype=np.float32)
        )
        supplemental_score_weights = (
            np.ones(len(supplemental_features), dtype=np.float32)
            if "scores" in supplemental_dataset and use_score_auxiliary
            else np.zeros(len(supplemental_features), dtype=np.float32)
        )
        supplemental_q_values = (
            supplemental_dataset["q_values"].astype(np.float32)
            if "q_values" in supplemental_dataset
            else np.zeros(
                (len(supplemental_features), policies.shape[1]),
                dtype=np.float32,
            )
        )
        supplemental_q_weights = (
            supplemental_dataset["q_weights"].astype(np.float32)
            if "q_weights" in supplemental_dataset
            else np.zeros_like(supplemental_q_values)
        )
        supplemental_attentions = (
            supplemental_dataset["attentions"].astype(np.float32)
            if "attentions" in supplemental_dataset
            else np.zeros(
                (len(supplemental_features), BOARD_SIZE, BOARD_SIZE),
                dtype=np.float32,
            )
        )
        supplemental_attention_weights = (
            np.ones(len(supplemental_features), dtype=np.float32)
            if "attentions" in supplemental_dataset
            else np.zeros(len(supplemental_features), dtype=np.float32)
        )
        supplemental_values = supplemental_dataset["values"].astype(np.float32)
        if "short_values" in supplemental_dataset:
            supplemental_values = (
                (1 - SHORT_VALUE_TARGET_WEIGHT) * supplemental_values
                + SHORT_VALUE_TARGET_WEIGHT
                * supplemental_dataset["short_values"].astype(np.float32)
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
                supplemental_values[supplemental_indexes],
            ]
        )
        training_ownerships = np.concatenate(
            [
                training_ownerships,
                supplemental_ownerships[supplemental_indexes],
            ]
        )
        training_ownership_weights = np.concatenate(
            [
                training_ownership_weights,
                supplemental_ownership_weights[supplemental_indexes],
            ]
        )
        training_scores = np.concatenate(
            [
                training_scores,
                supplemental_scores[supplemental_indexes],
            ]
        )
        training_score_weights = np.concatenate(
            [
                training_score_weights,
                supplemental_score_weights[supplemental_indexes],
            ]
        )
        training_q_values = np.concatenate(
            [
                training_q_values,
                supplemental_q_values[supplemental_indexes],
            ]
        )
        training_q_weights = np.concatenate(
            [
                training_q_weights,
                supplemental_q_weights[supplemental_indexes],
            ]
        )
        training_attentions = np.concatenate(
            [
                training_attentions,
                supplemental_attentions[supplemental_indexes],
            ]
        )
        training_attention_weights = np.concatenate(
            [
                training_attention_weights,
                supplemental_attention_weights[supplemental_indexes],
            ]
        )

    print(
        f"training={len(training_features):,} "
        f"validation={len(validation_indexes):,} "
        f"test={len(test_indexes):,}"
    )
    model = (
        MokaAttentionAuxiliaryNetwork()
        if use_attention_auxiliary
        else (
            MokaQAuxiliaryNetwork()
            if use_q_auxiliary
            else (
                MokaAuxiliaryNetwork()
                if use_auxiliary_network
                else (
                    MokaSoftPolicyNetwork()
                    if soft_policy_loss_weight > 0
                    else create_moka_network(
                        use_nested_network,
                        use_spatial_network,
                        use_recurrent_network,
                        use_context_network,
                        use_wide_network,
                        use_global_pool_network,
                        use_global_residual_network,
                        global_residual_block_interval,
                    )
                )
            )
        )
    )

    if initial_checkpoint_path:
        model.load_weights(
            str(initial_checkpoint_path),
            strict=(
                not use_spatial_network
                and soft_policy_loss_weight == 0
                and not use_auxiliary_network
                and not use_q_auxiliary
                and not use_attention_auxiliary
                and not use_global_pool_network
                and not use_global_residual_network
            ),
        )
    reference_model = None
    if policy_preservation_weight > 0:
        if initial_checkpoint_path is None:
            raise ValueError(
                "Policy preservation requires an initial checkpoint."
            )
        reference_model = create_moka_network(
            use_nested_network,
            use_spatial_network,
            use_recurrent_network,
            use_context_network,
            use_wide_network,
            use_global_pool_network,
            use_global_residual_network,
            global_residual_block_interval,
        )
        reference_model.load_weights(
            str(initial_checkpoint_path),
            strict=(
                not use_global_pool_network
                and not use_global_residual_network
            ),
        )
        if use_int8_quantization_aware_training:
            reference_model.update(
                fake_quantize_int8_parameters(reference_model.parameters())
            )
        reference_model.freeze()
        reference_model.eval()
    selected_global_adapter_mode_count = sum(
        [
            train_global_adapter_only,
            train_global_residual_adapter_only,
            train_new_global_residual_adapters_only,
        ]
    )
    if selected_global_adapter_mode_count > 1:
        raise ValueError("Select only one global adapter training mode.")
    if train_global_adapter_only:
        if not use_global_pool_network:
            raise ValueError(
                "Global adapter training requires the global-pool network."
            )
        model.freeze()
        model.global_pooling_hidden.unfreeze()
        model.global_policy_output.unfreeze()
        model.global_value_output.unfreeze()
    elif train_global_residual_adapter_only:
        if not use_global_residual_network:
            raise ValueError(
                "Global residual adapter training requires the "
                "global-residual network."
            )
        model.freeze()
        for residual_block in model.residual_blocks:
            if isinstance(
                residual_block,
                GlobalNestedBottleneckBlock,
            ):
                residual_block.global_pooling_hidden.unfreeze()
                residual_block.global_bias_output.unfreeze()
    elif train_new_global_residual_adapters_only:
        if (
            not use_global_residual_network
            or checkpoint_global_residual_block_interval <= 0
            or global_residual_block_interval
            >= checkpoint_global_residual_block_interval
        ):
            raise ValueError(
                "New global-residual adapter training requires a denser "
                "global-residual interval."
            )
        model.freeze()
        for block_index, residual_block in enumerate(
            model.residual_blocks
        ):
            block_number = block_index + 1
            if (
                isinstance(
                    residual_block,
                    GlobalNestedBottleneckBlock,
                )
                and block_number
                % checkpoint_global_residual_block_interval
                != 0
            ):
                residual_block.global_pooling_hidden.unfreeze()
                residual_block.global_bias_output.unfreeze()
    elif freeze_trunk:
        if (
            soft_policy_loss_weight > 0
            or use_auxiliary_network
            or use_q_auxiliary
            or use_attention_auxiliary
        ):
            raise ValueError(
                "Trunk freezing is only supported by the deployment network."
            )
        model.freeze()
        if not freeze_policy_convolution:
            model.policy_convolution.unfreeze()
        model.policy_linear.unfreeze()
    elif freeze_policy_convolution:
        raise ValueError(
            "Policy-convolution freezing requires trunk freezing."
        )
    if use_int8_quantization_aware_training and (
        train_global_adapter_only
        or soft_policy_loss_weight > 0
        or use_auxiliary_network
        or use_q_auxiliary
        or use_attention_auxiliary
    ):
        raise ValueError(
            "INT8 quantization-aware training requires a fully trainable "
            "deployment network."
        )

    optimizer = optim.AdamW(learning_rate=learning_rate)
    quantized_model = (
        create_moka_network(
            use_nested_network,
            use_spatial_network,
            use_recurrent_network,
            use_context_network,
            use_wide_network,
            use_global_pool_network,
            use_global_residual_network,
            global_residual_block_interval,
        )
        if use_int8_quantization_aware_training
        else None
    )

    if quantized_model is not None:
        quantized_model.update(
            fake_quantize_int8_parameters(model.parameters())
        )
        mx.eval(quantized_model.parameters())

        def calculate_quantized_loss(
            parameters: dict,
            *loss_arguments: object,
        ) -> tuple:
            quantized_model.update(fake_quantize_int8_parameters(parameters))
            return calculate_loss(quantized_model, *loss_arguments)

        loss_and_grad = mx.value_and_grad(calculate_quantized_loss)
    else:
        loss_and_grad = nn.value_and_grad(model, calculate_loss)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    best_move_agreement = 0.0

    for epoch_index in range(epoch_count):
        shuffled_indexes = random_generator.permutation(len(training_features))
        training_losses: list[float] = []

        for batch_start in range(0, len(shuffled_indexes), batch_size):
            batch_indexes = shuffled_indexes[batch_start : batch_start + batch_size]
            rotation_counts = random_generator.integers(
                SYMMETRY_ROTATION_COUNT,
                size=len(batch_indexes),
            )
            should_flip = random_generator.integers(
                SYMMETRY_FLIP_OPTION_COUNT,
                size=len(batch_indexes),
            ).astype(np.bool_)
            augmented_features, augmented_policies = apply_batch_board_symmetry(
                training_features[batch_indexes],
                training_policies[batch_indexes],
                rotation_counts,
                should_flip,
            )
            augmented_ownerships = apply_batch_spatial_symmetry(
                training_ownerships[batch_indexes],
                rotation_counts,
                should_flip,
            )
            augmented_attentions = apply_batch_spatial_symmetry(
                training_attentions[batch_indexes],
                rotation_counts,
                should_flip,
            )
            _, augmented_q_values = apply_batch_board_symmetry(
                training_features[batch_indexes],
                training_q_values[batch_indexes],
                rotation_counts,
                should_flip,
            )
            _, augmented_q_weights = apply_batch_board_symmetry(
                training_features[batch_indexes],
                training_q_weights[batch_indexes],
                rotation_counts,
                should_flip,
            )

            feature_batch = mx.array(augmented_features, dtype=mx.float32)
            if policy_top_k > 0:
                top_move_indexes = np.argpartition(
                    augmented_policies,
                    -policy_top_k,
                    axis=1,
                )[:, -policy_top_k:]
                top_move_mask = np.zeros_like(augmented_policies)
                np.put_along_axis(
                    top_move_mask,
                    top_move_indexes,
                    1,
                    axis=1,
                )
                augmented_policies *= top_move_mask
                augmented_policies /= np.sum(
                    augmented_policies,
                    axis=1,
                    keepdims=True,
                )
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
            ownership_batch = mx.array(
                augmented_ownerships,
                dtype=mx.float32,
            )
            ownership_weight_batch = mx.array(
                training_ownership_weights[batch_indexes],
                dtype=mx.float32,
            )
            score_batch = mx.array(
                training_scores[batch_indexes],
                dtype=mx.float32,
            )
            score_weight_batch = mx.array(
                training_score_weights[batch_indexes],
                dtype=mx.float32,
            )
            q_value_batch = mx.array(augmented_q_values, dtype=mx.float32)
            q_weight_batch = mx.array(augmented_q_weights, dtype=mx.float32)
            attention_batch = mx.array(
                augmented_attentions,
                dtype=mx.float32,
            )
            attention_weight_batch = mx.array(
                training_attention_weights[batch_indexes],
                dtype=mx.float32,
            )
            reference_policy_logits = (
                reference_model(feature_batch)[0]
                if reference_model is not None
                else mx.zeros_like(policy_batch)
            )
            mx.eval(reference_policy_logits)
            loss_arguments = (
                feature_batch,
                policy_batch,
                value_batch,
                weight_batch,
                ownership_batch,
                ownership_weight_batch,
                score_batch,
                score_weight_batch,
                q_value_batch,
                q_weight_batch,
                attention_batch,
                attention_weight_batch,
                ownership_loss_weight,
                reverse_kl_weight,
                soft_policy_loss_weight,
                use_auxiliary_network,
                use_q_auxiliary,
                use_attention_auxiliary,
                listwise_policy_weight,
                q_rank_policy_weight,
                reference_policy_logits,
                policy_preservation_weight,
            )
            (loss, _), gradients = (
                loss_and_grad(
                    model.trainable_parameters(),
                    *loss_arguments,
                )
                if quantized_model is not None
                else loss_and_grad(model, *loss_arguments)
            )
            optimizer.update(model, gradients)
            if quantized_model is not None:
                quantized_model.update(
                    fake_quantize_int8_parameters(model.parameters())
                )
                mx.eval(
                    model.parameters(),
                    quantized_model.parameters(),
                    optimizer.state,
                    loss,
                )
            else:
                mx.eval(model.parameters(), optimizer.state, loss)
            training_losses.append(float(loss.item()))

        evaluation_model = quantized_model if quantized_model is not None else model
        validation_loss, move_agreement, value_error = evaluate(
            evaluation_model,
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

        did_improve = (
            validation_loss < best_validation_loss
            if selection_metric == TRAIN_SELECTION_LOSS
            else move_agreement > best_move_agreement
        )

        if did_improve:
            best_validation_loss = validation_loss
            best_move_agreement = move_agreement
            model.save_weights(str(checkpoint_path))

    model.load_weights(str(checkpoint_path))
    if quantized_model is not None:
        quantized_model.update(fake_quantize_int8_parameters(model.parameters()))
    test_loss, test_move_agreement, test_value_error = evaluate(
        quantized_model if quantized_model is not None else model,
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
    if auxiliary_checkpoint_path:
        auxiliary_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(auxiliary_checkpoint_path))
    if (
        soft_policy_loss_weight > 0
        or use_auxiliary_network
        or use_q_auxiliary
        or use_attention_auxiliary
    ):
        deployment_model = MokaNestedNetwork()
        deployment_parameters = [
            parameter
            for parameter in tree_flatten(model.parameters())
            if not parameter[0].startswith(("soft_policy_", "auxiliary_"))
        ]
        deployment_model.update(tree_unflatten(deployment_parameters))
        deployment_model.save_weights(str(checkpoint_path))


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
        default=Path("checkpoints/moka-model.safetensors"),
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
    argument_parser.add_argument("--policy-top-k", type=int, default=0)
    argument_parser.add_argument("--reverse-kl-weight", type=float, default=0)
    argument_parser.add_argument("--nested", action="store_true")
    argument_parser.add_argument("--spatial", action="store_true")
    argument_parser.add_argument("--recurrent", action="store_true")
    argument_parser.add_argument("--context", action="store_true")
    argument_parser.add_argument("--wide", action="store_true")
    argument_parser.add_argument("--global-pool", action="store_true")
    argument_parser.add_argument("--global-residual", action="store_true")
    argument_parser.add_argument(
        "--global-residual-interval",
        type=int,
    )
    argument_parser.add_argument("--soft-policy", action="store_true")
    argument_parser.add_argument("--soft-policy-weight", type=float, default=0)
    argument_parser.add_argument("--auxiliary", action="store_true")
    argument_parser.add_argument("--auxiliary-score", action="store_true")
    argument_parser.add_argument("--q-auxiliary", action="store_true")
    argument_parser.add_argument("--attention-transfer", action="store_true")
    argument_parser.add_argument("--freeze-trunk", action="store_true")
    argument_parser.add_argument(
        "--freeze-policy-convolution",
        action="store_true",
    )
    argument_parser.add_argument(
        "--policy-preservation-weight",
        type=float,
        default=0,
    )
    argument_parser.add_argument(
        "--global-adapter-only",
        action="store_true",
    )
    argument_parser.add_argument(
        "--global-residual-adapter-only",
        action="store_true",
    )
    argument_parser.add_argument(
        "--new-global-residual-adapters-only",
        action="store_true",
    )
    argument_parser.add_argument(
        "--int8-quantization-aware",
        action="store_true",
    )
    argument_parser.add_argument("--auxiliary-checkpoint", type=Path)
    argument_parser.add_argument("--listwise-weight", type=float, default=0)
    argument_parser.add_argument("--q-rank-weight", type=float, default=0)
    argument_parser.add_argument(
        "--ownership-weight",
        type=float,
        default=OWNERSHIP_LOSS_WEIGHT,
    )
    argument_parser.add_argument(
        "--selection-metric",
        choices=TRAIN_SELECTION_METRICS,
        default=TRAIN_SELECTION_LOSS,
    )
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
        arguments.nested,
        arguments.spatial,
        arguments.recurrent,
        arguments.context,
        arguments.wide,
        arguments.global_pool,
        arguments.selection_metric,
        arguments.policy_top_k,
        arguments.reverse_kl_weight,
        (
            SOFT_POLICY_LOSS_WEIGHT
            if arguments.soft_policy
            else arguments.soft_policy_weight
        ),
        arguments.auxiliary,
        arguments.auxiliary_score,
        arguments.ownership_weight,
        arguments.q_auxiliary,
        arguments.attention_transfer,
        arguments.listwise_weight,
        arguments.q_rank_weight,
        arguments.auxiliary_checkpoint,
        arguments.freeze_trunk,
        arguments.freeze_policy_convolution,
        arguments.policy_preservation_weight,
        arguments.global_adapter_only,
        arguments.int8_quantization_aware,
        arguments.global_residual,
        arguments.global_residual_interval,
        arguments.global_residual_adapter_only,
        arguments.new_global_residual_adapters_only,
    )


if __name__ == "__main__":
    main()
