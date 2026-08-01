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
    PAIRWISE_VALUE_MINIMUM_GAP,
    POLICY_MOVE_COUNT,
    SCORE_HIDDEN_CHANNEL_COUNT,
    SHORT_VALUE_AUXILIARY_LOSS_WEIGHT,
    SYMMETRY_FLIP_OPTION_COUNT,
    SYMMETRY_ROTATION_COUNT,
    TEST_BUCKET_INDEX,
    VALIDATION_BUCKET_INDEX,
)
from go_model.model import (
    MokaGlobalResidualNetwork,
    MokaNestedNetwork,
    create_moka_network_for_checkpoint,
)
from go_model.quantization import fake_quantize_int8_parameters
from go_model.split import create_game_split_buckets
from go_model.symmetry import apply_batch_board_symmetry


def create_value_targets(
    dataset,
    key_prefix: str,
    outcome_target_weight: float,
    short_value_target_weight: float,
    root_search_target_weight: float,
) -> np.ndarray:
    if not 0 <= outcome_target_weight <= 1:
        raise ValueError("Outcome target weight must be between zero and one.")
    if not 0 <= short_value_target_weight <= 1:
        raise ValueError(
            "Short-value target weight must be between zero and one."
        )
    if not 0 <= root_search_target_weight <= 1:
        raise ValueError(
            "Root-search target weight must be between zero and one."
        )
    if (
        outcome_target_weight
        + short_value_target_weight
        + root_search_target_weight
        > 1
    ):
        raise ValueError(
            "Value target weights must sum to at most one."
        )

    values = dataset[f"{key_prefix}values"].astype(np.float32)
    if key_prefix:
        return values

    teacher_values = (
        dataset["teacher_values"].astype(np.float32)
        if "teacher_values" in dataset
        else values
    )
    outcome_values = np.clip(values, -1, 1)
    short_values = (
        dataset["teacher_short_values"].astype(np.float32)
        if "teacher_short_values" in dataset
        else teacher_values
    )
    root_search_values = teacher_values
    if "search_q_values" in dataset and "search_q_weights" in dataset:
        search_q_values = dataset["search_q_values"].astype(np.float32)
        search_q_weights = dataset["search_q_weights"].astype(np.float32)
        root_visit_counts = np.sum(search_q_weights, axis=1)
        searched_values = np.sum(
            search_q_values * search_q_weights,
            axis=1,
        ) / np.maximum(root_visit_counts, 1)
        root_search_values = np.where(
            root_visit_counts > 0,
            searched_values,
            teacher_values,
        )
    return np.clip(
        (
            1
            - outcome_target_weight
            - short_value_target_weight
            - root_search_target_weight
        )
        * teacher_values
        + short_value_target_weight * short_values
        + outcome_target_weight * outcome_values
        + root_search_target_weight * root_search_values,
        -1,
        1,
    )


def create_short_value_targets(dataset, key_prefix: str) -> np.ndarray:
    if key_prefix:
        raise ValueError(
            "Short-value auxiliary training does not support child targets."
        )
    if "teacher_short_values" not in dataset:
        raise ValueError("Dataset does not contain teacher short values.")
    return dataset["teacher_short_values"].astype(np.float32)


def load_value_dataset(
    dataset_path: Path,
    use_child_targets: bool,
    outcome_target_weight: float,
    short_value_target_weight: float,
    root_search_target_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = np.load(dataset_path)
    key_prefix = "child_" if use_child_targets else ""
    features = dataset[f"{key_prefix}features"].astype(np.float32)
    game_ids = dataset[f"{key_prefix}game_ids"]
    values = create_value_targets(
        dataset,
        key_prefix,
        outcome_target_weight,
        short_value_target_weight,
        root_search_target_weight,
    )
    sample_weights = (
        dataset[f"{key_prefix}weights"].astype(np.float32)
        if f"{key_prefix}weights" in dataset
        else np.ones(len(features), dtype=np.float32)
    )
    return features, game_ids, values, sample_weights


class MokaShortValueTrainingNetwork(nn.Module):
    def __init__(self, base_model: MokaNestedNetwork) -> None:
        super().__init__()
        self.base_model = base_model
        self.auxiliary_short_value_output = nn.Linear(
            SCORE_HIDDEN_CHANNEL_COUNT,
            1,
        )
        self.auxiliary_short_value_output.weight = mx.array(
            base_model.value_output.weight
        )
        self.auxiliary_short_value_output.bias = mx.array(
            base_model.value_output.bias
        )

    def __call__(
        self,
        inputs: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        trunk_values = self.base_model.get_trunk_values(inputs)
        policy_values = nn.relu(
            self.base_model.policy_convolution(trunk_values)
        )
        policy_logits = self.base_model.policy_linear(
            mx.flatten(policy_values, start_axis=1)
        )
        value_values = nn.relu(
            self.base_model.value_convolution(trunk_values)
        )
        value_hidden = nn.relu(
            self.base_model.value_hidden(
                mx.flatten(value_values, start_axis=1)
            )
        )
        long_value = mx.tanh(
            self.base_model.value_output(value_hidden)
        ).squeeze(-1)
        short_value = mx.tanh(
            self.auxiliary_short_value_output(value_hidden)
        ).squeeze(-1)
        return policy_logits, long_value, short_value


def apply_random_value_symmetry(
    features: np.ndarray,
    random_generator: np.random.Generator,
) -> np.ndarray:
    rotation_counts = random_generator.integers(
        0,
        SYMMETRY_ROTATION_COUNT,
        size=len(features),
    )
    should_flip = random_generator.integers(
        0,
        SYMMETRY_FLIP_OPTION_COUNT,
        size=len(features),
    ).astype(np.bool_)
    transformed_features, _ = apply_batch_board_symmetry(
        features,
        np.zeros(
            (len(features), POLICY_MOVE_COUNT),
            dtype=np.float32,
        ),
        rotation_counts,
        should_flip,
    )
    return transformed_features


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


def calculate_short_auxiliary_value_loss(
    model: MokaShortValueTrainingNetwork,
    features: mx.array,
    targets: mx.array,
    short_targets: mx.array,
    sample_weights: mx.array,
    reference_policy_logits: mx.array,
    policy_preservation_weight: float,
    short_value_auxiliary_weight: float,
) -> mx.array:
    policy_logits, values, short_values = model(features)
    weight_sum = mx.sum(sample_weights)
    value_loss = mx.sum(
        sample_weights * mx.square(values - targets)
    ) / weight_sum
    short_value_loss = mx.sum(
        sample_weights * mx.square(short_values - short_targets)
    ) / weight_sum
    policy_preservation_loss = mx.mean(
        mx.square(policy_logits - reference_policy_logits)
    )
    return (
        value_loss
        + short_value_auxiliary_weight * short_value_loss
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


def evaluate_short_auxiliary_value(
    model: MokaShortValueTrainingNetwork,
    features: np.ndarray,
    targets: np.ndarray,
    short_targets: np.ndarray,
    sample_weights: np.ndarray,
    batch_size: int,
) -> tuple[float, float]:
    value_error_sum = 0.0
    short_value_error_sum = 0.0
    weight_sum = 0.0

    for batch_start in range(0, len(features), batch_size):
        batch_end = batch_start + batch_size
        _, values, short_values = model(
            mx.array(features[batch_start:batch_end], dtype=mx.float32)
        )
        mx.eval(values, short_values)
        batch_weights = sample_weights[batch_start:batch_end]
        value_error_sum += float(
            np.sum(
                batch_weights
                * np.abs(
                    np.asarray(values) - targets[batch_start:batch_end]
                )
            )
        )
        short_value_error_sum += float(
            np.sum(
                batch_weights
                * np.abs(
                    np.asarray(short_values)
                    - short_targets[batch_start:batch_end]
                )
            )
        )
        weight_sum += float(np.sum(batch_weights))

    return (
        value_error_sum / weight_sum,
        short_value_error_sum / weight_sum,
    )


def train_value_head(
    dataset_path: Path,
    supplemental_dataset_paths: list[Path],
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
    outcome_target_weight: float,
    short_value_target_weight: float,
    short_value_auxiliary_weight: float,
    root_search_target_weight: float,
    use_int8_quantization_aware_training: bool,
    use_symmetry_augmentation: bool,
    game_pair_size: int,
) -> None:
    if short_value_auxiliary_weight < 0:
        raise ValueError(
            "Short-value auxiliary weight must not be negative."
        )
    if short_value_auxiliary_weight > 0 and use_child_targets:
        raise ValueError(
            "Short-value auxiliary training does not support child targets."
        )
    if short_value_auxiliary_weight > 0 and short_value_target_weight > 0:
        raise ValueError(
            "Short value cannot be both an auxiliary and blended target."
        )
    if short_value_auxiliary_weight > 0 and pairwise_ranking_weight > 0:
        raise ValueError(
            "Short-value auxiliary training does not support ranking loss."
        )
    dataset = np.load(dataset_path)
    key_prefix = "child_" if use_child_targets else ""
    features, game_ids, values, sample_weights = load_value_dataset(
        dataset_path,
        use_child_targets,
        outcome_target_weight,
        short_value_target_weight,
        root_search_target_weight,
    )
    game_buckets = create_game_split_buckets(game_ids, game_pair_size)
    validation_indexes = np.flatnonzero(game_buckets == VALIDATION_BUCKET_INDEX)
    test_indexes = np.flatnonzero(game_buckets == TEST_BUCKET_INDEX)
    training_indexes = np.flatnonzero(
        (game_buckets != VALIDATION_BUCKET_INDEX)
        & (game_buckets != TEST_BUCKET_INDEX)
    )
    training_features = features[training_indexes]
    training_values = values[training_indexes]
    short_values = (
        create_short_value_targets(dataset, key_prefix)
        if short_value_auxiliary_weight > 0
        else None
    )
    training_short_values = (
        short_values[training_indexes]
        if short_values is not None
        else None
    )
    training_sample_weights = sample_weights[training_indexes]
    for supplemental_dataset_path in supplemental_dataset_paths:
        (
            supplemental_features,
            supplemental_game_ids,
            supplemental_values,
            supplemental_sample_weights,
        ) = load_value_dataset(
            supplemental_dataset_path,
            False,
            outcome_target_weight,
            short_value_target_weight,
            root_search_target_weight,
        )
        supplemental_game_buckets = create_game_split_buckets(
            supplemental_game_ids,
            game_pair_size,
        )
        supplemental_training_indexes = np.flatnonzero(
            (supplemental_game_buckets != VALIDATION_BUCKET_INDEX)
            & (supplemental_game_buckets != TEST_BUCKET_INDEX)
        )
        training_features = np.concatenate(
            [
                training_features,
                supplemental_features[supplemental_training_indexes],
            ]
        )
        training_values = np.concatenate(
            [
                training_values,
                supplemental_values[supplemental_training_indexes],
            ]
        )
        if training_short_values is not None:
            supplemental_dataset = np.load(supplemental_dataset_path)
            supplemental_short_values = create_short_value_targets(
                supplemental_dataset,
                "",
            )
            training_short_values = np.concatenate(
                [
                    training_short_values,
                    supplemental_short_values[
                        supplemental_training_indexes
                    ],
                ]
            )
        training_sample_weights = np.concatenate(
            [
                training_sample_weights,
                supplemental_sample_weights[
                    supplemental_training_indexes
                ],
            ]
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
    if (
        short_value_auxiliary_weight > 0
        and model.__class__ is not MokaGlobalResidualNetwork
    ):
        raise ValueError(
            "Short-value auxiliary training requires a global-residual model."
        )
    short_training_model = (
        MokaShortValueTrainingNetwork(model)
        if short_value_auxiliary_weight > 0
        else None
    )
    optimization_model = (
        short_training_model if short_training_model is not None else model
    )
    optimization_model.freeze()

    if short_training_model is not None:
        short_training_model.base_model.value_convolution.unfreeze()
        short_training_model.base_model.value_hidden.unfreeze()
        short_training_model.base_model.value_output.unfreeze()
        short_training_model.auxiliary_short_value_output.unfreeze()
    elif should_unfreeze_trunk:
        model.unfreeze()
        model.policy_convolution.freeze()
        model.policy_linear.freeze()
    else:
        model.value_convolution.unfreeze()
        model.value_hidden.unfreeze()
        model.value_output.unfreeze()
    optimizer = optim.AdamW(learning_rate=learning_rate)
    quantized_model = (
        create_moka_network_for_checkpoint(str(initial_checkpoint_path))
        if use_int8_quantization_aware_training
        and short_training_model is None
        else None
    )
    quantized_short_training_model = (
        MokaShortValueTrainingNetwork(
            create_moka_network_for_checkpoint(
                str(initial_checkpoint_path)
            )
        )
        if use_int8_quantization_aware_training
        and short_training_model is not None
        else None
    )
    if quantized_model is not None:
        quantized_model.update(
            fake_quantize_int8_parameters(model.parameters())
        )
        mx.eval(quantized_model.parameters())

        def calculate_quantized_value_loss(
            parameters: dict,
            *loss_arguments: object,
        ) -> mx.array:
            quantized_model.update(
                fake_quantize_int8_parameters(parameters)
            )
            return calculate_value_loss(
                quantized_model,
                *loss_arguments,
            )

        def calculate_quantized_ranked_value_loss(
            parameters: dict,
            *loss_arguments: object,
        ) -> mx.array:
            quantized_model.update(
                fake_quantize_int8_parameters(parameters)
            )
            return calculate_ranked_value_loss(
                quantized_model,
                *loss_arguments,
            )

        loss_and_grad = mx.value_and_grad(
            calculate_quantized_value_loss
        )
        ranked_loss_and_grad = mx.value_and_grad(
            calculate_quantized_ranked_value_loss
        )
    elif quantized_short_training_model is not None:
        quantized_short_training_model.update(
            fake_quantize_int8_parameters(
                short_training_model.parameters()
            )
        )
        mx.eval(quantized_short_training_model.parameters())

        def calculate_quantized_short_auxiliary_value_loss(
            parameters: dict,
            *loss_arguments: object,
        ) -> mx.array:
            quantized_short_training_model.update(
                fake_quantize_int8_parameters(parameters)
            )
            return calculate_short_auxiliary_value_loss(
                quantized_short_training_model,
                *loss_arguments,
            )

        loss_and_grad = mx.value_and_grad(
            calculate_quantized_short_auxiliary_value_loss
        )
        ranked_loss_and_grad = None
    elif short_training_model is not None:
        loss_and_grad = nn.value_and_grad(
            short_training_model,
            calculate_short_auxiliary_value_loss,
        )
        ranked_loss_and_grad = None
    else:
        loss_and_grad = nn.value_and_grad(
            model,
            calculate_value_loss,
        )
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
                create_game_split_buckets(pair_game_ids, game_pair_size),
                [VALIDATION_BUCKET_INDEX, TEST_BUCKET_INDEX],
            )
        )
        print(
            f"pairs={len(preferred_indexes):,} "
            f"training_pairs={len(pair_training_indexes):,}"
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    short_training_checkpoint_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}-short-auxiliary{checkpoint_path.suffix}"
    )
    best_validation_error = float("inf")
    best_validation_ranking_error = float("inf")

    for epoch_index in range(epoch_count):
        shuffled_indexes = random_generator.permutation(
            len(training_features)
        )
        training_losses: list[float] = []

        for batch_start in range(0, len(shuffled_indexes), batch_size):
            batch_indexes = shuffled_indexes[batch_start : batch_start + batch_size]
            batch_features = training_features[batch_indexes]
            if use_symmetry_augmentation:
                batch_features = apply_random_value_symmetry(
                    batch_features,
                    random_generator,
                )
            feature_batch = mx.array(
                batch_features,
                dtype=mx.float32,
            )
            target_batch = mx.array(
                training_values[batch_indexes],
                dtype=mx.float32,
            )
            weight_batch = mx.array(
                training_sample_weights[batch_indexes],
                dtype=mx.float32,
            )
            reference_policy_logits, _ = reference_model(feature_batch)
            mx.eval(reference_policy_logits)
            if short_training_model is not None:
                short_target_batch = mx.array(
                    training_short_values[batch_indexes],
                    dtype=mx.float32,
                )
                loss_arguments = (
                    feature_batch,
                    target_batch,
                    short_target_batch,
                    weight_batch,
                    reference_policy_logits,
                    policy_preservation_weight,
                    short_value_auxiliary_weight,
                )
            elif pairwise_ranking_weight > 0:
                pair_batch_indexes = random_generator.choice(
                    pair_training_indexes,
                    size=len(batch_indexes),
                    replace=len(pair_training_indexes) < len(batch_indexes),
                )
                preferred_batch_features = features[
                    preferred_indexes[pair_batch_indexes]
                ]
                alternative_batch_features = features[
                    alternative_indexes[pair_batch_indexes]
                ]
                if use_symmetry_augmentation:
                    pair_rotation_counts = random_generator.integers(
                        0,
                        SYMMETRY_ROTATION_COUNT,
                        size=len(pair_batch_indexes),
                    )
                    pair_should_flip = random_generator.integers(
                        0,
                        SYMMETRY_FLIP_OPTION_COUNT,
                        size=len(pair_batch_indexes),
                    ).astype(np.bool_)
                    empty_pair_policies = np.zeros(
                        (
                            len(pair_batch_indexes),
                            POLICY_MOVE_COUNT,
                        ),
                        dtype=np.float32,
                    )
                    preferred_batch_features, _ = (
                        apply_batch_board_symmetry(
                            preferred_batch_features,
                            empty_pair_policies,
                            pair_rotation_counts,
                            pair_should_flip,
                        )
                    )
                    alternative_batch_features, _ = (
                        apply_batch_board_symmetry(
                            alternative_batch_features,
                            empty_pair_policies,
                            pair_rotation_counts,
                            pair_should_flip,
                        )
                    )
                preferred_feature_batch = mx.array(
                    preferred_batch_features,
                    dtype=mx.float32,
                )
                alternative_feature_batch = mx.array(
                    alternative_batch_features,
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
                loss_arguments = (
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
                loss_arguments = (
                    feature_batch,
                    target_batch,
                    weight_batch,
                    reference_policy_logits,
                    policy_preservation_weight,
                )
            loss, gradients = (
                (
                    ranked_loss_and_grad(
                        model.trainable_parameters(),
                        *loss_arguments,
                    )
                    if pairwise_ranking_weight > 0
                    else loss_and_grad(
                        model.trainable_parameters(),
                        *loss_arguments,
                    )
                )
                if quantized_model is not None
                else (
                    loss_and_grad(
                        short_training_model.trainable_parameters(),
                        *loss_arguments,
                    )
                    if quantized_short_training_model is not None
                    else loss_and_grad(
                        short_training_model,
                        *loss_arguments,
                    )
                )
                if short_training_model is not None
                else (
                    ranked_loss_and_grad(model, *loss_arguments)
                    if pairwise_ranking_weight > 0
                    else loss_and_grad(model, *loss_arguments)
                )
            )
            optimizer.update(optimization_model, gradients)
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
            elif quantized_short_training_model is not None:
                quantized_short_training_model.update(
                    fake_quantize_int8_parameters(
                        short_training_model.parameters()
                    )
                )
                mx.eval(
                    short_training_model.parameters(),
                    quantized_short_training_model.parameters(),
                    optimizer.state,
                    loss,
                )
            else:
                mx.eval(
                    optimization_model.parameters(),
                    optimizer.state,
                    loss,
                )
            training_losses.append(float(loss.item()))

        if short_training_model is not None:
            validation_error, validation_short_error = (
                evaluate_short_auxiliary_value(
                    (
                        quantized_short_training_model
                        if quantized_short_training_model is not None
                        else short_training_model
                    ),
                    features[validation_indexes],
                    values[validation_indexes],
                    short_values[validation_indexes],
                    sample_weights[validation_indexes],
                    batch_size,
                )
            )
        else:
            validation_error = evaluate_value(
                quantized_model if quantized_model is not None else model,
                features[validation_indexes],
                values[validation_indexes],
                sample_weights[validation_indexes],
                batch_size,
            )
            validation_short_error = None
        progress = (
            f"epoch {epoch_index + 1:02d} "
            f"train={np.mean(training_losses):.4f} "
            f"validation_mae={validation_error:.4f}"
        )
        if validation_short_error is not None:
            progress += (
                f" validation_short_mae={validation_short_error:.4f}"
            )
        if pairwise_ranking_weight > 0:
            pair_validation_indexes = np.flatnonzero(
                create_game_split_buckets(pair_game_ids, game_pair_size)
                == VALIDATION_BUCKET_INDEX
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
            if short_training_model is not None:
                short_training_model.save_weights(
                    str(short_training_checkpoint_path)
                )
            else:
                model.save_weights(str(checkpoint_path))

    if short_training_model is not None:
        short_training_model.load_weights(
            str(short_training_checkpoint_path)
        )
        if quantized_short_training_model is not None:
            quantized_short_training_model.update(
                fake_quantize_int8_parameters(
                    short_training_model.parameters()
                )
            )
        test_error, test_short_error = evaluate_short_auxiliary_value(
            (
                quantized_short_training_model
                if quantized_short_training_model is not None
                else short_training_model
            ),
            features[test_indexes],
            values[test_indexes],
            short_values[test_indexes],
            sample_weights[test_indexes],
            batch_size,
        )
        short_training_model.base_model.save_weights(str(checkpoint_path))
        print(
            f"test_mae={test_error:.4f} "
            f"test_short_mae={test_short_error:.4f}"
        )
    else:
        model.load_weights(str(checkpoint_path))
        if quantized_model is not None:
            quantized_model.update(
                fake_quantize_int8_parameters(model.parameters())
            )
        test_error = evaluate_value(
            quantized_model if quantized_model is not None else model,
            features[test_indexes],
            values[test_indexes],
            sample_weights[test_indexes],
            batch_size,
        )
        print(f"test_mae={test_error:.4f}")
    if pairwise_ranking_weight > 0:
        pair_test_indexes = np.flatnonzero(
            create_game_split_buckets(pair_game_ids, game_pair_size)
            == TEST_BUCKET_INDEX
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
    argument_parser.add_argument(
        "--supplemental-data",
        action="append",
        default=[],
        type=Path,
    )
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
    argument_parser.add_argument(
        "--outcome-target-weight",
        type=float,
        default=0,
    )
    argument_parser.add_argument(
        "--short-value-target-weight",
        type=float,
        default=0,
    )
    argument_parser.add_argument(
        "--short-value-auxiliary-weight",
        type=float,
        default=SHORT_VALUE_AUXILIARY_LOSS_WEIGHT,
    )
    argument_parser.add_argument(
        "--root-search-target-weight",
        type=float,
        default=0,
    )
    argument_parser.add_argument(
        "--int8-quantization-aware",
        action="store_true",
    )
    argument_parser.add_argument(
        "--symmetry-augmentation",
        action="store_true",
    )
    argument_parser.add_argument(
        "--game-pair-size",
        type=int,
        default=DEFAULT_GAME_PAIR_SIZE,
    )
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    train_value_head(
        arguments.data,
        arguments.supplemental_data,
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
        arguments.outcome_target_weight,
        arguments.short_value_target_weight,
        arguments.short_value_auxiliary_weight,
        arguments.root_search_target_weight,
        arguments.int8_quantization_aware,
        arguments.symmetry_augmentation,
        arguments.game_pair_size,
    )


if __name__ == "__main__":
    main()
