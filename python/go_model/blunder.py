from pathlib import Path

import mlx.core as mx
import numpy as np

from go_model.board import GameState
from go_model.config import (
    BLUNDER_RISK_FEATURE_MEANS,
    BLUNDER_RISK_FEATURE_STDS,
    BLUNDER_RISK_LOGIT_LIMIT,
    BLUNDER_RISK_WEIGHTS,
    BOARD_AREA,
    BOARD_SIZE,
    BOARD_SYMMETRY_REFLECTION_COUNT,
    BOARD_SYMMETRY_ROTATION_COUNT,
    MOKA_CURRENT_PLAYER_FEATURE_INDEX,
    MOKA_KO_FEATURE_INDEX,
    MOKA_OPPONENT_FEATURE_INDEX,
    POLICY_MOVE_COUNT,
    SEARCH_ROOT_SYMMETRY_GEOMETRIC_POLICY_WEIGHT,
)
from go_model.features import encode_moka_features
from go_model.model import (
    MokaNestedNetwork,
    create_moka_network_for_checkpoint,
)
from go_model.symmetry import (
    aggregate_symmetry_policies,
    apply_board_symmetry,
    invert_policy_symmetry,
)


def calculate_policy_entropy(policies: np.ndarray) -> np.ndarray:
    positive_policies = np.maximum(policies, np.finfo(np.float32).tiny)
    return -np.sum(policies * np.log(positive_policies), axis=-1)


def calculate_blunder_risk_features(
    aligned_policies: np.ndarray,
    symmetry_values: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    arithmetic_policies = np.mean(aligned_policies, axis=1)
    geometric_policies = np.stack(
        [
            aggregate_symmetry_policies(
                list(sample_policies),
                1,
            )
            for sample_policies in aligned_policies
        ]
    )
    accepted_policies = np.stack(
        [
            aggregate_symmetry_policies(
                list(sample_policies),
                SEARCH_ROOT_SYMMETRY_GEOMETRIC_POLICY_WEIGHT,
            )
            for sample_policies in aligned_policies
        ]
    )
    policy_divergences = np.sum(
        aligned_policies
        * (
            np.log(
                np.maximum(
                    aligned_policies,
                    np.finfo(np.float32).tiny,
                )
            )
            - np.log(
                np.maximum(
                    arithmetic_policies[:, None, :],
                    np.finfo(np.float32).tiny,
                )
            )
        ),
        axis=-1,
    )
    top_move_votes = np.stack(
        [
            np.bincount(
                np.argmax(sample_policies, axis=1),
                minlength=POLICY_MOVE_COUNT,
            )
            for sample_policies in aligned_policies
        ]
    )
    accepted_ranked_probabilities = np.sort(accepted_policies, axis=1)
    accepted_top_moves = np.argmax(accepted_policies, axis=1)
    arithmetic_top_moves = np.argmax(arithmetic_policies, axis=1)
    geometric_top_moves = np.argmax(geometric_policies, axis=1)
    canonical_top_moves = np.argmax(aligned_policies[:, 0], axis=1)
    current_player_stones = features[
        :, :, :, MOKA_CURRENT_PLAYER_FEATURE_INDEX
    ]
    opponent_stones = features[
        :, :, :, MOKA_OPPONENT_FEATURE_INDEX
    ]
    occupancy = current_player_stones + opponent_stones
    mean_values = np.mean(symmetry_values, axis=1)
    symmetry_count = aligned_policies.shape[1]

    return np.column_stack(
        [
            np.mean(policy_divergences, axis=1),
            np.std(symmetry_values, axis=1),
            np.max(top_move_votes, axis=1) / symmetry_count,
            accepted_ranked_probabilities[:, -1],
            (
                accepted_ranked_probabilities[:, -1]
                - accepted_ranked_probabilities[:, -2]
            ),
            calculate_policy_entropy(accepted_policies),
            np.mean(
                np.abs(
                    aligned_policies - arithmetic_policies[:, None, :]
                ),
                axis=(1, 2),
            ),
            np.mean(occupancy, axis=(1, 2)),
            np.mean(
                current_player_stones - opponent_stones,
                axis=(1, 2),
            ),
            mean_values,
            np.abs(mean_values),
            np.abs(symmetry_values[:, 0] - mean_values),
            canonical_top_moves == accepted_top_moves,
            arithmetic_top_moves == geometric_top_moves,
        ]
    ).astype(np.float32)


def calculate_blunder_risk_scores(
    risk_features: np.ndarray,
) -> np.ndarray:
    feature_means = np.asarray(
        BLUNDER_RISK_FEATURE_MEANS,
        dtype=np.float32,
    )
    feature_stds = np.asarray(
        BLUNDER_RISK_FEATURE_STDS,
        dtype=np.float32,
    )
    weights = np.asarray(BLUNDER_RISK_WEIGHTS, dtype=np.float32)
    standardized_features = (risk_features - feature_means) / feature_stds
    logits = weights[0] + standardized_features @ weights[1:]
    return 1 / (
        1
        + np.exp(
            -np.clip(
                logits,
                -BLUNDER_RISK_LOGIT_LIMIT,
                BLUNDER_RISK_LOGIT_LIMIT,
            )
        )
    )


def normalize_legal_policies(
    aligned_policies: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    empty_points = (
        features[:, :, :, MOKA_CURRENT_PLAYER_FEATURE_INDEX]
        + features[:, :, :, MOKA_OPPONENT_FEATURE_INDEX]
        + features[:, :, :, MOKA_KO_FEATURE_INDEX]
    ) == 0
    legal_moves = np.concatenate(
        [
            empty_points.reshape(-1, BOARD_AREA),
            np.ones((len(features), 1), dtype=bool),
        ],
        axis=1,
    )
    masked_policies = aligned_policies * legal_moves[:, None, :]
    policy_sums = np.sum(masked_policies, axis=2, keepdims=True)
    return masked_policies / np.maximum(
        policy_sums,
        np.finfo(np.float32).tiny,
    )


def evaluate_aligned_symmetry_outputs(
    model: MokaNestedNetwork,
    features: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    empty_policy = np.zeros(POLICY_MOVE_COUNT, dtype=np.float32)
    aligned_policy_batches: list[np.ndarray] = []
    symmetry_value_batches: list[np.ndarray] = []
    symmetry_count = (
        BOARD_SYMMETRY_ROTATION_COUNT
        * BOARD_SYMMETRY_REFLECTION_COUNT
    )

    for batch_start in range(0, len(features), batch_size):
        batch_features = features[batch_start:batch_start + batch_size]
        transformed_features: list[np.ndarray] = []
        symmetry_descriptors: list[tuple[int, bool]] = []

        for sample_features in batch_features:
            for rotation_count in range(BOARD_SYMMETRY_ROTATION_COUNT):
                for should_flip in (False, True):
                    symmetry_features, _ = apply_board_symmetry(
                        sample_features,
                        empty_policy,
                        rotation_count,
                        should_flip,
                    )
                    transformed_features.append(symmetry_features)
                    symmetry_descriptors.append(
                        (rotation_count, should_flip)
                    )

        policy_logits, values = model(
            mx.array(np.stack(transformed_features), dtype=mx.float32)
        )
        mx.eval(policy_logits, values)
        logits = np.asarray(policy_logits)
        maximum_logits = np.max(logits, axis=1, keepdims=True)
        transformed_policies = np.exp(logits - maximum_logits)
        transformed_policies /= np.sum(
            transformed_policies,
            axis=1,
            keepdims=True,
        )
        aligned_policies = np.stack(
            [
                invert_policy_symmetry(
                    transformed_policies[symmetry_index],
                    symmetry_descriptors[symmetry_index][0],
                    symmetry_descriptors[symmetry_index][1],
                )
                for symmetry_index in range(len(transformed_policies))
            ]
        ).reshape(
            len(batch_features),
            symmetry_count,
            POLICY_MOVE_COUNT,
        )
        symmetry_values = np.asarray(values).reshape(
            len(batch_features),
            symmetry_count,
        )
        aligned_policy_batches.append(aligned_policies)
        symmetry_value_batches.append(symmetry_values)

    return (
        (
            np.concatenate(aligned_policy_batches)
            if aligned_policy_batches
            else np.empty(
                (0, symmetry_count, POLICY_MOVE_COUNT),
                dtype=np.float32,
            )
        ),
        (
            np.concatenate(symmetry_value_batches)
            if symmetry_value_batches
            else np.empty((0, symmetry_count), dtype=np.float32)
        ),
    )


def evaluate_blunder_risk_scores(
    model: MokaNestedNetwork,
    features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    aligned_policies, symmetry_values = (
        evaluate_aligned_symmetry_outputs(
            model,
            features,
            batch_size,
        )
    )
    normalized_policies = normalize_legal_policies(
        aligned_policies,
        features,
    )
    return calculate_blunder_risk_scores(
        calculate_blunder_risk_features(
            normalized_policies,
            symmetry_values,
            features,
        )
    )


def calculate_game_blunder_risks(
    checkpoint_path: Path,
    game_state_histories: list[list[GameState]],
    eligible_turns: list[list[int] | None],
    batch_size: int,
) -> list[list[float]]:
    selected_game_turns = [
        (game_id, turn_number)
        for game_id, game_state_history in enumerate(game_state_histories)
        for turn_number in (
            eligible_turns[game_id]
            if eligible_turns[game_id] is not None
            else list(range(len(game_state_history)))
        )
    ]
    selected_features = np.asarray(
        [
            encode_moka_features(
                game_state_histories[game_id][turn_number]
            )
            for game_id, turn_number in selected_game_turns
        ],
        dtype=np.float32,
    )
    model = create_moka_network_for_checkpoint(str(checkpoint_path))
    model.load_weights(str(checkpoint_path))
    model.eval()
    selected_scores = evaluate_blunder_risk_scores(
        model,
        selected_features,
        batch_size,
    )
    game_risks = [
        [0.0 for _ in game_state_history]
        for game_state_history in game_state_histories
    ]

    for (game_id, turn_number), score in zip(
        selected_game_turns,
        selected_scores,
        strict=True,
    ):
        game_risks[game_id][turn_number] = float(score)

    return game_risks
