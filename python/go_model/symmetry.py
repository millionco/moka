import numpy as np

from go_model.config import BOARD_AREA, BOARD_SIZE


def apply_board_symmetry(
    features: np.ndarray,
    policy: np.ndarray,
    rotation_count: int,
    should_flip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    transformed_features = np.rot90(features, rotation_count, axes=(0, 1))
    board_policy = policy[:BOARD_AREA].reshape(BOARD_SIZE, BOARD_SIZE)
    transformed_board_policy = np.rot90(board_policy, rotation_count)

    if should_flip:
        transformed_features = np.flip(transformed_features, axis=1)
        transformed_board_policy = np.flip(transformed_board_policy, axis=1)

    transformed_policy = policy.copy()
    transformed_policy[:BOARD_AREA] = transformed_board_policy.reshape(BOARD_AREA)
    return transformed_features.copy(), transformed_policy


def invert_policy_symmetry(
    policy: np.ndarray,
    rotation_count: int,
    should_flip: bool,
) -> np.ndarray:
    board_policy = policy[:BOARD_AREA].reshape(BOARD_SIZE, BOARD_SIZE)

    if should_flip:
        board_policy = np.flip(board_policy, axis=1)

    board_policy = np.rot90(board_policy, -rotation_count)
    inverted_policy = policy.copy()
    inverted_policy[:BOARD_AREA] = board_policy.reshape(BOARD_AREA)
    return inverted_policy


def aggregate_symmetry_policies(
    policies: list[np.ndarray],
    geometric_policy_weight: float,
    trimmed_policy_weight: float = 0,
) -> np.ndarray:
    if not 0 <= geometric_policy_weight <= 1:
        raise ValueError("Geometric policy weight must be between zero and one.")

    if not 0 <= trimmed_policy_weight <= 1:
        raise ValueError("Trimmed policy weight must be between zero and one.")

    policy_array = np.asarray(policies, dtype=np.float32)
    arithmetic_policy = np.mean(policy_array, axis=0)

    if geometric_policy_weight > 0:
        mean_log_policy = np.mean(
            np.log(np.maximum(policy_array, np.finfo(np.float32).tiny)),
            axis=0,
        )
        geometric_policy = np.exp(mean_log_policy - np.max(mean_log_policy))
        geometric_policy /= np.sum(geometric_policy)
        blended_policy = (
            (1 - geometric_policy_weight) * arithmetic_policy
            + geometric_policy_weight * geometric_policy
        )
    else:
        blended_policy = arithmetic_policy

    if trimmed_policy_weight > 0:
        if policy_array.shape[0] < 3:
            raise ValueError("Trimmed policy aggregation requires three policies.")

        sorted_policy = np.sort(policy_array, axis=0)
        trimmed_policy = np.mean(sorted_policy[1:-1], axis=0)
        trimmed_policy /= np.sum(trimmed_policy)
        blended_policy = (
            (1 - trimmed_policy_weight) * blended_policy
            + trimmed_policy_weight * trimmed_policy
        )

    return blended_policy / np.sum(blended_policy)


def apply_batch_board_symmetry(
    features: np.ndarray,
    policies: np.ndarray,
    rotation_counts: np.ndarray,
    should_flip: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transformed_features = np.empty_like(features)
    transformed_policies = np.empty_like(policies)
    board_policies = policies[:, :BOARD_AREA].reshape(
        len(policies),
        BOARD_SIZE,
        BOARD_SIZE,
    )

    for rotation_count in np.unique(rotation_counts):
        for flip_value in np.unique(should_flip):
            symmetry_mask = (rotation_counts == rotation_count) & (
                should_flip == flip_value
            )

            if not np.any(symmetry_mask):
                continue

            symmetry_features = np.rot90(
                features[symmetry_mask],
                int(rotation_count),
                axes=(1, 2),
            )
            symmetry_board_policies = np.rot90(
                board_policies[symmetry_mask],
                int(rotation_count),
                axes=(1, 2),
            )

            if flip_value:
                symmetry_features = np.flip(symmetry_features, axis=2)
                symmetry_board_policies = np.flip(
                    symmetry_board_policies,
                    axis=2,
                )

            transformed_features[symmetry_mask] = symmetry_features
            transformed_policies[symmetry_mask, :BOARD_AREA] = (
                symmetry_board_policies.reshape(-1, BOARD_AREA)
            )
            transformed_policies[symmetry_mask, BOARD_AREA:] = policies[
                symmetry_mask,
                BOARD_AREA:,
            ]

    return transformed_features, transformed_policies


def apply_batch_spatial_symmetry(
    spatial_targets: np.ndarray,
    rotation_counts: np.ndarray,
    should_flip: np.ndarray,
) -> np.ndarray:
    transformed_targets = np.empty_like(spatial_targets)

    for rotation_count in np.unique(rotation_counts):
        for flip_value in np.unique(should_flip):
            symmetry_mask = (rotation_counts == rotation_count) & (
                should_flip == flip_value
            )

            if not np.any(symmetry_mask):
                continue

            symmetry_targets = np.rot90(
                spatial_targets[symmetry_mask],
                int(rotation_count),
                axes=(1, 2),
            )

            if flip_value:
                symmetry_targets = np.flip(symmetry_targets, axis=2)

            transformed_targets[symmetry_mask] = symmetry_targets

    return transformed_targets
