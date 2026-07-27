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

