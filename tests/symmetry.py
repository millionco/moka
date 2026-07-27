import unittest

import numpy as np

from go_model.config import (
    BOARD_AREA,
    BOARD_SIZE,
    BOARD_SYMMETRY_ROTATION_COUNT,
    INPUT_PLANE_COUNT,
)
from go_model.symmetry import (
    apply_batch_board_symmetry,
    apply_batch_spatial_symmetry,
    apply_board_symmetry,
    invert_policy_symmetry,
)


class SymmetryTest(unittest.TestCase):
    def test_policy_and_features_transform_together(self) -> None:
        features = np.zeros(
            (BOARD_SIZE, BOARD_SIZE, INPUT_PLANE_COUNT),
            dtype=np.float32,
        )
        policy = np.zeros(BOARD_AREA + 1, dtype=np.float32)
        features[1, 2, 0] = 1
        policy[1 * BOARD_SIZE + 2] = 0.75
        policy[BOARD_AREA] = 0.25

        transformed_features, transformed_policy = apply_board_symmetry(
            features,
            policy,
            rotation_count=1,
            should_flip=True,
        )
        transformed_move = int(np.argmax(transformed_policy[:BOARD_AREA]))
        transformed_row, transformed_column = divmod(
            transformed_move,
            BOARD_SIZE,
        )
        self.assertEqual(
            transformed_features[transformed_row, transformed_column, 0],
            1,
        )
        self.assertEqual(transformed_policy[transformed_move], 0.75)
        self.assertEqual(transformed_policy[BOARD_AREA], 0.25)

    def test_batch_symmetry_matches_individual_transformations(self) -> None:
        features = np.stack(
            [
                np.arange(9 * 9 * 2, dtype=np.float32).reshape(9, 9, 2),
                np.arange(9 * 9 * 2, dtype=np.float32).reshape(9, 9, 2) + 1,
                np.arange(9 * 9 * 2, dtype=np.float32).reshape(9, 9, 2) + 2,
            ]
        )
        policies = np.stack(
            [
                np.arange(82, dtype=np.float32),
                np.arange(82, dtype=np.float32) + 1,
                np.arange(82, dtype=np.float32) + 2,
            ]
        )
        rotation_counts = np.asarray([0, 1, 3], dtype=np.int32)
        should_flip = np.asarray([False, True, False])
        batch_features, batch_policies = apply_batch_board_symmetry(
            features,
            policies,
            rotation_counts,
            should_flip,
        )

        for sample_index in range(len(features)):
            expected_features, expected_policy = apply_board_symmetry(
                features[sample_index],
                policies[sample_index],
                int(rotation_counts[sample_index]),
                bool(should_flip[sample_index]),
            )
            np.testing.assert_array_equal(
                batch_features[sample_index],
                expected_features,
            )
            np.testing.assert_array_equal(
                batch_policies[sample_index],
                expected_policy,
            )

    def test_policy_symmetry_inverse_restores_policy(self) -> None:
        features = np.zeros(
            (BOARD_SIZE, BOARD_SIZE, INPUT_PLANE_COUNT),
            dtype=np.float32,
        )
        policy = np.arange(BOARD_AREA + 1, dtype=np.float32)

        for rotation_count in range(BOARD_SYMMETRY_ROTATION_COUNT):
            for should_flip in (False, True):
                _, transformed_policy = apply_board_symmetry(
                    features,
                    policy,
                    rotation_count,
                    should_flip,
                )
                restored_policy = invert_policy_symmetry(
                    transformed_policy,
                    rotation_count,
                    should_flip,
                )
                np.testing.assert_array_equal(restored_policy, policy)

    def test_spatial_targets_follow_feature_symmetry(self) -> None:
        spatial_targets = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        spatial_targets[0, 1, 2] = 1
        spatial_targets[1, 3, 4] = -1
        spatial_targets[2, 7, 6] = 0.5
        features = spatial_targets[:, :, :, None]
        policies = np.zeros((3, BOARD_AREA + 1), dtype=np.float32)
        rotation_counts = np.asarray([0, 1, 3], dtype=np.int32)
        should_flip = np.asarray([False, True, False])
        transformed_features, _ = apply_batch_board_symmetry(
            features,
            policies,
            rotation_counts,
            should_flip,
        )
        transformed_targets = apply_batch_spatial_symmetry(
            spatial_targets,
            rotation_counts,
            should_flip,
        )

        np.testing.assert_array_equal(
            transformed_targets,
            transformed_features[:, :, :, 0],
        )


if __name__ == "__main__":
    unittest.main()
