import unittest

import numpy as np

from go_model.config import (
    BOARD_AREA,
    BOARD_SIZE,
    BOARD_SYMMETRY_ROTATION_COUNT,
    INPUT_PLANE_COUNT,
)
from go_model.symmetry import (
    aggregate_symmetry_policies,
    aggregate_symmetry_values,
    apply_batch_board_symmetry,
    apply_batch_spatial_symmetry,
    apply_board_symmetry,
    invert_policy_symmetry,
)


class SymmetryTest(unittest.TestCase):
    def test_geometric_policy_aggregation_rewards_symmetry_consensus(
        self,
    ) -> None:
        policies = [
            np.asarray([0.9, 0.1], dtype=np.float32),
            np.asarray([0.01, 0.99], dtype=np.float32),
        ]

        arithmetic_policy = aggregate_symmetry_policies(policies, 0)
        geometric_policy = aggregate_symmetry_policies(policies, 1)

        self.assertAlmostEqual(float(np.sum(arithmetic_policy)), 1)
        self.assertAlmostEqual(float(np.sum(geometric_policy)), 1)
        self.assertLess(geometric_policy[0], arithmetic_policy[0])
        self.assertGreater(geometric_policy[1], arithmetic_policy[1])

    def test_geometric_policy_aggregation_rejects_invalid_weight(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_symmetry_policies(
                [np.asarray([0.5, 0.5], dtype=np.float32)],
                1.1,
            )

    def test_trimmed_policy_aggregation_suppresses_symmetry_outlier(
        self,
    ) -> None:
        policies = [
            np.asarray([0.99, 0.01], dtype=np.float32),
            np.asarray([0.5, 0.5], dtype=np.float32),
            np.asarray([0.5, 0.5], dtype=np.float32),
        ]

        arithmetic_policy = aggregate_symmetry_policies(policies, 0)
        trimmed_policy = aggregate_symmetry_policies(policies, 0, 1)

        self.assertGreater(arithmetic_policy[0], trimmed_policy[0])
        self.assertTrue(
            np.allclose(
                trimmed_policy,
                np.asarray([0.5, 0.5], dtype=np.float32),
            )
        )

    def test_trimmed_value_aggregation_suppresses_symmetry_outliers(
        self,
    ) -> None:
        values = np.asarray([-1, 0.4, 0.5, 1], dtype=np.float32)

        arithmetic_value = aggregate_symmetry_values(values, 0)
        trimmed_value = aggregate_symmetry_values(values, 1)

        self.assertAlmostEqual(arithmetic_value, 0.225)
        self.assertAlmostEqual(trimmed_value, 0.45)

    def test_rank_policy_aggregation_rewards_orientation_votes(
        self,
    ) -> None:
        policies = [
            np.asarray([0.99, 0.01], dtype=np.float32),
            np.asarray([0.49, 0.51], dtype=np.float32),
            np.asarray([0.49, 0.51], dtype=np.float32),
        ]

        arithmetic_policy = aggregate_symmetry_policies(policies, 0, 0, 0, 1)
        rank_policy = aggregate_symmetry_policies(policies, 0, 0, 1, 1)

        self.assertGreater(arithmetic_policy[0], arithmetic_policy[1])
        self.assertGreater(rank_policy[1], rank_policy[0])

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
