import unittest

import numpy as np

from go_model.blunder import (
    calculate_blunder_risk_features,
    calculate_blunder_risk_scores,
    normalize_legal_policies,
)
from go_model.config import (
    BLUNDER_RISK_FEATURE_MEANS,
    BOARD_SIZE,
    INPUT_PLANE_COUNT,
    POLICY_MOVE_COUNT,
)
from go_model.symmetry_distill import (
    create_ensemble_symmetry_consensus_targets,
    create_symmetry_consensus_targets,
)


class BlunderRiskTests(unittest.TestCase):
    def test_features_and_scores_are_finite(self) -> None:
        random_generator = np.random.default_rng(11)
        policies = random_generator.random(
            (2, 8, POLICY_MOVE_COUNT),
            dtype=np.float32,
        )
        policies /= np.sum(policies, axis=2, keepdims=True)
        values = random_generator.uniform(-1, 1, size=(2, 8))
        features = np.zeros(
            (2, BOARD_SIZE, BOARD_SIZE, INPUT_PLANE_COUNT),
            dtype=np.float32,
        )
        risk_features = calculate_blunder_risk_features(
            policies,
            values,
            features,
        )
        scores = calculate_blunder_risk_scores(risk_features)

        self.assertEqual(
            risk_features.shape,
            (2, len(BLUNDER_RISK_FEATURE_MEANS)),
        )
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(np.all(np.isfinite(risk_features)))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))

    def test_legal_policy_normalization_masks_stones_and_ko(self) -> None:
        policies = np.ones(
            (1, 8, POLICY_MOVE_COUNT),
            dtype=np.float32,
        )
        features = np.zeros(
            (1, BOARD_SIZE, BOARD_SIZE, INPUT_PLANE_COUNT),
            dtype=np.float32,
        )
        features[0, 0, 0, 0] = 1
        features[0, 0, 1, 6] = 1
        normalized_policies = normalize_legal_policies(
            policies,
            features,
        )

        self.assertEqual(float(normalized_policies[0, 0, 0]), 0)
        self.assertEqual(float(normalized_policies[0, 0, 1]), 0)
        np.testing.assert_allclose(
            np.sum(normalized_policies, axis=2),
            1,
            atol=np.finfo(np.float32).eps,
        )

    def test_symmetry_consensus_aggregates_policies_and_values(self) -> None:
        policies = np.zeros(
            (1, 8, POLICY_MOVE_COUNT),
            dtype=np.float32,
        )
        policies[:, :, 0] = 0.75
        policies[:, :, 1] = 0.25
        values = np.arange(8, dtype=np.float32).reshape(1, 8)

        consensus_policies, consensus_values = (
            create_symmetry_consensus_targets(policies, values)
        )

        np.testing.assert_allclose(
            consensus_policies[0, :2],
            np.asarray([0.75, 0.25]),
        )
        self.assertAlmostEqual(float(consensus_values[0]), 3.5)

    def test_symmetry_consensus_aggregates_checkpoint_ensemble(self) -> None:
        first_policies = np.zeros(
            (1, 8, POLICY_MOVE_COUNT),
            dtype=np.float32,
        )
        second_policies = np.zeros_like(first_policies)
        first_policies[:, :, 0] = 0.75
        first_policies[:, :, 1] = 0.25
        second_policies[:, :, 0] = 0.25
        second_policies[:, :, 1] = 0.75
        first_values = np.ones((1, 8), dtype=np.float32)
        second_values = np.full((1, 8), -0.5, dtype=np.float32)

        consensus_policies, consensus_values = (
            create_ensemble_symmetry_consensus_targets(
                [first_policies, second_policies],
                [first_values, second_values],
            )
        )

        np.testing.assert_allclose(
            consensus_policies[0, :2],
            np.asarray([0.5, 0.5]),
        )
        self.assertAlmostEqual(float(consensus_values[0]), 0.25)

    def test_symmetry_consensus_rejects_empty_checkpoint_ensemble(self) -> None:
        with self.assertRaises(ValueError):
            create_ensemble_symmetry_consensus_targets([], [])


if __name__ == "__main__":
    unittest.main()
