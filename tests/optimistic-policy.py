import unittest

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from go_model.config import BOARD_SIZE, INPUT_PLANE_COUNT, POLICY_MOVE_COUNT
from go_model.model import MokaOptimisticPolicyNetwork
from go_model.optimistic_policy import (
    calculate_optimistic_policy_loss,
    calculate_policy_ranking_loss,
    evaluate_policy_logits,
)


class OptimisticPolicyTests(unittest.TestCase):
    def test_tiny_head_outputs_every_move(self) -> None:
        model = MokaOptimisticPolicyNetwork()
        features = mx.zeros((2, BOARD_SIZE, BOARD_SIZE, INPUT_PLANE_COUNT))

        policy_logits, values, optimistic_policy_logits = (
            model.get_optimistic_policy_outputs(features)
        )

        self.assertEqual(policy_logits.shape, (2, POLICY_MOVE_COUNT))
        self.assertEqual(values.shape, (2,))
        self.assertEqual(
            optimistic_policy_logits.shape,
            (2, POLICY_MOVE_COUNT),
        )
        optimistic_parameter_count = sum(
            int(parameter.size)
            for name, parameter in tree_flatten(model.parameters())
            if name.startswith("optimistic_policy_")
        )
        self.assertEqual(optimistic_parameter_count, 654)

    def test_zero_head_reproduces_normal_policy(self) -> None:
        model = MokaOptimisticPolicyNetwork()
        features = mx.zeros((2, BOARD_SIZE, BOARD_SIZE, INPUT_PLANE_COUNT))

        policy_logits, _, optimistic_policy_logits = (
            model.get_optimistic_policy_outputs(features)
        )

        np.testing.assert_array_equal(
            np.asarray(policy_logits),
            np.asarray(optimistic_policy_logits),
        )

    def test_loss_ignores_zero_target_moves(self) -> None:
        model = MokaOptimisticPolicyNetwork()
        features = mx.zeros((1, BOARD_SIZE, BOARD_SIZE, INPUT_PLANE_COUNT))
        targets = np.zeros((1, POLICY_MOVE_COUNT), dtype=np.float32)
        targets[0, 0] = 1

        loss = calculate_optimistic_policy_loss(
            model,
            features,
            mx.array(targets),
        )

        self.assertAlmostEqual(float(loss.item()), 0)

    def test_policy_metrics_respect_legal_mask(self) -> None:
        targets = np.zeros((1, POLICY_MOVE_COUNT), dtype=np.float32)
        targets[0, 3] = 1
        logits = np.zeros((1, POLICY_MOVE_COUNT), dtype=np.float32)
        logits[0, 4] = 100

        cross_entropy, top_move_agreement = evaluate_policy_logits(
            logits,
            targets,
        )

        self.assertAlmostEqual(cross_entropy, 0)
        self.assertEqual(top_move_agreement, 1)

    def test_ranking_loss_requires_teacher_top_move_margin(self) -> None:
        targets = np.zeros((1, POLICY_MOVE_COUNT), dtype=np.float32)
        targets[0, 3] = 0.6
        targets[0, 4] = 0.4
        logits = np.zeros((1, POLICY_MOVE_COUNT), dtype=np.float32)
        logits[0, 4] = 1

        loss = calculate_policy_ranking_loss(
            mx.array(logits),
            mx.array(targets),
        )

        self.assertGreater(float(loss.item()), 1)


if __name__ == "__main__":
    unittest.main()
