import unittest
from pathlib import Path

import numpy as np

from go_model.action_value_prior import (
    ActionValueDataset,
    create_rank_pairs,
    fit_action_value_prior,
)
from go_model.config import POLICY_MOVE_COUNT


class ActionValuePriorTest(unittest.TestCase):
    def test_rank_pairs_orient_better_action_first(self) -> None:
        representations = np.zeros((1, POLICY_MOVE_COUNT, 3), dtype=np.float32)
        representations[0, 0, 0] = 1
        representations[0, 1, 0] = -1
        q_values = np.zeros((1, POLICY_MOVE_COUNT), dtype=np.float32)
        q_values[0, 0] = 0.4
        q_values[0, 1] = 0.1
        q_weights = np.zeros_like(q_values)
        q_weights[0, :2] = 8
        policies = np.zeros_like(q_values)
        policies[0, 0] = 0.75
        policies[0, 1] = 0.25
        dataset = ActionValueDataset(
            path=Path("synthetic.npz"),
            representations=representations,
            q_values=q_values,
            q_weights=q_weights,
            policies=policies,
            split_buckets=np.array([2]),
        )

        pair_features, pair_targets, policy_orders = create_rank_pairs(
            dataset,
            {2},
        )

        np.testing.assert_array_equal(pair_features, [[2, 0, 0]])
        np.testing.assert_allclose(pair_targets, [0.3])
        np.testing.assert_array_equal(policy_orders, [True])

        weights = fit_action_value_prior([dataset])

        self.assertGreater(float(pair_features[0] @ weights), 0)


if __name__ == "__main__":
    unittest.main()
