import unittest

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from go_model.config import BOARD_SIZE, INPUT_PLANE_COUNT, POLICY_MOVE_COUNT
from go_model.model import MokaUncertaintyNetwork
from go_model.uncertainty import calculate_uncertainty_metrics


class UncertaintyTests(unittest.TestCase):
    def test_head_outputs_log_uncertainty(self) -> None:
        model = MokaUncertaintyNetwork()
        features = mx.zeros((2, BOARD_SIZE, BOARD_SIZE, INPUT_PLANE_COUNT))

        policy_logits, values, log_uncertainties = (
            model.get_uncertainty_outputs(features)
        )

        self.assertEqual(policy_logits.shape, (2, POLICY_MOVE_COUNT))
        self.assertEqual(values.shape, (2,))
        self.assertEqual(log_uncertainties.shape, (2,))
        parameter_count = sum(
            int(parameter.size)
            for name, parameter in tree_flatten(model.parameters())
            if name.startswith("uncertainty_output.")
        )
        self.assertEqual(parameter_count, 65)

    def test_perfect_prediction_explains_all_error(self) -> None:
        targets = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        predictions = np.log(targets + 1e-4)

        _, explained_error, correlation = calculate_uncertainty_metrics(
            predictions,
            targets,
            float(np.mean(predictions)),
        )

        self.assertAlmostEqual(explained_error, 1)
        self.assertAlmostEqual(correlation, 1)


if __name__ == "__main__":
    unittest.main()
