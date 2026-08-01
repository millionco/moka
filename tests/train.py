import unittest

import mlx.core as mx
import numpy as np

from go_model.config import (
    BOARD_SIZE,
    INPUT_PLANE_COUNT,
    POLICY_MOVE_COUNT,
)
from go_model.model import MokaNetwork
from go_model.train import evaluate


class TrainTest(unittest.TestCase):
    def test_evaluate_respects_sample_weights(self) -> None:
        model = MokaNetwork()
        features = np.zeros(
            (2, BOARD_SIZE, BOARD_SIZE, INPUT_PLANE_COUNT),
            dtype=np.float32,
        )
        policy_logits, values = model(mx.array(features[:1]))
        mx.eval(policy_logits, values)
        predicted_move = int(np.argmax(np.asarray(policy_logits)[0]))
        other_move = (predicted_move + 1) % POLICY_MOVE_COUNT
        policies = np.zeros((2, POLICY_MOVE_COUNT), dtype=np.float32)
        policies[0, predicted_move] = 1
        policies[1, other_move] = 1
        predicted_value = float(np.asarray(values)[0])
        value_targets = np.array(
            [predicted_value, -predicted_value],
            dtype=np.float32,
        )

        _, move_agreement, value_error = evaluate(
            model,
            features,
            policies,
            value_targets,
            1,
            np.array([1, 0], dtype=np.float32),
        )

        self.assertEqual(move_agreement, 1)
        self.assertAlmostEqual(value_error, 0)


if __name__ == "__main__":
    unittest.main()
