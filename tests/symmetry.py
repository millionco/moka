import unittest

import numpy as np

from go_model.config import BOARD_AREA, BOARD_SIZE, INPUT_PLANE_COUNT
from go_model.symmetry import apply_board_symmetry


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


if __name__ == "__main__":
    unittest.main()
