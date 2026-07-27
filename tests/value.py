import unittest

import numpy as np

from go_model.config import POLICY_MOVE_COUNT
from go_model.value import create_child_value_pairs


class ChildValuePairTest(unittest.TestCase):
    def test_pairs_rank_lowest_child_value_as_root_preference(self) -> None:
        q_values = np.zeros((2, POLICY_MOVE_COUNT), dtype=np.float32)
        q_values[0, :3] = [0.2, 0.5, -0.4]
        q_values[1, :2] = [0.3, -0.7]
        dataset = {
            "child_values": np.asarray(
                [0.4, -0.6, 0.1, -0.2, -0.25],
                dtype=np.float32,
            ),
            "child_weights": np.asarray(
                [4, 8, 2, 3, 6],
                dtype=np.float32,
            ),
            "child_game_ids": np.asarray([2, 2, 2, 3, 3]),
            "q_values": q_values,
        }

        (
            preferred_indexes,
            alternative_indexes,
            target_value_gaps,
            pair_weights,
            pair_game_ids,
        ) = create_child_value_pairs(dataset)

        np.testing.assert_array_equal(preferred_indexes, [1, 1])
        np.testing.assert_array_equal(alternative_indexes, [0, 2])
        np.testing.assert_allclose(target_value_gaps, [1.0, 0.7])
        np.testing.assert_allclose(
            pair_weights,
            [np.sqrt(32), np.sqrt(16)],
        )
        np.testing.assert_array_equal(pair_game_ids, [2, 2])


if __name__ == "__main__":
    unittest.main()
