import unittest

import numpy as np

from go_model.board import GameState, play_move
from go_model.collect import (
    calculate_goldilocks_weight,
    select_greedy_rollout_move,
)
from go_model.config import BOARD_AREA


class GoldilocksWeightTests(unittest.TestCase):
    def test_mid_difficulty_position_gets_more_weight(self) -> None:
        medium_difficulty_weight = calculate_goldilocks_weight(0.45)
        easy_position_weight = calculate_goldilocks_weight(0.95)
        overwhelming_position_weight = calculate_goldilocks_weight(0.01)

        self.assertGreater(medium_difficulty_weight, easy_position_weight)
        self.assertGreater(medium_difficulty_weight, overwhelming_position_weight)

    def test_greedy_rollout_respects_legal_moves(self) -> None:
        game_state = play_move(GameState(), 0)
        self.assertIsNotNone(game_state)
        probabilities = np.zeros(BOARD_AREA + 1, dtype=np.float32)
        probabilities[0] = 1
        probabilities[1] = 0.5

        selected_move = select_greedy_rollout_move(game_state, probabilities)

        self.assertEqual(selected_move, 1)


if __name__ == "__main__":
    unittest.main()
