import unittest

import numpy as np

from go_model.benson import (
    calculate_benson_score_bounds,
    get_benson_pass_alive_stones,
    get_benson_proven_winner,
)


class BensonTest(unittest.TestCase):
    def test_two_eye_group_is_pass_alive(self) -> None:
        board = np.zeros((9, 9), dtype=np.int8)
        board[2, 2:7] = 1
        board[4, 2:7] = 1
        board[3, [2, 4, 6]] = 1
        expected_stones = set(np.flatnonzero(board.reshape(-1) == 1))

        pass_alive_stones = get_benson_pass_alive_stones(board, 1)

        self.assertEqual(pass_alive_stones, expected_stones)

    def test_single_eye_group_remains_unknown(self) -> None:
        board = np.zeros((9, 9), dtype=np.int8)
        board[2, 2:5] = 1
        board[4, 2:5] = 1
        board[3, [2, 4]] = 1

        pass_alive_stones = get_benson_pass_alive_stones(board, 1)

        self.assertEqual(pass_alive_stones, set())

    def test_score_bounds_use_only_certified_points(self) -> None:
        self.assertEqual(calculate_benson_score_bounds(45, 0), (2, 74))
        self.assertEqual(calculate_benson_score_bounds(0, 37), (-88, 0))

    def test_unsettled_board_has_no_proven_winner(self) -> None:
        board = np.zeros((9, 9), dtype=np.int8)

        self.assertEqual(get_benson_proven_winner(board), 0)


if __name__ == "__main__":
    unittest.main()
