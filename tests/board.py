import unittest

import numpy as np

from go_model.board import (
    GameState,
    get_area_score,
    get_group,
    get_legal_moves,
    play_move,
    remove_dead_stones,
)
from go_model.config import BOARD_AREA, BOARD_SIZE


class BoardTest(unittest.TestCase):
    def test_capture_removes_surrounded_stone(self) -> None:
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board[1, 1] = -1
        board[0, 1] = 1
        board[1, 0] = 1
        board[2, 1] = 1
        game_state = GameState(board=board)
        next_state = play_move(game_state, 1 * BOARD_SIZE + 2)
        self.assertIsNotNone(next_state)
        self.assertEqual(next_state.board[1, 1], 0)

    def test_suicide_is_illegal(self) -> None:
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board[0, 1] = -1
        board[1, 0] = -1
        game_state = GameState(board=board)
        self.assertIsNone(play_move(game_state, 0))

    def test_pass_is_legal(self) -> None:
        self.assertIn(BOARD_AREA, get_legal_moves(GameState()))

    def test_empty_board_score_is_komi(self) -> None:
        self.assertEqual(get_area_score(GameState()), -7)

    def test_connected_group_has_shared_liberties(self) -> None:
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board[4, 4] = 1
        board[4, 5] = 1
        stones, liberties = get_group(board, 4 * BOARD_SIZE + 4)
        self.assertEqual(len(stones), 2)
        self.assertEqual(len(liberties), 6)

        corner_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        corner_board[0, 0] = 1
        corner_board[1, 0] = 1
        _, corner_liberties = get_group(corner_board, 0)
        self.assertEqual(len(corner_liberties), 3)

    def test_black_border_encloses_the_board(self) -> None:
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board[0, :] = 1
        board[-1, :] = 1
        board[:, 0] = 1
        board[:, -1] = 1
        self.assertEqual(get_area_score(GameState(board=board)), 74)

    def test_dead_stones_can_be_removed_before_scoring(self) -> None:
        board = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        dead_move = 4 * BOARD_SIZE + 4
        board[4, 4] = -1
        game_state = GameState(board=board)

        adjudicated_state = remove_dead_stones(game_state, [dead_move])

        self.assertEqual(game_state.board[4, 4], -1)
        self.assertEqual(adjudicated_state.board[4, 4], 0)
        self.assertEqual(get_area_score(adjudicated_state), 74)

    def test_dead_stone_removal_rejects_moves_outside_the_board(self) -> None:
        with self.assertRaises(ValueError):
            remove_dead_stones(GameState(), [BOARD_AREA])


if __name__ == "__main__":
    unittest.main()
