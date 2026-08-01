import unittest

import numpy as np

from go_model.board import (
    GameState,
    get_area_score,
    get_group,
    get_legal_move_states,
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

    def test_legal_move_states_match_individual_play(self) -> None:
        game_state = GameState()

        for _ in range(BOARD_AREA):
            legal_move_states = get_legal_move_states(game_state)
            expected_move_states = [
                (move, play_move(game_state, move))
                for move in range(BOARD_AREA + 1)
            ]
            expected_move_states = [
                (move, next_state)
                for move, next_state in expected_move_states
                if next_state is not None
            ]

            self.assertEqual(
                [move for move, _ in legal_move_states],
                [move for move, _ in expected_move_states],
            )

            for (_, next_state), (_, expected_state) in zip(
                legal_move_states,
                expected_move_states,
                strict=True,
            ):
                self.assertTrue(
                    np.array_equal(next_state.board, expected_state.board)
                )
                self.assertEqual(next_state.ko_move, expected_state.ko_move)
                self.assertEqual(
                    next_state.consecutive_pass_count,
                    expected_state.consecutive_pass_count,
                )
                self.assertEqual(next_state.move_count, expected_state.move_count)
                self.assertEqual(
                    next_state.move_history,
                    expected_state.move_history,
                )
                self.assertEqual(next_state.next_color, expected_state.next_color)

            non_pass_move_states = [
                move_and_state
                for move_and_state in expected_move_states
                if move_and_state[0] != BOARD_AREA
            ]

            if not non_pass_move_states:
                break

            selected_index = (
                game_state.move_count * BOARD_SIZE + BOARD_SIZE // 2
            ) % len(non_pass_move_states)
            game_state = non_pass_move_states[selected_index][1]

    def test_positional_superko_rejects_a_repeated_board(self) -> None:
        initial_state = GameState()
        played_state = play_move(initial_state, 0)
        self.assertIsNotNone(played_state)

        repeated_position_state = GameState(
            position_history=played_state.position_history,
        )

        self.assertIsNone(play_move(repeated_position_state, 0))

    def test_pass_remains_legal_when_the_board_has_appeared(self) -> None:
        initial_state = GameState()
        passed_state = play_move(initial_state, BOARD_AREA)

        self.assertIsNotNone(passed_state)

    def test_position_history_is_persistent(self) -> None:
        initial_state = GameState()
        first_state = play_move(initial_state, 0)
        self.assertIsNotNone(first_state)
        second_state = play_move(first_state, 1)
        self.assertIsNotNone(second_state)

        self.assertIsNotNone(initial_state.position_history)
        self.assertIsNotNone(first_state.position_history)
        self.assertIsNotNone(second_state.position_history)
        self.assertFalse(
            initial_state.position_history.contains(
                first_state.board.tobytes(),
            )
        )
        self.assertTrue(
            second_state.position_history.contains(
                first_state.board.tobytes(),
            )
        )

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
