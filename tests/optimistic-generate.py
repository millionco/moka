import unittest

import numpy as np

from go_model.board import GameState
from go_model.optimistic_generate import select_moka_game_states


class OptimisticGenerationTests(unittest.TestCase):
    def test_selects_only_moka_turns_for_each_color(self) -> None:
        histories = [
            [GameState(next_color=1), GameState(next_color=-1)],
            [GameState(next_color=1), GameState(next_color=-1)],
        ]

        game_states, game_ids = select_moka_game_states(histories)

        self.assertEqual([game_state.next_color for game_state in game_states], [1, -1])
        np.testing.assert_array_equal(game_ids, [0, 1])


if __name__ == "__main__":
    unittest.main()
