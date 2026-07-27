import unittest

import numpy as np

from go_model.config import BOARD_AREA, KATAGO_SIMPLE_AREA_RULES
from go_model.search_generate import (
    convert_parent_value_to_child,
    coordinate_to_move,
    create_analysis_query,
    move_to_coordinate,
    select_analysis_turns,
)


class SearchGenerationTests(unittest.TestCase):
    def test_move_coordinates_round_trip(self) -> None:
        for move in range(BOARD_AREA + 1):
            self.assertEqual(
                coordinate_to_move(move_to_coordinate(move)),
                move,
            )

    def test_gtp_coordinates_use_bottom_origin(self) -> None:
        self.assertEqual(coordinate_to_move("A9"), 0)
        self.assertEqual(coordinate_to_move("J9"), 8)
        self.assertEqual(coordinate_to_move("A1"), 72)
        self.assertEqual(coordinate_to_move("J1"), 80)

    def test_query_preserves_move_order_and_requested_turns(self) -> None:
        moves = [0, 1, BOARD_AREA, 9]
        query = create_analysis_query(7, moves, 32)

        self.assertEqual(query["id"], "7")
        self.assertEqual(
            query["moves"],
            [
                ["B", "(0,0)"],
                ["W", "(1,0)"],
                ["B", "pass"],
                ["W", "(0,1)"],
            ],
        )
        self.assertEqual(query["analyzeTurns"], [0, 1, 2, 3])
        self.assertEqual(query["maxVisits"], 32)
        self.assertEqual(query["rules"], KATAGO_SIMPLE_AREA_RULES)

    def test_selective_reanalysis_mixes_uniform_and_surprising_turns(self) -> None:
        selected_turns = select_analysis_turns(
            [0, 1, 2, 3, 4, 5, 6, 7],
            0.5,
            0.5,
            np.random.default_rng(3),
        )

        self.assertEqual(len(selected_turns), 4)
        self.assertEqual(len(set(selected_turns)), 4)
        self.assertEqual(selected_turns, sorted(selected_turns))
        self.assertTrue(6 in selected_turns or 7 in selected_turns)

    def test_selective_reanalysis_respects_eligible_turns(self) -> None:
        selected_turns = select_analysis_turns(
            [0, 1, 2, 3, 4, 5, 6, 7],
            0.5,
            0.5,
            np.random.default_rng(5),
            [2, 3, 4, 5],
        )

        self.assertEqual(len(selected_turns), 2)
        self.assertTrue(set(selected_turns).issubset({2, 3, 4, 5}))

    def test_parent_value_is_negated_for_child_state(self) -> None:
        self.assertEqual(
            convert_parent_value_to_child(0.75),
            -0.75,
        )


if __name__ == "__main__":
    unittest.main()
