import unittest

import numpy as np

from go_model.config import BOARD_AREA, KATAGO_SIMPLE_AREA_RULES
from go_model.board import GameState
from go_model.search_generate import (
    convert_parent_value_to_child,
    coordinate_to_move,
    create_analysis_query,
    extract_analysis_targets,
    extract_auxiliary_targets,
    get_eligible_analysis_turns,
    is_moka_turn,
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
        self.assertNotIn("includeOwnership", query)

    def test_query_can_request_auxiliary_targets(self) -> None:
        query = create_analysis_query(
            7,
            [0, 1],
            32,
            include_auxiliary_targets=True,
        )

        self.assertTrue(query["includeOwnership"])

    def test_auxiliary_targets_preserve_board_order_and_score(self) -> None:
        ownership = np.linspace(-1, 1, BOARD_AREA, dtype=np.float32)
        extracted_ownership, extracted_score = extract_auxiliary_targets(
            {
                "ownership": ownership.tolist(),
                "rootInfo": {"scoreLead": 12.5},
            }
        )

        np.testing.assert_allclose(
            extracted_ownership,
            ownership.reshape(9, 9),
        )
        self.assertEqual(extracted_score, 12.5)

    def test_analysis_targets_normalize_visits_and_child_values(self) -> None:
        targets = extract_analysis_targets(
            {
                "moveInfos": [
                    {
                        "move": "A9",
                        "winrate": 0.8,
                        "edgeVisits": 2,
                    },
                    {
                        "move": "B9",
                        "winrate": 0.4,
                        "edgeVisits": 1,
                    },
                ],
                "rootInfo": {"winrate": 0.75},
            },
            GameState(),
            False,
        )

        self.assertIsNotNone(targets)
        if targets is None:
            return
        self.assertAlmostEqual(float(targets.policy[0]), 2 / 3)
        self.assertAlmostEqual(float(targets.policy[1]), 1 / 3)
        self.assertAlmostEqual(float(targets.value), 0.5)
        self.assertEqual(len(targets.child_features), 2)
        self.assertAlmostEqual(targets.child_values[0], -0.6)
        self.assertAlmostEqual(targets.child_values[1], 0.2)

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

    def test_moka_turn_alternates_with_game_color(self) -> None:
        black_turn = GameState(next_color=1)
        white_turn = GameState(next_color=-1)

        self.assertTrue(is_moka_turn(0, black_turn))
        self.assertFalse(is_moka_turn(0, white_turn))
        self.assertFalse(is_moka_turn(1, black_turn))
        self.assertTrue(is_moka_turn(1, white_turn))

    def test_eligible_turns_can_select_only_moka_decisions(self) -> None:
        game_state_history = [
            GameState(next_color=1),
            GameState(next_color=-1),
            GameState(next_color=1),
            GameState(next_color=-1),
        ]

        self.assertEqual(
            get_eligible_analysis_turns(
                game_state_history,
                0,
                False,
                True,
            ),
            [0, 2],
        )
        self.assertIsNone(
            get_eligible_analysis_turns(
                game_state_history,
                0,
                False,
                False,
            )
        )


if __name__ == "__main__":
    unittest.main()
