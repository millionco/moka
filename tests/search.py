import unittest

import numpy as np

from go_model.board import GameState, play_move
from go_model.config import POLICY_MOVE_COUNT
from go_model.search import (
    MokaSearchSession,
    SearchNode,
    expand_node_with_evaluation,
    run_search_simulations,
)


class UniformEvaluator:
    def evaluate(self, game_state: GameState) -> tuple[np.ndarray, float]:
        return np.full(POLICY_MOVE_COUNT, 1 / POLICY_MOVE_COUNT), 0.0

    def evaluate_batch(
        self,
        game_states: list[GameState],
    ) -> list[tuple[np.ndarray, float]]:
        return [self.evaluate(game_state) for game_state in game_states]


class SearchTest(unittest.TestCase):
    def test_batched_search_preserves_requested_visit_count(self) -> None:
        evaluator = UniformEvaluator()
        root = SearchNode(GameState(), 1)

        run_search_simulations(root, evaluator, 17)

        self.assertEqual(root.visit_count, 17)
        self.assertTrue(root.children)

    def test_search_session_reuses_opponent_reply_subtree(self) -> None:
        evaluator = UniformEvaluator()
        search_session = MokaSearchSession(evaluator)
        starting_state = GameState()
        root = search_session.align_root(starting_state)
        first_state = play_move(starting_state, 0)
        self.assertIsNotNone(first_state)
        first_child = SearchNode(first_state, 1)
        root.children[0] = first_child
        second_state = play_move(first_state, 1)
        self.assertIsNotNone(second_state)
        second_child = SearchNode(second_state, 1)
        first_child.children[1] = second_child
        search_session.root = first_child

        aligned_root = search_session.align_root(second_state)

        self.assertIs(aligned_root, second_child)

    def test_opponent_branch_pruning_keeps_highest_policy_moves(self) -> None:
        starting_state = GameState()
        opponent_state = play_move(starting_state, 0)
        self.assertIsNotNone(opponent_state)
        policy = np.arange(POLICY_MOVE_COUNT, dtype=np.float32)
        node = SearchNode(opponent_state, 1)

        expand_node_with_evaluation(
            node,
            policy,
            starting_state.next_color,
            3,
        )

        self.assertEqual(
            list(node.children),
            [POLICY_MOVE_COUNT - 2, POLICY_MOVE_COUNT - 3, POLICY_MOVE_COUNT - 4],
        )

    def test_opponent_branch_pruning_preserves_root_branching(self) -> None:
        starting_state = GameState()
        policy = np.arange(POLICY_MOVE_COUNT, dtype=np.float32)
        node = SearchNode(starting_state, 1)

        expand_node_with_evaluation(
            node,
            policy,
            starting_state.next_color,
            1,
        )

        self.assertGreater(len(node.children), 1)


if __name__ == "__main__":
    unittest.main()
