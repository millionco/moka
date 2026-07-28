import unittest

import numpy as np

from go_model.arena import get_search_simulation_count
from go_model.board import GameState, play_move
from go_model.config import POLICY_MOVE_COUNT
from go_model.search import (
    MokaSearchSession,
    SearchNode,
    expand_node_with_evaluation,
    prune_root_children,
    resolve_first_play_urgency_reduction,
    run_search_simulations,
    select_child,
)
from go_model.search_collect import create_search_q_policy


class UniformEvaluator:
    def evaluate(self, game_state: GameState) -> tuple[np.ndarray, float]:
        return np.full(POLICY_MOVE_COUNT, 1 / POLICY_MOVE_COUNT), 0.0

    def evaluate_batch(
        self,
        game_states: list[GameState],
    ) -> list[tuple[np.ndarray, float]]:
        return [self.evaluate(game_state) for game_state in game_states]


class SearchTest(unittest.TestCase):
    def test_late_search_simulation_count_only_applies_after_cutoff(self) -> None:
        early_game_state = GameState(move_count=59)
        late_game_state = GameState(move_count=60)

        self.assertEqual(
            get_search_simulation_count(early_game_state, 64, 40, 60),
            64,
        )
        self.assertEqual(
            get_search_simulation_count(late_game_state, 64, 40, 60),
            40,
        )

    def test_disabled_late_search_keeps_base_simulation_count(self) -> None:
        self.assertEqual(
            get_search_simulation_count(GameState(move_count=80), 64, 0, 60),
            64,
        )

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

    def test_first_play_urgency_uses_parent_value_for_unvisited_child(
        self,
    ) -> None:
        parent = SearchNode(GameState(), 1, visit_count=4, value_sum=3)
        unvisited_child = SearchNode(GameState(), 0.1)
        visited_child = SearchNode(
            GameState(),
            0.9,
            visit_count=1,
            value_sum=-0.4,
        )
        parent.children = {
            0: unvisited_child,
            1: visited_child,
        }

        baseline_child = select_child(parent, exploration=0)
        urgency_child = select_child(
            parent,
            exploration=0,
            first_play_urgency_reduction=0.1,
        )

        self.assertIs(baseline_child, visited_child)
        self.assertIs(urgency_child, unvisited_child)

    def test_first_play_urgency_can_scale_reduction_by_visited_prior(
        self,
    ) -> None:
        parent = SearchNode(GameState(), 1, visit_count=4, value_sum=2)
        unvisited_child = SearchNode(GameState(), 0.96)
        visited_child = SearchNode(
            GameState(),
            0.04,
            visit_count=1,
            value_sum=-0.3,
        )
        parent.children = {
            0: unvisited_child,
            1: visited_child,
        }

        fixed_reduction_child = select_child(
            parent,
            exploration=0,
            first_play_urgency_reduction=0.5,
        )
        prior_mass_child = select_child(
            parent,
            exploration=0,
            first_play_urgency_reduction=0.5,
            use_first_play_urgency_prior_mass=True,
        )

        self.assertIs(fixed_reduction_child, visited_child)
        self.assertIs(prior_mass_child, unvisited_child)

    def test_first_play_urgency_can_apply_at_root_only(self) -> None:
        root_reduction = resolve_first_play_urgency_reduction(
            0.25,
            True,
            True,
        )
        descendant_reduction = resolve_first_play_urgency_reduction(
            0.25,
            True,
            False,
        )

        self.assertEqual(root_reduction, 0.25)
        self.assertEqual(descendant_reduction, -1.0)

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

    def test_root_branch_pruning_keeps_highest_policy_moves(self) -> None:
        node = SearchNode(GameState(), 1)
        node.children = {
            move: SearchNode(GameState(), move / 10)
            for move in range(5)
        }

        prune_root_children(node, 2)

        self.assertEqual(list(node.children), [3, 4])

    def test_root_policy_temperature_sharpens_priors(self) -> None:
        policy = np.zeros(POLICY_MOVE_COUNT, dtype=np.float32)
        policy[0] = 0.8
        policy[1] = 0.2

        class RootEvaluator:
            def evaluate(
                self,
                game_state: GameState,
            ) -> tuple[np.ndarray, float]:
                return policy, 0

        evaluator = RootEvaluator()
        search_session = MokaSearchSession(
            evaluator,
            root_evaluator=evaluator,
            root_policy_temperature=0.5,
        )
        starting_state = GameState()
        root = search_session.align_root(starting_state)

        search_session.refresh_root_evaluation(root, starting_state)

        self.assertGreater(root.children[0].prior, 0.9)
        self.assertLess(root.children[1].prior, 0.1)

    def test_search_targets_report_root_perspective_q_values(self) -> None:
        class PositiveEvaluator(UniformEvaluator):
            def evaluate(
                self,
                game_state: GameState,
            ) -> tuple[np.ndarray, float]:
                policy, _ = super().evaluate(game_state)
                return policy, 0.75

        evaluator = PositiveEvaluator()
        search_session = MokaSearchSession(evaluator)

        _, _, q_values, q_weights = (
            search_session.select_move_with_search_targets(
                GameState(),
                2,
            )
        )

        self.assertEqual(np.sum(q_weights), 1)
        self.assertTrue(
            np.allclose(
                q_values[q_weights > 0],
                -0.75,
            )
        )

    def test_search_q_policy_uses_only_visited_moves(self) -> None:
        q_values = np.asarray([0.5, 0.25, 0.75, 1.0], dtype=np.float32)
        q_weights = np.asarray([1, 1, 0, 0], dtype=np.float32)

        q_policy = create_search_q_policy(q_values, q_weights, 0.25)

        self.assertAlmostEqual(float(np.sum(q_policy)), 1)
        self.assertGreater(q_policy[0], q_policy[1])
        self.assertEqual(q_policy[2], 0)
        self.assertEqual(q_policy[3], 0)


if __name__ == "__main__":
    unittest.main()
