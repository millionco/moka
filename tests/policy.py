import unittest

import mlx.core as mx
import numpy as np

from go_model.config import GRPO_OPENING_MOVE_COUNTS, POLICY_MOVE_COUNT
from go_model.outcomes import create_group_opening_states
from go_model.policy import (
    calculate_contribution_weighted_advantages,
    calculate_group_relative_advantages,
    calculate_n_distill_advantages,
    calculate_sample_routed_teacher_policies,
    calculate_strict_sample_routing,
)
from go_model.reweight import (
    calculate_counterfactual_critical_weights,
    calculate_counterfactual_regret_weights,
    calculate_rollout_regret_weights,
    calculate_rollout_regret_critical_weights,
)
from go_model.train import (
    calculate_listwise_policy_loss,
    calculate_q_rank_policy_loss,
    normalize_sample_weights,
    normalize_score_targets,
)


class UniformTeacher:
    def evaluate_batch(self, game_states):
        policy = np.full(
            POLICY_MOVE_COUNT,
            1 / POLICY_MOVE_COUNT,
            dtype=np.float32,
        )
        return [(policy, 0) for _ in game_states]


class FeatureLogitModel:
    def __call__(self, features):
        logits = mx.zeros((features.shape[0], POLICY_MOVE_COUNT))
        logits[:, 0] = features[:, 0, 0, 0]
        return logits, mx.zeros(features.shape[0])


class GroupRelativePolicyOptimizationTests(unittest.TestCase):
    def test_score_targets_are_bounded_to_the_auxiliary_output(self) -> None:
        normalized_scores = normalize_score_targets(
            np.asarray([-96, -40, 0, 40, 96], dtype=np.float32)
        )

        np.testing.assert_allclose(
            normalized_scores,
            np.asarray([-1, -1, 0, 1, 1], dtype=np.float32),
        )

    def test_zero_sample_weights_remain_finite(self) -> None:
        normalized_weights = normalize_sample_weights(mx.zeros(8))

        self.assertTrue(bool(mx.all(mx.isfinite(normalized_weights)).item()))
        self.assertEqual(float(mx.sum(normalized_weights).item()), 0)

    def test_same_color_games_use_independent_group_advantages(self) -> None:
        game_ids = np.repeat(np.arange(16, dtype=np.int32), 2)
        rewards = np.repeat(
            np.asarray(
                [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
                dtype=np.float32,
            ),
            2,
        )
        advantages = calculate_group_relative_advantages(
            game_ids,
            rewards,
            4,
        )

        self.assertAlmostEqual(float(np.mean(advantages[game_ids % 2 == 0])), 0)
        self.assertAlmostEqual(float(np.mean(advantages[game_ids % 2 == 1])), 0)

    def test_games_in_each_group_share_the_same_opening(self) -> None:
        group_size = 4
        games_per_opening = group_size * 2
        opening_states = create_group_opening_states(
            UniformTeacher(),
            game_count=24,
            group_size=group_size,
            random_generator=np.random.default_rng(17),
        )

        for group_start in range(0, len(opening_states), games_per_opening):
            group_states = opening_states[
                group_start : group_start + games_per_opening
            ]
            opening_history = group_states[0].move_history
            self.assertIn(len(opening_history), GRPO_OPENING_MOVE_COUNTS)
            self.assertTrue(
                all(
                    game_state.move_history == opening_history
                    for game_state in group_states
                )
            )

    def test_contribution_weighting_emphasizes_helpful_steps(self) -> None:
        advantages = np.asarray([1, 1, -1, -1], dtype=np.float32)
        action_advantages = np.asarray([0.5, -0.5, -0.5, 0.5], dtype=np.float32)
        weighted_advantages = calculate_contribution_weighted_advantages(
            advantages,
            action_advantages,
            0.5,
        )

        np.testing.assert_allclose(
            weighted_advantages,
            np.asarray([1.25, 0.75, -1.25, -0.75], dtype=np.float32),
        )

    def test_sample_routing_distills_failed_games_more(self) -> None:
        teacher_policies = np.full((2, POLICY_MOVE_COUNT), 0.5, dtype=np.float32)
        routed_policies = calculate_sample_routed_teacher_policies(
            teacher_policies,
            np.asarray([-1, 1], dtype=np.float32),
        )

        np.testing.assert_allclose(routed_policies[0], 0.75)
        np.testing.assert_allclose(routed_policies[1], 0.25)

    def test_strict_sample_routing_separates_objectives(self) -> None:
        teacher_policies = np.full((2, POLICY_MOVE_COUNT), 0.5, dtype=np.float32)
        routed_advantages, routed_policies = calculate_strict_sample_routing(
            teacher_policies,
            np.asarray([-1, 1], dtype=np.float32),
            np.asarray([False, True]),
        )

        np.testing.assert_allclose(routed_advantages, [0, 1])
        np.testing.assert_allclose(routed_policies[0], 1)
        np.testing.assert_allclose(routed_policies[1], 0)

    def test_n_distill_rewards_trajectories_with_better_next_state_match(
        self,
    ) -> None:
        features = np.zeros((4, 9, 9, 12), dtype=np.float32)
        features[1, 0, 0, 0] = 5
        features[3, 0, 0, 0] = -5
        game_ids = np.asarray([0, 0, 2, 2], dtype=np.int32)
        legal_masks = np.ones((4, POLICY_MOVE_COUNT), dtype=np.bool_)
        moka_action_masks = np.asarray([True, False, True, False])
        teacher_policies = np.zeros((4, POLICY_MOVE_COUNT), dtype=np.float32)
        teacher_policies[:, 0] = 1
        advantages = calculate_n_distill_advantages(
            FeatureLogitModel(),
            features,
            game_ids,
            legal_masks,
            moka_action_masks,
            teacher_policies,
            group_size=2,
            batch_size=4,
        )

        self.assertGreater(advantages[0], 0)
        self.assertLess(advantages[2], 0)

    def test_listwise_loss_prefers_teacher_ranking(self) -> None:
        teacher_targets = mx.array(
            [[0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]]
        )
        ranked_logits = mx.array([[4, 3, 2, 1, 0, -1, -2, -3]])
        reversed_logits = ranked_logits[:, ::-1]

        ranked_loss = calculate_listwise_policy_loss(
            ranked_logits,
            teacher_targets,
            mx.ones(1),
        )
        reversed_loss = calculate_listwise_policy_loss(
            reversed_logits,
            teacher_targets,
            mx.ones(1),
        )

        self.assertLess(float(ranked_loss.item()), float(reversed_loss.item()))

    def test_q_rank_loss_prefers_materially_better_teacher_move(
        self,
    ) -> None:
        policy_targets = mx.array([[0.6, 0.3, 0.1]])
        q_values = mx.array([[0.8, 0.7, -0.2]])
        q_weights = mx.array([[12, 8, 4]])
        ranked_loss = calculate_q_rank_policy_loss(
            mx.array([[3, 2, 0]]),
            policy_targets,
            q_values,
            q_weights,
            mx.ones(1),
        )
        reversed_loss = calculate_q_rank_policy_loss(
            mx.array([[0, 2, 3]]),
            policy_targets,
            q_values,
            q_weights,
            mx.ones(1),
        )

        self.assertLess(float(ranked_loss.item()), float(reversed_loss.item()))

    def test_q_rank_loss_ignores_unvisited_and_equivalent_moves(
        self,
    ) -> None:
        loss = calculate_q_rank_policy_loss(
            mx.array([[0, 10, 10]]),
            mx.array([[0.6, 0.3, 0.1]]),
            mx.array([[0.8, 0.79, -0.2]]),
            mx.array([[12, 8, 1]]),
            mx.ones(1),
        )

        self.assertEqual(float(loss.item()), 0)

    def test_counterfactual_weights_focus_positive_regret(self) -> None:
        policies = np.asarray(
            [
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
            ],
            dtype=np.float32,
        )
        q_values = np.asarray(
            [
                [0.5, 0],
                [0.5, 0],
                [0.5, 0],
                [0.5, 0],
            ],
            dtype=np.float32,
        )
        counterfactual_values = np.asarray(
            [np.nan, 0.75, 0, -0.5],
            dtype=np.float32,
        )
        weights = calculate_counterfactual_regret_weights(
            policies,
            q_values,
            counterfactual_values,
        )

        self.assertEqual(weights[0], weights[1])
        self.assertGreater(weights[2], weights[1])
        self.assertGreater(weights[3], weights[2])

    def test_counterfactual_critical_weights_ignore_small_regret(self) -> None:
        policies = np.asarray([[1, 0], [1, 0], [1, 0]], dtype=np.float32)
        q_values = np.asarray([[0.5, 0], [0.5, 0], [0.5, 0]], dtype=np.float32)
        counterfactual_values = np.asarray([0.4, 0.2, -0.5], dtype=np.float32)
        weights = calculate_counterfactual_critical_weights(
            policies,
            q_values,
            counterfactual_values,
        )

        np.testing.assert_array_equal(weights, [0, 1, 4])

    def test_rollout_regret_weights_use_searched_moka_move(self) -> None:
        policies = np.asarray(
            [
                [0.8, 0.2, 0],
                [0.7, 0.3, 0],
                [0.6, 0.4, 0],
            ],
            dtype=np.float32,
        )
        q_values = np.asarray(
            [
                [0.7, 0.7, 0],
                [0.8, 0.4, 0],
                [0.9, -0.1, 0],
            ],
            dtype=np.float32,
        )
        q_weights = np.asarray(
            [
                [4, 2, 0],
                [4, 2, 0],
                [4, 2, 0],
            ],
            dtype=np.float32,
        )

        weights = calculate_rollout_regret_weights(
            policies,
            q_values,
            q_weights,
            np.asarray([0, 1, 2]),
        )

        np.testing.assert_allclose(
            weights,
            np.asarray([0.25, 1.75, 0.25], dtype=np.float32),
        )

    def test_rollout_regret_critical_weights_keep_material_errors(
        self,
    ) -> None:
        policies = np.asarray(
            [
                [0.8, 0.2],
                [0.7, 0.3],
                [0.6, 0.4],
            ],
            dtype=np.float32,
        )
        q_values = np.asarray(
            [
                [0.7, 0.6],
                [0.8, 0.4],
                [0.5, -0.5],
            ],
            dtype=np.float32,
        )
        q_weights = np.ones_like(q_values)

        weights = calculate_rollout_regret_critical_weights(
            policies,
            q_values,
            q_weights,
            np.asarray([1, 1, 1]),
        )

        np.testing.assert_array_equal(weights, [0, 1, 4])


if __name__ == "__main__":
    unittest.main()
