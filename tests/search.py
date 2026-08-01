import unittest

import mlx.core as mx
import numpy as np

from go_model.arena import (
    create_argument_parser,
    get_search_simulation_count,
    get_search_value_weight,
    should_accept_opponent_pass,
    should_resign_selected_pass,
)
from go_model.board import GameState, get_area_score, play_move
from go_model.config import BOARD_SIZE, POLICY_MOVE_COUNT
from go_model.features import encode_moka_features
from go_model.model import (
    MokaNetwork,
    MokaOptimisticPolicyNetwork,
    MokaUncertaintyNetwork,
)
from go_model.search import (
    MokaSearchSession,
    MokaEvaluator,
    SearchNode,
    backup_search_node,
    blend_search_value,
    blend_child_q_value,
    calculate_dynamic_exploration,
    calculate_child_lcb,
    calculate_uncertainty_weight,
    expand_node_with_evaluation,
    prune_root_children,
    resolve_first_play_urgency_reduction,
    resolve_q_value_normalization_weight,
    resolve_root_policy_temperature,
    resolve_score_value_weight,
    resolve_search_exploration,
    run_search_simulations,
    select_child,
    select_root_child_with_lcb,
)
from go_model.search_collect import (
    create_search_q_policy,
    get_visited_child_value_targets,
)


class UniformEvaluator:
    def evaluate(self, game_state: GameState) -> tuple[np.ndarray, float]:
        return np.full(POLICY_MOVE_COUNT, 1 / POLICY_MOVE_COUNT), 0.0

    def evaluate_batch(
        self,
        game_states: list[GameState],
    ) -> list[tuple[np.ndarray, float]]:
        return [self.evaluate(game_state) for game_state in game_states]


class FreeUniformEvaluator(UniformEvaluator):
    evaluation_count = 0


class FixedValueEvaluator(UniformEvaluator):
    def __init__(self, value: float) -> None:
        self.value = value
        self.evaluation_count = 0

    def evaluate(self, game_state: GameState) -> tuple[np.ndarray, float]:
        self.evaluation_count += 1
        return (
            np.full(POLICY_MOVE_COUNT, 1 / POLICY_MOVE_COUNT),
            self.value,
        )

    def evaluate_batch(
        self,
        game_states: list[GameState],
    ) -> list[tuple[np.ndarray, float]]:
        return [self.evaluate(game_state) for game_state in game_states]


class FixedSpreadEvaluator(FixedValueEvaluator):
    def __init__(self, value: float, spread: float) -> None:
        super().__init__(value)
        self.spread = spread

    def get_symmetry_value_spread(self, game_state: GameState) -> float:
        return self.spread

    def get_uncertainty_prediction(self, game_state: GameState) -> float:
        return 0


class FixedUncertaintyEvaluator(FixedSpreadEvaluator):
    def __init__(self, value: float, uncertainty: float) -> None:
        super().__init__(value, 0)
        self.uncertainty = uncertainty

    def get_uncertainty_prediction(self, game_state: GameState) -> float:
        return self.uncertainty


class BiasedPolicyModel:
    def __call__(self, features: mx.array) -> tuple[mx.array, mx.array]:
        policy_logits = np.zeros(
            (features.shape[0], POLICY_MOVE_COUNT),
            dtype=np.float32,
        )
        policy_logits[:, 0] = 1
        return (
            mx.array(policy_logits),
            mx.zeros((features.shape[0],), dtype=mx.float32),
        )


class SearchTest(unittest.TestCase):
    def test_zero_area_weight_preserves_network_value_exactly(self) -> None:
        network_value = 0.375

        self.assertEqual(
            blend_search_value(GameState(), network_value, 0),
            network_value,
        )

    def test_fast_child_selection_matches_generic_selection(self) -> None:
        parent = SearchNode(GameState(), 1)

        for move in range(BOARD_SIZE):
            child = SearchNode(
                GameState(),
                prior=(move + 1) / BOARD_SIZE**2,
                action_value_prior=(move - BOARD_SIZE // 2) / BOARD_SIZE,
            )

            for visit_index in range(move % (BOARD_SIZE // 2)):
                backup_search_node(
                    child,
                    (visit_index - move) / BOARD_SIZE,
                    1,
                )

            parent.children[move] = child

        for value in (-0.5, 0.25, 0.75):
            backup_search_node(parent, value, 1)

        fast_selection = select_child(parent)
        generic_selection = select_child(
            parent,
            reservation_counts={id(parent): 0},
        )

        self.assertIs(fast_selection, generic_selection)

    def test_terminal_search_uses_area_score_over_network_value(
        self,
    ) -> None:
        game_state = GameState(
            board=np.ones((9, 9), dtype=np.int8),
            consecutive_pass_count=2,
        )
        root = SearchNode(game_state, 1)
        evaluator = FixedValueEvaluator(-0.75)

        run_search_simulations(root, evaluator, 3)

        self.assertGreater(get_area_score(game_state), 0)
        self.assertEqual(root.visit_count, 3)
        self.assertEqual(root.mean_value, 1)

    def test_child_value_targets_keep_visited_child_perspective(
        self,
    ) -> None:
        root = SearchNode(GameState(), 1)
        visited_child = SearchNode(
            GameState(next_color=-1),
            0.6,
            visit_count=4,
            value_sum=-2,
        )
        root.children = {
            0: visited_child,
            1: SearchNode(GameState(), 0.4),
        }

        targets = get_visited_child_value_targets(root)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0][0].shape, (9, 9, 12))
        self.assertEqual(targets[0][1], -0.5)
        self.assertEqual(targets[0][2], 4)

    def test_arena_defaults_match_accepted_search_player(self) -> None:
        arguments = create_argument_parser().parse_args(
            ["--checkpoint", "checkpoint.safetensors"],
        )

        self.assertEqual(arguments.simulations, 256)
        self.assertEqual(arguments.search_exploration, 2.0)
        self.assertIsNone(arguments.descendant_search_exploration)
        self.assertEqual(arguments.search_child_q_pseudo_count, 0.0)
        self.assertEqual(
            arguments.search_evaluation_budget_extra_simulations,
            0,
        )
        self.assertEqual(arguments.search_fpu_reduction, 0.25)
        self.assertEqual(arguments.opponent_branches, 4)
        self.assertEqual(arguments.resignation_area_margin, 60.0)
        self.assertEqual(arguments.search_score_value_start_move, 0)
        self.assertEqual(arguments.search_optimistic_policy_weight, 0)
        self.assertEqual(arguments.root_optimistic_policy_weight, 0)
        self.assertEqual(arguments.search_uncertainty_coefficient, 0)
        self.assertEqual(arguments.root_uncertainty_coefficient, 0)
        self.assertEqual(arguments.uncertainty_maximum_weight, 8)
        self.assertTrue(np.isinf(arguments.adaptive_uncertainty_threshold))
        self.assertEqual(arguments.search_utility_stdev_prior, 0.4)
        self.assertEqual(arguments.search_utility_stdev_prior_weight, 2)
        self.assertEqual(arguments.search_utility_stdev_scale, 0)
        self.assertEqual(arguments.root_lcb_stdevs, 0)
        self.assertEqual(arguments.root_lcb_minimum_visit_proportion, 0.2)
        self.assertTrue(arguments.symmetry_ensemble)
        self.assertEqual(arguments.descendant_policy_temperature, 1.0)
        self.assertEqual(
            arguments.descendant_symmetry_geometric_policy_weight,
            0.125,
        )
        self.assertTrue(arguments.root_symmetry_ensemble)
        self.assertFalse(arguments.shared_root_evaluator)
        self.assertEqual(
            arguments.root_symmetry_geometric_policy_weight,
            0.125,
        )
        control_arguments = create_argument_parser().parse_args(
            [
                "--checkpoint",
                "checkpoint.safetensors",
                "--no-symmetry-ensemble",
                "--descendant-symmetry-geometric-policy-weight",
                "0.5",
                "--descendant-policy-temperature",
                "0.75",
                "--descendant-search-exploration",
                "1.5",
                "--search-child-q-pseudo-count",
                "1",
            ],
        )
        self.assertFalse(control_arguments.symmetry_ensemble)
        self.assertEqual(
            control_arguments.descendant_search_exploration,
            1.5,
        )
        self.assertEqual(control_arguments.search_child_q_pseudo_count, 1)
        self.assertEqual(
            control_arguments.descendant_policy_temperature,
            0.75,
        )
        self.assertEqual(
            control_arguments.descendant_symmetry_geometric_policy_weight,
            0.5,
        )

    def test_score_value_weight_starts_at_configured_move(self) -> None:
        self.assertEqual(resolve_score_value_weight(0.25, 60, 59), 0)
        self.assertEqual(resolve_score_value_weight(0.25, 60, 60), 0.25)

    def test_descendant_policy_temperature_sharpens_evaluator_policy(
        self,
    ) -> None:
        model = BiasedPolicyModel()
        default_evaluator = MokaEvaluator(
            model,
            use_symmetry_ensemble=False,
        )
        sharpened_evaluator = MokaEvaluator(
            model,
            use_symmetry_ensemble=False,
            policy_temperature=0.5,
        )

        default_policy, _ = default_evaluator.evaluate(GameState())
        sharpened_policy, _ = sharpened_evaluator.evaluate(GameState())

        self.assertGreater(sharpened_policy[0], default_policy[0])
        self.assertAlmostEqual(float(np.sum(sharpened_policy)), 1, places=6)

    def test_optimistic_policy_requires_compatible_model(self) -> None:
        with self.assertRaises(ValueError):
            MokaEvaluator(MokaNetwork(), optimistic_policy_weight=1)

    def test_uncertainty_weighting_requires_compatible_model(self) -> None:
        with self.assertRaises(ValueError):
            MokaEvaluator(MokaNetwork(), uncertainty_coefficient=0.25)

        evaluator = MokaEvaluator(
            MokaUncertaintyNetwork(),
            uncertainty_coefficient=0.25,
        )
        self.assertEqual(evaluator.get_uncertainty_weight(GameState()), 1)

    def test_uncertainty_prediction_keeps_unit_backup_weight(self) -> None:
        evaluator = MokaEvaluator(
            MokaUncertaintyNetwork(),
            use_symmetry_ensemble=False,
            predict_uncertainty=True,
        )
        game_state = GameState()

        evaluator.evaluate(game_state)

        self.assertGreater(
            evaluator.get_uncertainty_prediction(game_state),
            0,
        )
        self.assertEqual(evaluator.get_uncertainty_weight(game_state), 1)

    def test_uncertainty_weight_matches_katago_formula(self) -> None:
        maximum_weight = calculate_uncertainty_weight(
            np.array([np.log(1e-4)]),
            coefficient=0.25,
            maximum_weight=8,
        )
        unit_weight = calculate_uncertainty_weight(
            np.array([np.log(0.21885)]),
            coefficient=0.25,
            maximum_weight=8,
        )
        low_weight = calculate_uncertainty_weight(
            np.array([np.log(0.5001)]),
            coefficient=0.25,
            maximum_weight=8,
        )

        self.assertAlmostEqual(maximum_weight, 8, places=3)
        self.assertAlmostEqual(unit_weight, 1, places=3)
        self.assertLess(low_weight, unit_weight)

    def test_weighted_backup_preserves_simulation_count(self) -> None:
        node = SearchNode(GameState(), 1)

        backup_search_node(node, 1, 2)
        backup_search_node(node, -1, 0.5)

        self.assertEqual(node.visit_count, 2)
        self.assertEqual(node.visit_weight, 2.5)
        self.assertEqual(node.mean_value, 0.6)
        self.assertEqual(node.value_square_sum, 2.5)

    def test_dynamic_exploration_tracks_empirical_value_spread(self) -> None:
        calm_node = SearchNode(GameState(), 1)
        volatile_node = SearchNode(GameState(), 1)

        for value in (0.5, 0.5, 0.5, 0.5):
            backup_search_node(calm_node, value, 1)
        for value in (-1, 1, -1, 1):
            backup_search_node(volatile_node, value, 1)

        calm_exploration = calculate_dynamic_exploration(
            1.75,
            calm_node,
            0.4,
            2,
            0.85,
        )
        volatile_exploration = calculate_dynamic_exploration(
            1.75,
            volatile_node,
            0.4,
            2,
            0.85,
        )

        self.assertLess(calm_exploration, 1.75)
        self.assertGreater(volatile_exploration, 1.75)

    def test_root_lcb_prefers_stable_high_value_child(self) -> None:
        stable_child = SearchNode(GameState(), 0.5)
        volatile_child = SearchNode(GameState(), 0.5)
        for value in (-0.6, -0.6, -0.6, -0.6):
            backup_search_node(stable_child, value, 1)
        for value in (-1, 0, -1, 0, -1):
            backup_search_node(volatile_child, value, 1)

        stable_lcb, stable_radius = calculate_child_lcb(stable_child, 5)
        volatile_lcb, volatile_radius = calculate_child_lcb(
            volatile_child,
            5,
        )
        selected_move, _ = select_root_child_with_lcb(
            [(0, stable_child), (1, volatile_child)],
            5,
            0.2,
        )

        self.assertGreater(stable_lcb, volatile_lcb)
        self.assertLess(stable_radius, volatile_radius)
        self.assertEqual(selected_move, 0)

    def test_optimistic_policy_can_replace_normal_policy(self) -> None:
        model = MokaOptimisticPolicyNetwork()
        model.optimistic_policy_pass.bias = mx.ones_like(
            model.optimistic_policy_pass.bias
        )
        normal_evaluator = MokaEvaluator(
            model,
            use_symmetry_ensemble=False,
        )
        optimistic_evaluator = MokaEvaluator(
            model,
            use_symmetry_ensemble=False,
            optimistic_policy_weight=1,
        )

        normal_policy, normal_value = normal_evaluator.evaluate(GameState())
        optimistic_policy, optimistic_value = optimistic_evaluator.evaluate(
            GameState()
        )

        self.assertFalse(np.allclose(normal_policy, optimistic_policy))
        self.assertAlmostEqual(float(np.sum(optimistic_policy)), 1, places=6)
        self.assertAlmostEqual(normal_value, optimistic_value, places=6)

    def test_optimistic_policy_blends_logits(self) -> None:
        model = MokaOptimisticPolicyNetwork()
        model.optimistic_policy_pass.bias = mx.ones_like(
            model.optimistic_policy_pass.bias
        )
        evaluator = MokaEvaluator(
            model,
            use_symmetry_ensemble=False,
            optimistic_policy_weight=0.5,
        )
        game_state = GameState()
        features = mx.array(encode_moka_features(game_state)[None])
        normal_logits, _, optimistic_logits = (
            model.get_optimistic_policy_outputs(features)
        )
        expected_policy = np.asarray(
            mx.softmax((normal_logits + optimistic_logits) * 0.5, axis=1)
        )[0]

        policy, _ = evaluator.evaluate(game_state)

        np.testing.assert_allclose(policy, expected_policy, rtol=1e-6)

    def test_evaluator_cache_can_be_isolated_between_games(self) -> None:
        evaluator = MokaEvaluator(
            BiasedPolicyModel(),
            use_symmetry_ensemble=False,
        )
        game_state = GameState()

        evaluator.evaluate(game_state)
        evaluator.evaluate(game_state)
        self.assertEqual(evaluator.evaluation_count, 1)

        evaluator.clear_cache()
        evaluator.evaluate(game_state)
        self.assertEqual(evaluator.evaluation_count, 2)

    def test_saved_root_evaluation_buys_one_replacement_simulation(self) -> None:
        evaluator = MokaEvaluator(
            BiasedPolicyModel(),
            use_symmetry_ensemble=False,
        )
        game_state = GameState()
        evaluator.evaluate(game_state)
        search_session = MokaSearchSession(
            evaluator,
            root_evaluator=evaluator,
            use_saved_root_evaluation_budget=True,
        )
        root = search_session.align_root(game_state)

        search_session.select_move_with_search_targets(game_state, 3)

        self.assertEqual(root.visit_count, 4)

    def test_symmetry_spread_can_extend_adaptive_search(self) -> None:
        game_state = GameState()
        evaluator = FixedValueEvaluator(0)
        root_evaluator = FixedSpreadEvaluator(0, 0.21)
        search_session = MokaSearchSession(
            evaluator,
            root_evaluator=root_evaluator,
            adaptive_max_simulation_count=4,
            adaptive_visit_margin_ratio=0,
            adaptive_symmetry_value_spread_threshold=0.2,
        )
        root = search_session.align_root(game_state)

        search_session.select_move_with_search_targets(game_state, 2)

        self.assertEqual(root.visit_count, 4)
        self.assertEqual(search_session.adaptive_extension_count, 1)
        self.assertEqual(search_session.adaptive_extra_simulation_count, 2)

    def test_low_symmetry_spread_preserves_base_search_budget(self) -> None:
        game_state = GameState()
        evaluator = FixedValueEvaluator(0)
        root_evaluator = FixedSpreadEvaluator(0, 0.19)
        search_session = MokaSearchSession(
            evaluator,
            root_evaluator=root_evaluator,
            adaptive_max_simulation_count=4,
            adaptive_visit_margin_ratio=0,
            adaptive_symmetry_value_spread_threshold=0.2,
        )
        root = search_session.align_root(game_state)

        search_session.select_move_with_search_targets(game_state, 2)

        self.assertEqual(root.visit_count, 2)
        self.assertEqual(search_session.adaptive_extension_count, 0)

    def test_uncertainty_can_extend_adaptive_search(self) -> None:
        game_state = GameState()
        evaluator = FixedValueEvaluator(0)
        root_evaluator = FixedUncertaintyEvaluator(0, 0.03)
        search_session = MokaSearchSession(
            evaluator,
            root_evaluator=root_evaluator,
            adaptive_max_simulation_count=4,
            adaptive_visit_margin_ratio=0,
            adaptive_uncertainty_threshold=0.02,
        )
        root = search_session.align_root(game_state)

        search_session.select_move_with_search_targets(game_state, 2)

        self.assertEqual(root.visit_count, 4)
        self.assertEqual(search_session.adaptive_extension_count, 1)
        self.assertEqual(search_session.adaptive_extra_simulation_count, 2)

    def test_descendant_exploration_inherits_or_overrides_root(self) -> None:
        self.assertEqual(resolve_search_exploration(2, None, False), 2)
        self.assertEqual(resolve_search_exploration(2, 1.5, True), 2)
        self.assertEqual(resolve_search_exploration(2, 1.5, False), 1.5)

    def test_child_q_pseudo_count_shrinks_toward_parent_value(self) -> None:
        self.assertEqual(blend_child_q_value(-1, 0, 1, 1), -0.5)
        self.assertEqual(blend_child_q_value(-1, 0, 3, 1), -0.75)
        self.assertEqual(blend_child_q_value(-1, 0, 1, 0), -1)

        with self.assertRaises(ValueError):
            blend_child_q_value(-1, 0, 1, -1)

    def test_child_q_pseudo_count_changes_puct_selection(self) -> None:
        parent = SearchNode(GameState(), 1, visit_count=2)
        higher_value_child = SearchNode(
            GameState(),
            0.01,
            visit_count=1,
            value_sum=-1,
        )
        higher_prior_child = SearchNode(
            GameState(),
            0.99,
            visit_count=1,
            value_sum=0,
        )
        parent.children = {
            0: higher_value_child,
            1: higher_prior_child,
        }

        raw_selection = select_child(parent, exploration=0.1)
        shrunk_selection = select_child(
            parent,
            exploration=0.1,
            child_q_pseudo_count=20,
        )

        self.assertIs(raw_selection, higher_value_child)
        self.assertIs(shrunk_selection, higher_prior_child)

    def test_resignation_requires_hopeless_selected_pass(
        self,
    ) -> None:
        game_state = GameState(
            board=np.ones((9, 9), dtype=np.int8),
            next_color=-1,
            move_count=80,
            consecutive_pass_count=1,
        )

        self.assertFalse(
            should_resign_selected_pass(game_state, 81, 0),
        )
        self.assertTrue(
            should_resign_selected_pass(game_state, 81, 40),
        )
        self.assertFalse(
            should_resign_selected_pass(game_state, 0, 40),
        )

    def test_opponent_pass_acceptance_uses_area_score(
        self,
    ) -> None:
        losing_game_state = GameState(
            board=np.ones((9, 9), dtype=np.int8),
            next_color=-1,
            move_count=80,
            consecutive_pass_count=1,
        )
        winning_game_state = GameState(
            board=np.ones((9, 9), dtype=np.int8),
            next_color=1,
            move_count=80,
            consecutive_pass_count=1,
        )

        self.assertFalse(
            should_accept_opponent_pass(losing_game_state),
        )
        self.assertTrue(
            should_accept_opponent_pass(winning_game_state),
        )

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

    def test_late_search_value_weight_applies_after_cutoff(self) -> None:
        self.assertEqual(
            get_search_value_weight(GameState(move_count=49), 1.25, 0.75, 50),
            1.25,
        )
        self.assertEqual(
            get_search_value_weight(GameState(move_count=50), 1.25, 0.75, 50),
            0.75,
        )
        self.assertEqual(
            get_search_value_weight(GameState(move_count=80), 1.25, 0, 50),
            1.25,
        )

    def test_q_normalization_can_overcome_small_raw_value_scale(self) -> None:
        parent = SearchNode(GameState(), 1, visit_count=2)
        better_value_child = SearchNode(
            GameState(),
            0.01,
            visit_count=1,
            value_sum=-0.2,
        )
        higher_prior_child = SearchNode(
            GameState(),
            0.99,
            visit_count=1,
            value_sum=-0.19,
        )
        parent.children = {
            0: better_value_child,
            1: higher_prior_child,
        }

        raw_value_child = select_child(parent, exploration=0.1)
        normalized_value_child = select_child(
            parent,
            exploration=0.1,
            q_value_normalization_weight=1,
        )

        self.assertIs(raw_value_child, higher_prior_child)
        self.assertIs(normalized_value_child, better_value_child)

    def test_q_normalization_can_apply_at_root_only(self) -> None:
        root_weight = resolve_q_value_normalization_weight(0.5, True, True)
        descendant_weight = resolve_q_value_normalization_weight(
            0.5,
            True,
            False,
        )

        self.assertEqual(root_weight, 0.5)
        self.assertEqual(descendant_weight, 0)

    def test_batched_search_preserves_requested_visit_count(self) -> None:
        evaluator = UniformEvaluator()
        root = SearchNode(GameState(), 1)

        run_search_simulations(root, evaluator, 17)

        self.assertEqual(root.visit_count, 17)
        self.assertTrue(root.children)

    def test_evaluation_budget_adds_only_bounded_free_simulations(self) -> None:
        evaluator = FreeUniformEvaluator()
        root = SearchNode(GameState(), 1)

        run_search_simulations(
            root,
            evaluator,
            17,
            maximum_extra_simulation_count=3,
        )

        self.assertEqual(root.visit_count, 20)

    def test_evaluation_budget_rejects_negative_extra_simulations(self) -> None:
        with self.assertRaises(ValueError):
            run_search_simulations(
                SearchNode(GameState(), 1),
                FreeUniformEvaluator(),
                17,
                maximum_extra_simulation_count=-1,
            )

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

        baseline_child = select_child(
            parent,
            exploration=0,
            first_play_urgency_reduction=-1,
        )
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

    def test_action_value_prior_guides_only_unvisited_children(self) -> None:
        parent = SearchNode(GameState(), 1, value_sum=0, visit_count=1)
        lower_prior_child = SearchNode(
            GameState(),
            0.5,
            action_value_prior=-0.1,
        )
        higher_prior_child = SearchNode(
            GameState(),
            0.5,
            action_value_prior=0.1,
        )
        parent.children = {0: lower_prior_child, 1: higher_prior_child}

        selected_child = select_child(
            parent,
            exploration=0,
            first_play_urgency_reduction=0.25,
        )

        self.assertIs(selected_child, higher_prior_child)

        higher_prior_child.visit_count = 1
        higher_prior_child.value_sum = 0.5
        lower_prior_child.visit_count = 1
        lower_prior_child.value_sum = 0
        selected_visited_child = select_child(
            parent,
            exploration=0,
            first_play_urgency_reduction=0.25,
        )

        self.assertIs(selected_visited_child, lower_prior_child)

    def test_expansion_centers_action_value_priors(self) -> None:
        node = SearchNode(GameState(), 1)
        policy = np.full(POLICY_MOVE_COUNT, 1, dtype=np.float32)
        action_value_prior = np.arange(
            POLICY_MOVE_COUNT,
            dtype=np.float32,
        )

        expand_node_with_evaluation(
            node,
            policy,
            action_value_prior=action_value_prior,
        )

        child_priors = np.asarray(
            [child.action_value_prior for child in node.children.values()]
        )
        self.assertAlmostEqual(float(np.mean(child_priors)), 0)
        self.assertGreater(float(np.max(child_priors)), 0)
        self.assertLess(float(np.min(child_priors)), 0)

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

    def test_root_policy_temperature_returns_to_one_after_cutoff(self) -> None:
        self.assertEqual(
            resolve_root_policy_temperature(0.75, 60, 59),
            0.75,
        )
        self.assertEqual(
            resolve_root_policy_temperature(0.75, 60, 60),
            1.0,
        )
        self.assertEqual(
            resolve_root_policy_temperature(0.75, 0, 80),
            0.75,
        )

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
