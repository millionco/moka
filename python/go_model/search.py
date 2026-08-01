from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from go_model.board import (
    GameState,
    get_area_score,
    get_group,
    get_legal_move_states,
    get_legal_moves,
    is_game_over,
    play_move,
)
from go_model.config import (
    BOARD_AREA,
    BOARD_SIZE,
    BOARD_SYMMETRY_REFLECTION_COUNT,
    BOARD_SYMMETRY_ROTATION_COUNT,
    MINIMUM_TEACHER_PASS_MOVE_COUNT,
    POLICY_MOVE_COUNT,
    SEARCH_ADAPTIVE_MAX_SIMULATION_COUNT,
    SEARCH_ADAPTIVE_SYMMETRY_VALUE_SPREAD_THRESHOLD,
    SEARCH_ADAPTIVE_UNCERTAINTY_THRESHOLD,
    SEARCH_ADAPTIVE_VISIT_MARGIN_RATIO,
    SEARCH_AREA_VALUE_SCALE_POINTS,
    SEARCH_AREA_VALUE_START_MOVE_COUNT,
    SEARCH_AREA_VALUE_RAMP_MOVE_COUNT,
    SEARCH_AREA_VALUE_WEIGHT,
    SEARCH_CHILD_Q_PSEUDO_COUNT,
    SEARCH_DESCENDANT_PUCT_EXPLORATION,
    SEARCH_DESCENDANT_POLICY_TEMPERATURE,
    SEARCH_DESCENDANT_SYMMETRY_ENSEMBLE,
    SEARCH_DESCENDANT_SYMMETRY_GEOMETRIC_POLICY_WEIGHT,
    SEARCH_FIRST_PLAY_URGENCY_REDUCTION,
    SEARCH_FIRST_PLAY_URGENCY_ROOT_ONLY,
    SEARCH_FIRST_PLAY_URGENCY_USE_PRIOR_MASS,
    SEARCH_MAXIMUM_EXTRA_EVALUATION_BUDGET_SIMULATION_COUNT,
    SEARCH_OPPONENT_BRANCH_COUNT,
    SEARCH_POLICY_EPSILON,
    SEARCH_PUCT_EXPLORATION,
    SEARCH_PUCT_UTILITY_STDEV_PRIOR,
    SEARCH_PUCT_UTILITY_STDEV_PRIOR_WEIGHT,
    SEARCH_PUCT_UTILITY_STDEV_SCALE,
    SEARCH_PUCT_VALUE_WEIGHT,
    SEARCH_Q_VALUE_NORMALIZATION_EPSILON,
    SEARCH_Q_VALUE_NORMALIZATION_ROOT_ONLY,
    SEARCH_Q_VALUE_NORMALIZATION_WEIGHT,
    SEARCH_ROLLOUT_DEPTH,
    SEARCH_SCORE_VALUE_START_MOVE_COUNT,
    SEARCH_SCORE_VALUE_WEIGHT,
    SEARCH_ROOT_BRANCH_COUNT,
    SEARCH_ROOT_LCB_MINIMUM_VISIT_PROPORTION,
    SEARCH_ROOT_LCB_STDEVS,
    SEARCH_ROOT_LCB_VARIANCE_EPSILON,
    SEARCH_ROOT_LCB_WEIGHT_GAIN_LIMIT,
    SEARCH_ROOT_POLICY_TEMPERATURE_END_MOVE_COUNT,
    SEARCH_ROOT_SYMMETRY_RANK_MINIMUM_TOP_MOVE_VOTE_COUNT,
    SEARCH_ROOT_SYMMETRY_RANK_MOVE_COUNT,
    SEARCH_ROOT_SYMMETRY_RANK_POLICY_END_MOVE_COUNT,
    SEARCH_ROOT_SYMMETRY_RANK_POLICY_WEIGHT,
    SEARCH_ROOT_SYMMETRY_TOP_MOVE_VOTE_POLICY_WEIGHT,
    SEARCH_ROOT_SYMMETRY_TRIMMED_POLICY_WEIGHT,
    SEARCH_ROOT_SYMMETRY_TRIMMED_VALUE_WEIGHT,
    SEARCH_ROOT_POLICY_TEMPERATURE,
    SEARCH_SEQUENTIAL_HALVING_REDUCTION_FACTOR,
    SEARCH_SIMULATION_BATCH_SIZE,
    SEARCH_UNCERTAINTY_COEFFICIENT,
    SEARCH_UNCERTAINTY_LOG_MAXIMUM,
    SEARCH_UNCERTAINTY_LOG_MINIMUM,
    SEARCH_UNCERTAINTY_MAXIMUM_WEIGHT,
    SEARCH_UNCERTAINTY_TARGET_EPSILON,
    SEARCH_VALUE_WEIGHT,
)
from go_model.features import encode_moka_features
from go_model.model import (
    MokaActionValueNetwork,
    MokaGlobalScoreNetwork,
    MokaNetwork,
    MokaOptimisticPolicyNetwork,
    MokaUncertaintyNetwork,
)
from go_model.symmetry import (
    aggregate_symmetry_policies,
    aggregate_symmetry_values,
    apply_board_symmetry,
    invert_policy_symmetry,
    resolve_symmetry_rank_policy_weight,
)


@dataclass
class SearchNode:
    game_state: GameState
    prior: float
    children: dict[int, "SearchNode"] = field(default_factory=dict)
    value_sum: float = 0
    value_square_sum: float = 0
    visit_count: int = 0
    visit_weight: float = 0
    visit_weight_square_sum: float = 0
    action_value_prior: float = 0

    @property
    def effective_visit_weight(self) -> float:
        return self.visit_weight if self.visit_weight > 0 else self.visit_count

    @property
    def effective_visit_weight_square_sum(self) -> float:
        return (
            self.visit_weight_square_sum
            if self.visit_weight_square_sum > 0
            else self.visit_count
        )

    @property
    def mean_value(self) -> float:
        visit_weight = self.effective_visit_weight
        return self.value_sum / visit_weight if visit_weight else 0


class MokaEvaluator:
    def __init__(
        self,
        model: MokaNetwork,
        use_symmetry_ensemble: bool = SEARCH_DESCENDANT_SYMMETRY_ENSEMBLE,
        symmetry_rotation_count: int = 0,
        should_flip_symmetry: bool = False,
        use_symmetry_pair: bool = False,
        policy_temperature: float = SEARCH_DESCENDANT_POLICY_TEMPERATURE,
        symmetry_geometric_policy_weight: float = (
            SEARCH_DESCENDANT_SYMMETRY_GEOMETRIC_POLICY_WEIGHT
        ),
        symmetry_trimmed_policy_weight: float = (
            SEARCH_ROOT_SYMMETRY_TRIMMED_POLICY_WEIGHT
        ),
        symmetry_trimmed_value_weight: float = (
            SEARCH_ROOT_SYMMETRY_TRIMMED_VALUE_WEIGHT
        ),
        symmetry_rank_policy_weight: float = (
            SEARCH_ROOT_SYMMETRY_RANK_POLICY_WEIGHT
        ),
        symmetry_rank_move_count: int = SEARCH_ROOT_SYMMETRY_RANK_MOVE_COUNT,
        symmetry_rank_policy_end_move_count: int = (
            SEARCH_ROOT_SYMMETRY_RANK_POLICY_END_MOVE_COUNT
        ),
        symmetry_rank_minimum_top_move_vote_count: int = (
            SEARCH_ROOT_SYMMETRY_RANK_MINIMUM_TOP_MOVE_VOTE_COUNT
        ),
        symmetry_top_move_vote_policy_weight: float = (
            SEARCH_ROOT_SYMMETRY_TOP_MOVE_VOTE_POLICY_WEIGHT
        ),
        score_value_weight: float = SEARCH_SCORE_VALUE_WEIGHT,
        score_value_start_move_count: int = (
            SEARCH_SCORE_VALUE_START_MOVE_COUNT
        ),
        action_value_prior_weight: float = 0,
        optimistic_policy_weight: float = 0,
        uncertainty_coefficient: float = 0,
        uncertainty_maximum_weight: float = (
            SEARCH_UNCERTAINTY_MAXIMUM_WEIGHT
        ),
        predict_uncertainty: bool = False,
    ) -> None:
        if not 0 <= score_value_weight <= 1:
            raise ValueError("Score value weight must be between zero and one.")
        if score_value_weight > 0 and not isinstance(
            model,
            MokaGlobalScoreNetwork,
        ):
            raise ValueError(
                "Score value blending requires a global-score checkpoint."
            )
        if score_value_start_move_count < 0:
            raise ValueError("Score value start move must not be negative.")
        if action_value_prior_weight < 0:
            raise ValueError(
                "Action-value prior weight must not be negative."
            )
        if action_value_prior_weight > 0 and not isinstance(
            model,
            MokaActionValueNetwork,
        ):
            raise ValueError(
                "Action-value priors require an action-value checkpoint."
            )
        if not 0 <= optimistic_policy_weight <= 1:
            raise ValueError(
                "Optimistic-policy weight must be between zero and one."
            )
        if optimistic_policy_weight > 0 and not isinstance(
            model,
            MokaOptimisticPolicyNetwork,
        ):
            raise ValueError(
                "Optimistic policy requires an optimistic-policy checkpoint."
            )
        if uncertainty_coefficient < 0:
            raise ValueError("Uncertainty coefficient must not be negative.")
        if uncertainty_maximum_weight <= 0:
            raise ValueError("Maximum uncertainty weight must be positive.")
        if (uncertainty_coefficient > 0 or predict_uncertainty) and not isinstance(
            model,
            MokaUncertaintyNetwork,
        ):
            raise ValueError(
                "Uncertainty weighting requires an uncertainty checkpoint."
            )
        self.model = model
        self.use_symmetry_ensemble = use_symmetry_ensemble
        self.symmetry_rotation_count = symmetry_rotation_count
        self.should_flip_symmetry = should_flip_symmetry
        self.use_symmetry_pair = use_symmetry_pair
        self.policy_temperature = policy_temperature
        self.symmetry_geometric_policy_weight = (
            symmetry_geometric_policy_weight
        )
        self.symmetry_trimmed_policy_weight = symmetry_trimmed_policy_weight
        self.symmetry_trimmed_value_weight = symmetry_trimmed_value_weight
        self.symmetry_rank_policy_weight = symmetry_rank_policy_weight
        self.symmetry_rank_move_count = symmetry_rank_move_count
        self.symmetry_rank_policy_end_move_count = (
            symmetry_rank_policy_end_move_count
        )
        self.symmetry_rank_minimum_top_move_vote_count = (
            symmetry_rank_minimum_top_move_vote_count
        )
        self.symmetry_top_move_vote_policy_weight = (
            symmetry_top_move_vote_policy_weight
        )
        self.score_value_weight = score_value_weight
        self.score_value_start_move_count = score_value_start_move_count
        self.action_value_prior_weight = action_value_prior_weight
        self.optimistic_policy_weight = optimistic_policy_weight
        self.uncertainty_coefficient = uncertainty_coefficient
        self.uncertainty_maximum_weight = uncertainty_maximum_weight
        self.predict_uncertainty = predict_uncertainty
        self.evaluation_count = 0
        self.cache: dict[
            tuple[bytes, int, int, tuple[int, ...]],
            tuple[np.ndarray, float],
        ] = {}
        self.symmetry_value_spreads: dict[
            tuple[bytes, int, int, tuple[int, ...]],
            float,
        ] = {}
        self.action_value_priors: dict[
            tuple[bytes, int, int, tuple[int, ...]],
            np.ndarray,
        ] = {}
        self.uncertainty_weights: dict[
            tuple[bytes, int, int, tuple[int, ...]],
            float,
        ] = {}
        self.uncertainty_predictions: dict[
            tuple[bytes, int, int, tuple[int, ...]],
            float,
        ] = {}

    def clear_cache(self) -> None:
        self.cache.clear()
        self.symmetry_value_spreads.clear()
        self.action_value_priors.clear()
        self.uncertainty_weights.clear()
        self.uncertainty_predictions.clear()

    def get_output_configuration(self) -> tuple[object, ...]:
        return (
            self.use_symmetry_ensemble,
            self.symmetry_rotation_count,
            self.should_flip_symmetry,
            self.use_symmetry_pair,
            self.policy_temperature,
            self.symmetry_geometric_policy_weight,
            self.symmetry_trimmed_policy_weight,
            self.symmetry_trimmed_value_weight,
            self.symmetry_rank_policy_weight,
            self.symmetry_rank_move_count,
            self.symmetry_rank_policy_end_move_count,
            self.symmetry_rank_minimum_top_move_vote_count,
            self.symmetry_top_move_vote_policy_weight,
            self.score_value_weight,
            self.score_value_start_move_count,
            self.action_value_prior_weight,
            self.optimistic_policy_weight,
            self.uncertainty_coefficient,
            self.uncertainty_maximum_weight,
            self.predict_uncertainty,
        )

    def get_symmetry_value_spread(self, game_state: GameState) -> float:
        return self.symmetry_value_spreads.get(
            self.get_cache_key(game_state),
            0,
        )

    def get_action_value_prior(self, game_state: GameState) -> np.ndarray:
        return self.action_value_priors.get(
            self.get_cache_key(game_state),
            np.zeros(POLICY_MOVE_COUNT, dtype=np.float32),
        )

    def get_uncertainty_weight(self, game_state: GameState) -> float:
        return self.uncertainty_weights.get(
            self.get_cache_key(game_state),
            1,
        )

    def get_uncertainty_prediction(self, game_state: GameState) -> float:
        return self.uncertainty_predictions.get(
            self.get_cache_key(game_state),
            0,
        )

    def get_cache_key(
        self,
        game_state: GameState,
    ) -> tuple[bytes, int, int, tuple[int, ...]]:
        return (
            game_state.board.tobytes(),
            game_state.next_color,
            game_state.ko_move,
            tuple(game_state.move_history[-2:]),
        )

    def evaluate(self, game_state: GameState) -> tuple[np.ndarray, float]:
        return self.evaluate_batch([game_state])[0]

    def evaluate_batch(
        self,
        game_states: list[GameState],
    ) -> list[tuple[np.ndarray, float]]:
        evaluations: list[tuple[np.ndarray, float] | None] = []
        missing_game_states: list[GameState] = []

        for game_state in game_states:
            cached_evaluation = self.cache.get(self.get_cache_key(game_state))
            evaluations.append(cached_evaluation)

            if cached_evaluation is None:
                missing_game_states.append(game_state)

        if missing_game_states:
            self.evaluation_count += len(missing_game_states)
            base_features = [
                encode_moka_features(game_state)
                for game_state in missing_game_states
            ]
            symmetry_descriptors: list[tuple[int, bool]] = []

            use_fixed_symmetry = (
                self.symmetry_rotation_count != 0
                or self.should_flip_symmetry
            )
            use_transformed_symmetry = (
                self.use_symmetry_ensemble
                or use_fixed_symmetry
                or self.use_symmetry_pair
            )

            if use_transformed_symmetry:
                empty_policy = np.zeros(POLICY_MOVE_COUNT, dtype=np.float32)
                transformed_features: list[np.ndarray] = []

                for features in base_features:
                    if self.use_symmetry_ensemble:
                        symmetry_options = [
                            (rotation_count, should_flip)
                            for rotation_count in range(
                                BOARD_SYMMETRY_ROTATION_COUNT
                            )
                            for should_flip in (False, True)
                        ]
                    elif self.use_symmetry_pair:
                        symmetry_options = [
                            (0, False),
                            (
                                self.symmetry_rotation_count,
                                self.should_flip_symmetry,
                            )
                        ]
                    else:
                        symmetry_options = [
                            (
                                self.symmetry_rotation_count,
                                self.should_flip_symmetry,
                            )
                        ]

                    for rotation_count, should_flip in symmetry_options:
                        symmetry_features, _ = apply_board_symmetry(
                            features,
                            empty_policy,
                            rotation_count,
                            should_flip,
                        )
                        transformed_features.append(symmetry_features)
                        symmetry_descriptors.append(
                            (rotation_count, should_flip)
                        )

                features = np.stack(transformed_features)
            else:
                features = np.stack(base_features)

            feature_values = mx.array(features, dtype=mx.float32)
            action_value_array: np.ndarray | None = None
            optimistic_policy_logits = None
            log_uncertainty_array = None
            if self.uncertainty_coefficient > 0 or self.predict_uncertainty:
                if not isinstance(self.model, MokaUncertaintyNetwork):
                    raise ValueError(
                        "Uncertainty weighting requires an uncertainty model."
                    )
                policy_logits, values, log_uncertainties = (
                    self.model.get_uncertainty_outputs(feature_values)
                )
                mx.eval(policy_logits, values, log_uncertainties)
                value_array = np.asarray(values)
                score_array = None
                log_uncertainty_array = np.asarray(log_uncertainties)
            elif self.optimistic_policy_weight > 0:
                if not isinstance(self.model, MokaOptimisticPolicyNetwork):
                    raise ValueError(
                        "Optimistic policy requires an optimistic-policy model."
                    )
                policy_logits, values, optimistic_policy_logits = (
                    self.model.get_optimistic_policy_outputs(feature_values)
                )
                mx.eval(policy_logits, values, optimistic_policy_logits)
                value_array = np.asarray(values)
                score_array = None
            elif self.action_value_prior_weight > 0:
                if not isinstance(self.model, MokaActionValueNetwork):
                    raise ValueError(
                        "Action-value priors require an action-value model."
                    )
                policy_logits, values, action_values = (
                    self.model.get_action_value_outputs(feature_values)
                )
                mx.eval(policy_logits, values, action_values)
                value_array = np.asarray(values)
                score_array = None
                action_value_array = np.asarray(action_values)
            elif self.score_value_weight > 0:
                policy_logits, values, scores = (
                    self.model.get_search_outputs(feature_values)
                )
                mx.eval(policy_logits, values, scores)
                value_array = np.asarray(values)
                score_array: np.ndarray | None = np.asarray(scores)
            else:
                policy_logits, values = self.model(feature_values)
                mx.eval(policy_logits, values)
                value_array = np.asarray(values)
                score_array = None
            logits = np.asarray(policy_logits)
            if optimistic_policy_logits is not None:
                logits = (
                    (1 - self.optimistic_policy_weight) * logits
                    + self.optimistic_policy_weight
                    * np.asarray(optimistic_policy_logits)
                )
            maximum_logits = np.max(logits, axis=1, keepdims=True)
            policies = np.exp(logits - maximum_logits)
            policies /= np.sum(policies, axis=1, keepdims=True)

            if use_transformed_symmetry:
                symmetry_count = (
                    BOARD_SYMMETRY_ROTATION_COUNT
                    * BOARD_SYMMETRY_REFLECTION_COUNT
                    if self.use_symmetry_ensemble
                    else 2
                    if self.use_symmetry_pair
                    else 1
                )

                for missing_index, game_state in enumerate(missing_game_states):
                    symmetry_start = missing_index * symmetry_count
                    symmetry_end = symmetry_start + symmetry_count
                    aligned_policies = [
                        invert_policy_symmetry(
                            policies[symmetry_index],
                            symmetry_descriptors[symmetry_index][0],
                            symmetry_descriptors[symmetry_index][1],
                        )
                        for symmetry_index in range(
                            symmetry_start,
                            symmetry_end,
                        )
                    ]
                    aggregated_policy = aggregate_symmetry_policies(
                            aligned_policies,
                            self.symmetry_geometric_policy_weight,
                            self.symmetry_trimmed_policy_weight,
                            resolve_symmetry_rank_policy_weight(
                                self.symmetry_rank_policy_weight,
                                self.symmetry_rank_policy_end_move_count,
                                game_state.move_count,
                            ),
                            self.symmetry_rank_move_count,
                            self.symmetry_rank_minimum_top_move_vote_count,
                            self.symmetry_top_move_vote_policy_weight,
                        )
                    symmetry_value_slice = value_array[
                        symmetry_start:symmetry_end
                    ]
                    score_value_weight = resolve_score_value_weight(
                        self.score_value_weight,
                        self.score_value_start_move_count,
                        game_state.move_count,
                    )
                    if score_array is not None:
                        symmetry_value_slice = (
                            (1 - score_value_weight)
                            * symmetry_value_slice
                            + score_value_weight
                            * score_array[symmetry_start:symmetry_end]
                        )
                    value_spread = float(np.std(symmetry_value_slice))
                    cache_key = self.get_cache_key(game_state)
                    self.cache[cache_key] = (
                        apply_search_policy_temperature(
                            aggregated_policy,
                            self.policy_temperature,
                        ),
                        aggregate_symmetry_values(
                            symmetry_value_slice,
                            self.symmetry_trimmed_value_weight,
                        ),
                    )
                    if action_value_array is not None:
                        aligned_action_values = [
                            invert_policy_symmetry(
                                action_value_array[symmetry_index],
                                symmetry_descriptors[symmetry_index][0],
                                symmetry_descriptors[symmetry_index][1],
                            )
                            for symmetry_index in range(
                                symmetry_start,
                                symmetry_end,
                            )
                        ]
                        self.action_value_priors[cache_key] = (
                            self.action_value_prior_weight
                            * np.mean(aligned_action_values, axis=0)
                        )
                    self.symmetry_value_spreads[cache_key] = value_spread
                    if log_uncertainty_array is not None:
                        log_uncertainty_slice = log_uncertainty_array[
                            symmetry_start:symmetry_end
                        ]
                        self.uncertainty_predictions[cache_key] = (
                            calculate_uncertainty(log_uncertainty_slice)
                        )
                        self.uncertainty_weights[cache_key] = (
                            calculate_uncertainty_weight(
                                log_uncertainty_slice,
                                self.uncertainty_coefficient,
                                self.uncertainty_maximum_weight,
                            )
                        )
            else:
                for missing_index, game_state in enumerate(missing_game_states):
                    score_value_weight = resolve_score_value_weight(
                        self.score_value_weight,
                        self.score_value_start_move_count,
                        game_state.move_count,
                    )
                    blended_value = float(value_array[missing_index])
                    if score_array is not None:
                        blended_value = float(
                            (1 - score_value_weight)
                            * value_array[missing_index]
                            + score_value_weight * score_array[missing_index]
                        )
                    cache_key = self.get_cache_key(game_state)
                    self.cache[cache_key] = (
                        apply_search_policy_temperature(
                            policies[missing_index],
                            self.policy_temperature,
                        ),
                        blended_value,
                    )
                    if action_value_array is not None:
                        self.action_value_priors[cache_key] = (
                            self.action_value_prior_weight
                            * action_value_array[missing_index]
                        )
                    self.symmetry_value_spreads[cache_key] = 0
                    if log_uncertainty_array is not None:
                        log_uncertainty_slice = log_uncertainty_array[
                            missing_index : missing_index + 1
                        ]
                        self.uncertainty_predictions[cache_key] = (
                            calculate_uncertainty(log_uncertainty_slice)
                        )
                        self.uncertainty_weights[cache_key] = (
                            calculate_uncertainty_weight(
                                log_uncertainty_slice,
                                self.uncertainty_coefficient,
                                self.uncertainty_maximum_weight,
                            )
                        )

        return [
            self.cache[self.get_cache_key(game_state)] for game_state in game_states
        ]


def get_evaluator_action_value_prior(
    evaluator: object,
    game_state: GameState,
) -> np.ndarray:
    return (
        evaluator.get_action_value_prior(game_state)
        if isinstance(evaluator, MokaEvaluator)
        else np.zeros(POLICY_MOVE_COUNT, dtype=np.float32)
    )


def get_evaluator_uncertainty_weight(
    evaluator: object,
    game_state: GameState,
) -> float:
    return (
        evaluator.get_uncertainty_weight(game_state)
        if isinstance(evaluator, MokaEvaluator)
        else 1
    )


def calculate_uncertainty_weight(
    log_uncertainties: np.ndarray,
    coefficient: float = SEARCH_UNCERTAINTY_COEFFICIENT,
    maximum_weight: float = SEARCH_UNCERTAINTY_MAXIMUM_WEIGHT,
) -> float:
    if coefficient <= 0:
        return 1
    uncertainty = calculate_uncertainty(log_uncertainties)
    baseline_uncertainty = coefficient / maximum_weight
    return coefficient / (uncertainty + baseline_uncertainty)


def calculate_uncertainty(log_uncertainties: np.ndarray) -> float:
    uncertainties = np.maximum(
        np.exp(
            np.clip(
                log_uncertainties,
                SEARCH_UNCERTAINTY_LOG_MINIMUM,
                SEARCH_UNCERTAINTY_LOG_MAXIMUM,
            )
        )
        - SEARCH_UNCERTAINTY_TARGET_EPSILON,
        0,
    )
    return float(np.mean(uncertainties))


def resolve_score_value_weight(
    configured_weight: float,
    start_move_count: int,
    move_count: int,
) -> float:
    return configured_weight if move_count >= start_move_count else 0


def get_terminal_value(game_state: GameState) -> float:
    did_black_win = get_area_score(game_state) > 0
    did_current_player_win = did_black_win == (game_state.next_color == 1)
    return 1 if did_current_player_win else -1


def blend_search_value(
    game_state: GameState,
    network_value: float,
    area_value_weight: float,
) -> float:
    if area_value_weight == 0:
        return network_value

    perspective_area_score = get_area_score(game_state) * game_state.next_color
    area_value = float(
        np.tanh(perspective_area_score / SEARCH_AREA_VALUE_SCALE_POINTS)
    )
    phase_weight = np.clip(
        (
            game_state.move_count
            - SEARCH_AREA_VALUE_START_MOVE_COUNT
        )
        / SEARCH_AREA_VALUE_RAMP_MOVE_COUNT,
        0,
        1,
    )
    effective_area_value_weight = area_value_weight * phase_weight
    return (
        (1 - effective_area_value_weight) * network_value
        + effective_area_value_weight * area_value
    )


def evaluate_rollout_values(
    evaluator: MokaEvaluator,
    game_states: list[GameState],
    rollout_depth: int,
    area_value_weight: float,
) -> list[float]:
    rollout_states = game_states.copy()
    perspective_signs = np.ones(len(game_states), dtype=np.float32)
    resolved_values = np.zeros(len(game_states), dtype=np.float32)
    unresolved_indexes = list(range(len(game_states)))

    for rollout_step in range(rollout_depth + 1):
        if not unresolved_indexes:
            break

        active_states = [
            rollout_states[rollout_index]
            for rollout_index in unresolved_indexes
        ]
        evaluations = evaluator.evaluate_batch(active_states)
        next_unresolved_indexes: list[int] = []

        for active_index, rollout_index in enumerate(unresolved_indexes):
            game_state = rollout_states[rollout_index]

            if is_game_over(game_state):
                resolved_values[rollout_index] = (
                    perspective_signs[rollout_index]
                    * get_terminal_value(game_state)
                )
                continue

            policy, network_value = evaluations[active_index]

            if rollout_step == rollout_depth:
                resolved_values[rollout_index] = (
                    perspective_signs[rollout_index]
                    * blend_search_value(
                        game_state,
                        network_value,
                        area_value_weight,
                    )
                )
                continue

            legal_moves = get_legal_moves(game_state)
            selectable_moves = (
                [move for move in legal_moves if move != BOARD_AREA]
                if game_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT
                else legal_moves
            )
            move = max(
                selectable_moves,
                key=lambda selectable_move: float(policy[selectable_move]),
            )
            next_state = play_move(game_state, move)

            if next_state is None:
                next_state = play_move(game_state, BOARD_AREA)

            if next_state is None:
                resolved_values[rollout_index] = (
                    perspective_signs[rollout_index]
                    * blend_search_value(
                        game_state,
                        network_value,
                        area_value_weight,
                    )
                )
                continue

            rollout_states[rollout_index] = next_state
            perspective_signs[rollout_index] *= -1
            next_unresolved_indexes.append(rollout_index)

        unresolved_indexes = next_unresolved_indexes

    return [float(value) for value in resolved_values]


def expand_node_with_evaluation(
    node: SearchNode,
    policy: np.ndarray,
    root_player_color: int | None = None,
    opponent_branch_count: int = SEARCH_OPPONENT_BRANCH_COUNT,
    action_value_prior: np.ndarray | None = None,
) -> None:
    legal_move_states = get_legal_move_states(node.game_state)
    selectable_move_states = (
        [
            move_and_state
            for move_and_state in legal_move_states
            if move_and_state[0] != BOARD_AREA
        ]
        if node.game_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT
        else legal_move_states
    )

    if (
        root_player_color is not None
        and node.game_state.next_color != root_player_color
        and opponent_branch_count > 0
    ):
        selectable_move_states = sorted(
            selectable_move_states,
            key=lambda move_and_state: float(policy[move_and_state[0]]),
            reverse=True,
        )[:opponent_branch_count]

    selectable_moves = [
        move for move, _ in selectable_move_states
    ]
    prior_sum = float(np.sum(policy[selectable_moves]))
    action_value_center = (
        float(np.mean(action_value_prior[selectable_moves]))
        if action_value_prior is not None and selectable_moves
        else 0
    )

    for move, next_state in selectable_move_states:
        prior = (
            float(policy[move] / prior_sum)
            if prior_sum > 0
            else 1 / len(selectable_moves)
        )
        node.children[move] = SearchNode(
            game_state=next_state,
            prior=prior,
            action_value_prior=(
                float(action_value_prior[move] - action_value_center)
                if action_value_prior is not None
                else 0
            ),
        )


def expand_node(node: SearchNode, evaluator: MokaEvaluator) -> float:
    policy, value = evaluator.evaluate(node.game_state)
    expand_node_with_evaluation(
        node,
        policy,
        action_value_prior=get_evaluator_action_value_prior(
            evaluator,
            node.game_state,
        ),
    )
    return value


def adjust_root_tactical_priors(
    root: SearchNode,
    capture_prior_bonus: float,
    self_atari_prior_penalty: float,
) -> None:
    if capture_prior_bonus == 0 and self_atari_prior_penalty == 0:
        return

    opponent_color = -root.game_state.next_color
    opponent_stone_count = int(
        np.count_nonzero(root.game_state.board == opponent_color)
    )

    for move, child in root.children.items():
        next_opponent_stone_count = int(
            np.count_nonzero(child.game_state.board == opponent_color)
        )
        captured_stone_count = (
            opponent_stone_count - next_opponent_stone_count
        )
        is_non_capturing_self_atari = False

        if move < BOARD_AREA and captured_stone_count == 0:
            _, liberties = get_group(child.game_state.board, move)
            is_non_capturing_self_atari = len(liberties) == 1

        child.prior *= float(
            np.exp(
                capture_prior_bonus * captured_stone_count
                - self_atari_prior_penalty
                * int(is_non_capturing_self_atari)
            )
        )

    adjusted_prior_sum = sum(child.prior for child in root.children.values())

    if adjusted_prior_sum > 0:
        for child in root.children.values():
            child.prior /= adjusted_prior_sum


def prune_root_children(
    root: SearchNode,
    root_branch_count: int,
) -> None:
    if root_branch_count <= 0 or len(root.children) <= root_branch_count:
        return

    retained_moves = set(
        sorted(
            root.children,
            key=lambda move: root.children[move].prior,
            reverse=True,
        )[:root_branch_count]
    )
    root.children = {
        move: child
        for move, child in root.children.items()
        if move in retained_moves
    }


def apply_search_policy_temperature(
    policy: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Search policy temperature must be positive.")

    if temperature == 1:
        return policy

    tempered_policy = np.power(policy, 1 / temperature)
    tempered_policy_sum = float(np.sum(tempered_policy))
    return (
        tempered_policy / tempered_policy_sum
        if tempered_policy_sum > 0
        else policy
    )


def blend_child_q_value(
    child_value: float,
    parent_value: float,
    child_visit_count: float,
    pseudo_count: float,
) -> float:
    if pseudo_count < 0:
        raise ValueError("Child Q pseudo-count must not be negative.")
    if pseudo_count == 0:
        return child_value

    return (
        child_visit_count * child_value + pseudo_count * parent_value
    ) / (child_visit_count + pseudo_count)


def select_child(
    node: SearchNode,
    reservation_counts: dict[int, int] | None = None,
    exploration: float = SEARCH_PUCT_EXPLORATION,
    value_weight: float = SEARCH_PUCT_VALUE_WEIGHT,
    first_play_urgency_reduction: float = (
        SEARCH_FIRST_PLAY_URGENCY_REDUCTION
    ),
    use_first_play_urgency_prior_mass: bool = (
        SEARCH_FIRST_PLAY_URGENCY_USE_PRIOR_MASS
    ),
    q_value_normalization_weight: float = (
        SEARCH_Q_VALUE_NORMALIZATION_WEIGHT
    ),
    child_q_pseudo_count: float = SEARCH_CHILD_Q_PSEUDO_COUNT,
    utility_stdev_prior: float = SEARCH_PUCT_UTILITY_STDEV_PRIOR,
    utility_stdev_prior_weight: float = (
        SEARCH_PUCT_UTILITY_STDEV_PRIOR_WEIGHT
    ),
    utility_stdev_scale: float = SEARCH_PUCT_UTILITY_STDEV_SCALE,
) -> SearchNode:
    reservation_counts = reservation_counts or {}

    if (
        not reservation_counts
        and not use_first_play_urgency_prior_mass
        and q_value_normalization_weight == 0
        and child_q_pseudo_count == 0
        and utility_stdev_scale == 0
    ):
        parent_visit_scale = np.sqrt(
            max(node.effective_visit_weight, 1)
        )
        parent_value = node.mean_value
        selected_child: SearchNode | None = None
        selected_score = -np.inf

        for child in node.children.values():
            child_visit_weight = child.effective_visit_weight
            child_value = (
                parent_value
                - first_play_urgency_reduction
                + child.action_value_prior
                if (
                    child_visit_weight == 0
                    and first_play_urgency_reduction >= 0
                )
                else -child.mean_value
            )
            child_score = (
                value_weight * child_value
                + exploration
                * child.prior
                * parent_visit_scale
                / (child_visit_weight + 1)
            )

            if selected_child is None or child_score > selected_score:
                selected_child = child
                selected_score = child_score

        if selected_child is None:
            raise ValueError("Cannot select from a node without children.")

        return selected_child

    parent_visit_scale = np.sqrt(
        max(
            node.effective_visit_weight
            + reservation_counts.get(id(node), 0),
            1,
        )
    )
    dynamic_exploration = calculate_dynamic_exploration(
        exploration,
        node,
        utility_stdev_prior,
        utility_stdev_prior_weight,
        utility_stdev_scale,
    )
    visited_prior_mass = sum(
        child.prior
        for child in node.children.values()
        if (
            child.effective_visit_weight
            + reservation_counts.get(id(child), 0)
        )
        > 0
    )
    effective_first_play_urgency_reduction = (
        first_play_urgency_reduction * np.sqrt(visited_prior_mass)
        if use_first_play_urgency_prior_mass
        else first_play_urgency_reduction
    )
    visited_child_values = [
        -child.mean_value
        for child in node.children.values()
        if (
            child.effective_visit_weight
            + reservation_counts.get(id(child), 0)
        )
        > 0
    ]
    minimum_child_value = (
        min(visited_child_values)
        if len(visited_child_values) >= 2
        else 0
    )
    maximum_child_value = (
        max(visited_child_values)
        if len(visited_child_values) >= 2
        else 0
    )
    child_value_range = maximum_child_value - minimum_child_value

    def get_child_score(child: SearchNode) -> float:
        child_reservation_count = reservation_counts.get(id(child), 0)
        effective_child_visit_count = (
            child.effective_visit_weight + child_reservation_count
        )
        raw_child_value = (
            node.mean_value
            - effective_first_play_urgency_reduction
            + child.action_value_prior
            if (
                effective_child_visit_count == 0
                and first_play_urgency_reduction >= 0
            )
            else -child.mean_value
        )
        child_value = (
            blend_child_q_value(
                raw_child_value,
                node.mean_value,
                child.effective_visit_weight,
                child_q_pseudo_count,
            )
            if child.effective_visit_weight > 0
            else raw_child_value
        )
        normalized_child_value = (
            float(
                np.clip(
                    (child_value - minimum_child_value) / child_value_range,
                    0,
                    1,
                )
            )
            if (
                q_value_normalization_weight > 0
                and child_value_range
                > SEARCH_Q_VALUE_NORMALIZATION_EPSILON
            )
            else child_value
        )
        effective_child_value = (
            (1 - q_value_normalization_weight) * child_value
            + q_value_normalization_weight * normalized_child_value
        )
        return (
            value_weight * effective_child_value
            + dynamic_exploration
            * child.prior
            * parent_visit_scale
            / (effective_child_visit_count + 1)
        )

    return max(
        node.children.values(),
        key=get_child_score,
    )


def calculate_dynamic_exploration(
    exploration: float,
    node: SearchNode,
    utility_stdev_prior: float,
    utility_stdev_prior_weight: float,
    utility_stdev_scale: float,
) -> float:
    if utility_stdev_prior <= 0:
        raise ValueError("Utility standard-deviation prior must be positive.")
    if utility_stdev_prior_weight < 0:
        raise ValueError("Utility prior weight must not be negative.")
    if not 0 <= utility_stdev_scale <= 1:
        raise ValueError("Utility standard-deviation scale must be in [0, 1].")
    if utility_stdev_scale == 0 or node.effective_visit_weight <= 1:
        return exploration

    visit_weight = node.effective_visit_weight
    mean_value = node.mean_value
    mean_square_value = node.value_square_sum / visit_weight
    mean_square_value = max(mean_square_value, mean_value * mean_value)
    prior_variance = utility_stdev_prior * utility_stdev_prior
    utility_variance = max(
        0,
        (
            (mean_value * mean_value + prior_variance)
            * utility_stdev_prior_weight
            + mean_square_value * visit_weight
        )
        / (utility_stdev_prior_weight + visit_weight - 1)
        - mean_value * mean_value,
    )
    utility_stdev = np.sqrt(utility_variance)
    exploration_factor = 1 + utility_stdev_scale * (
        utility_stdev / utility_stdev_prior - 1
    )
    return exploration * exploration_factor


def calculate_child_lcb(
    child: SearchNode,
    lcb_stdevs: float,
) -> tuple[float, float]:
    if lcb_stdevs < 0:
        raise ValueError("LCB standard deviations must not be negative.")
    visit_weight = child.effective_visit_weight
    weight_square_sum = child.effective_visit_weight_square_sum
    if visit_weight <= 0 or weight_square_sum <= 0:
        maximum_radius = 2 * lcb_stdevs
        return -maximum_radius, maximum_radius

    effective_sample_size = visit_weight * visit_weight / weight_square_sum
    mean_value = child.mean_value
    mean_square_value = max(
        child.value_square_sum / visit_weight,
        mean_value * mean_value + SEARCH_ROOT_LCB_VARIANCE_EPSILON,
    )
    prior_weight = visit_weight / effective_sample_size**3
    mean_square_value = (
        mean_square_value * visit_weight
        + (mean_square_value + 1) * prior_weight
    ) / (visit_weight + prior_weight)
    visit_weight += prior_weight
    weight_square_sum += prior_weight * prior_weight
    effective_sample_size = visit_weight * visit_weight / weight_square_sum
    utility_variance = max(
        mean_square_value - mean_value * mean_value,
        0,
    )
    radius = np.sqrt(utility_variance / effective_sample_size) * lcb_stdevs
    root_value = -mean_value
    return root_value - radius, radius


def select_root_child_with_lcb(
    root_children: list[tuple[int, SearchNode]],
    lcb_stdevs: float,
    minimum_visit_proportion: float,
) -> tuple[int, SearchNode]:
    if not 0 <= minimum_visit_proportion <= 1:
        raise ValueError("Minimum LCB visit proportion must be in [0, 1].")
    if lcb_stdevs <= 0:
        return max(
            root_children,
            key=lambda move_and_child: (
                move_and_child[1].effective_visit_weight
            ),
        )

    maximum_visit_weight = max(
        child.effective_visit_weight for _, child in root_children
    )
    lcb_values = [
        calculate_child_lcb(child, lcb_stdevs)
        for _, child in root_children
    ]
    eligible_indexes = [
        child_index
        for child_index, (_, child) in enumerate(root_children)
        if child.effective_visit_weight
        >= minimum_visit_proportion * maximum_visit_weight
    ]
    best_lcb_index = max(
        eligible_indexes,
        key=lambda child_index: lcb_values[child_index][0],
    )
    best_lcb = lcb_values[best_lcb_index][0]
    adjusted_visit_weight = root_children[
        best_lcb_index
    ][1].effective_visit_weight

    for child_index, (_, child) in enumerate(root_children):
        if child_index == best_lcb_index:
            continue
        excess_value = best_lcb - lcb_values[child_index][0]
        if excess_value < 0:
            continue
        radius = lcb_values[child_index][1]
        radius_factor = (radius + excess_value) / (
            radius
            + excess_value / SEARCH_ROOT_LCB_WEIGHT_GAIN_LIMIT
        )
        adjusted_visit_weight = max(
            adjusted_visit_weight,
            radius_factor
            * radius_factor
            * child.effective_visit_weight,
        )

    selection_weights = [
        (
            adjusted_visit_weight
            if child_index == best_lcb_index
            else child.effective_visit_weight
        )
        for child_index, (_, child) in enumerate(root_children)
    ]
    selected_index = max(
        range(len(root_children)),
        key=lambda child_index: selection_weights[child_index],
    )
    return root_children[selected_index]


def resolve_first_play_urgency_reduction(
    first_play_urgency_reduction: float,
    use_first_play_urgency_at_root_only: bool,
    is_root: bool,
) -> float:
    return (
        first_play_urgency_reduction
        if is_root or not use_first_play_urgency_at_root_only
        else -1.0
    )


def resolve_search_exploration(
    exploration: float,
    descendant_exploration: float | None,
    is_root: bool,
) -> float:
    return (
        exploration
        if is_root or descendant_exploration is None
        else descendant_exploration
    )


def resolve_q_value_normalization_weight(
    q_value_normalization_weight: float,
    use_q_value_normalization_at_root_only: bool,
    is_root: bool,
) -> float:
    return (
        q_value_normalization_weight
        if is_root or not use_q_value_normalization_at_root_only
        else 0
    )


def resolve_root_policy_temperature(
    root_policy_temperature: float,
    root_policy_temperature_end_move_count: int,
    move_count: int,
) -> float:
    return (
        root_policy_temperature
        if (
            root_policy_temperature_end_move_count <= 0
            or move_count < root_policy_temperature_end_move_count
        )
        else 1.0
    )


def run_simulation(
    node: SearchNode,
    evaluator: MokaEvaluator,
    exploration: float = SEARCH_PUCT_EXPLORATION,
    value_weight: float = SEARCH_PUCT_VALUE_WEIGHT,
    area_value_weight: float = SEARCH_AREA_VALUE_WEIGHT,
    rollout_depth: int = SEARCH_ROLLOUT_DEPTH,
    root_player_color: int | None = None,
    opponent_branch_count: int = SEARCH_OPPONENT_BRANCH_COUNT,
    first_play_urgency_reduction: float = (
        SEARCH_FIRST_PLAY_URGENCY_REDUCTION
    ),
    use_first_play_urgency_prior_mass: bool = (
        SEARCH_FIRST_PLAY_URGENCY_USE_PRIOR_MASS
    ),
    use_first_play_urgency_at_root_only: bool = (
        SEARCH_FIRST_PLAY_URGENCY_ROOT_ONLY
    ),
    q_value_normalization_weight: float = (
        SEARCH_Q_VALUE_NORMALIZATION_WEIGHT
    ),
    use_q_value_normalization_at_root_only: bool = (
        SEARCH_Q_VALUE_NORMALIZATION_ROOT_ONLY
    ),
    is_root: bool = True,
    descendant_exploration: float | None = (
        SEARCH_DESCENDANT_PUCT_EXPLORATION
    ),
    child_q_pseudo_count: float = SEARCH_CHILD_Q_PSEUDO_COUNT,
    utility_stdev_prior: float = SEARCH_PUCT_UTILITY_STDEV_PRIOR,
    utility_stdev_prior_weight: float = (
        SEARCH_PUCT_UTILITY_STDEV_PRIOR_WEIGHT
    ),
    utility_stdev_scale: float = SEARCH_PUCT_UTILITY_STDEV_SCALE,
) -> tuple[float, float]:
    if is_game_over(node.game_state):
        value = get_terminal_value(node.game_state)
        uncertainty_weight = 1
    elif not node.children:
        policy, network_value = evaluator.evaluate(node.game_state)
        expand_node_with_evaluation(
            node,
            policy,
            root_player_color,
            opponent_branch_count,
            get_evaluator_action_value_prior(evaluator, node.game_state),
        )
        value = (
            evaluate_rollout_values(
                evaluator,
                [node.game_state],
                rollout_depth,
                area_value_weight,
            )[0]
            if rollout_depth > 0
            else blend_search_value(
                node.game_state,
                network_value,
                area_value_weight,
            )
        )
        uncertainty_weight = get_evaluator_uncertainty_weight(
            evaluator,
            node.game_state,
        )
    else:
        child = select_child(
            node,
            exploration=resolve_search_exploration(
                exploration,
                descendant_exploration,
                is_root,
            ),
            value_weight=value_weight,
            first_play_urgency_reduction=(
                resolve_first_play_urgency_reduction(
                    first_play_urgency_reduction,
                    use_first_play_urgency_at_root_only,
                    is_root,
                )
            ),
            use_first_play_urgency_prior_mass=(
                use_first_play_urgency_prior_mass
            ),
            q_value_normalization_weight=(
                resolve_q_value_normalization_weight(
                    q_value_normalization_weight,
                    use_q_value_normalization_at_root_only,
                    is_root,
                )
            ),
            child_q_pseudo_count=child_q_pseudo_count,
            utility_stdev_prior=utility_stdev_prior,
            utility_stdev_prior_weight=utility_stdev_prior_weight,
            utility_stdev_scale=utility_stdev_scale,
        )
        child_value, uncertainty_weight = run_simulation(
            child,
            evaluator,
            exploration,
            value_weight,
            area_value_weight,
            rollout_depth,
            root_player_color,
            opponent_branch_count,
            first_play_urgency_reduction,
            use_first_play_urgency_prior_mass,
            use_first_play_urgency_at_root_only,
            q_value_normalization_weight,
            use_q_value_normalization_at_root_only,
            is_root=False,
            descendant_exploration=descendant_exploration,
            child_q_pseudo_count=child_q_pseudo_count,
            utility_stdev_prior=utility_stdev_prior,
            utility_stdev_prior_weight=utility_stdev_prior_weight,
            utility_stdev_scale=utility_stdev_scale,
        )
        value = -child_value

    backup_search_node(node, value, uncertainty_weight)
    return value, uncertainty_weight


def backup_search_node(
    node: SearchNode,
    value: float,
    weight: float,
) -> None:
    node.visit_count += 1
    node.visit_weight += weight
    node.visit_weight_square_sum += weight * weight
    node.value_sum += weight * value
    node.value_square_sum += weight * value * value


def run_simulation_batch(
    root: SearchNode,
    evaluator: MokaEvaluator,
    simulation_count: int,
    exploration: float = SEARCH_PUCT_EXPLORATION,
    value_weight: float = SEARCH_PUCT_VALUE_WEIGHT,
    area_value_weight: float = SEARCH_AREA_VALUE_WEIGHT,
    rollout_depth: int = SEARCH_ROLLOUT_DEPTH,
    root_player_color: int | None = None,
    opponent_branch_count: int = SEARCH_OPPONENT_BRANCH_COUNT,
    first_play_urgency_reduction: float = (
        SEARCH_FIRST_PLAY_URGENCY_REDUCTION
    ),
    use_first_play_urgency_prior_mass: bool = (
        SEARCH_FIRST_PLAY_URGENCY_USE_PRIOR_MASS
    ),
    use_first_play_urgency_at_root_only: bool = (
        SEARCH_FIRST_PLAY_URGENCY_ROOT_ONLY
    ),
    q_value_normalization_weight: float = (
        SEARCH_Q_VALUE_NORMALIZATION_WEIGHT
    ),
    use_q_value_normalization_at_root_only: bool = (
        SEARCH_Q_VALUE_NORMALIZATION_ROOT_ONLY
    ),
    descendant_exploration: float | None = (
        SEARCH_DESCENDANT_PUCT_EXPLORATION
    ),
    child_q_pseudo_count: float = SEARCH_CHILD_Q_PSEUDO_COUNT,
    utility_stdev_prior: float = SEARCH_PUCT_UTILITY_STDEV_PRIOR,
    utility_stdev_prior_weight: float = (
        SEARCH_PUCT_UTILITY_STDEV_PRIOR_WEIGHT
    ),
    utility_stdev_scale: float = SEARCH_PUCT_UTILITY_STDEV_SCALE,
) -> None:
    reservation_counts: dict[int, int] = {}
    search_paths: list[list[SearchNode]] = []

    for _ in range(simulation_count):
        node = root
        search_path = [node]

        while node.children and not is_game_over(node.game_state):
            node = select_child(
                node,
                reservation_counts,
                resolve_search_exploration(
                    exploration,
                    descendant_exploration,
                    node is root,
                ),
                value_weight,
                resolve_first_play_urgency_reduction(
                    first_play_urgency_reduction,
                    use_first_play_urgency_at_root_only,
                    node is root,
                ),
                use_first_play_urgency_prior_mass,
                resolve_q_value_normalization_weight(
                    q_value_normalization_weight,
                    use_q_value_normalization_at_root_only,
                    node is root,
                ),
                child_q_pseudo_count,
                utility_stdev_prior,
                utility_stdev_prior_weight,
                utility_stdev_scale,
            )
            search_path.append(node)

        for path_node in search_path:
            node_identifier = id(path_node)
            reservation_counts[node_identifier] = (
                reservation_counts.get(node_identifier, 0) + 1
            )

        search_paths.append(search_path)

    unevaluated_nodes: list[SearchNode] = []
    seen_node_identifiers: set[int] = set()

    for search_path in search_paths:
        leaf_node = search_path[-1]
        leaf_identifier = id(leaf_node)

        if (
            not is_game_over(leaf_node.game_state)
            and not leaf_node.children
            and leaf_identifier not in seen_node_identifiers
        ):
            unevaluated_nodes.append(leaf_node)
            seen_node_identifiers.add(leaf_identifier)

    evaluations = evaluator.evaluate_batch(
        [node.game_state for node in unevaluated_nodes]
    )

    for node, evaluation in zip(unevaluated_nodes, evaluations, strict=True):
        expand_node_with_evaluation(
            node,
            evaluation[0],
            root_player_color,
            opponent_branch_count,
            get_evaluator_action_value_prior(evaluator, node.game_state),
        )

    rollout_values = (
        evaluate_rollout_values(
            evaluator,
            [node.game_state for node in unevaluated_nodes],
            rollout_depth,
            area_value_weight,
        )
        if rollout_depth > 0
        else [
            blend_search_value(
                node.game_state,
                evaluation[1],
                area_value_weight,
            )
            for node, evaluation in zip(
                unevaluated_nodes,
                evaluations,
                strict=True,
            )
        ]
    )
    rollout_value_by_identifier = {
        id(node): rollout_value
        for node, rollout_value in zip(
            unevaluated_nodes,
            rollout_values,
            strict=True,
        )
    }
    rollout_weight_by_identifier = {
        id(node): get_evaluator_uncertainty_weight(
            evaluator,
            node.game_state,
        )
        for node in unevaluated_nodes
    }

    for search_path in search_paths:
        leaf_node = search_path[-1]
        value = (
            get_terminal_value(leaf_node.game_state)
            if is_game_over(leaf_node.game_state)
            else rollout_value_by_identifier[id(leaf_node)]
        )
        uncertainty_weight = (
            1
            if is_game_over(leaf_node.game_state)
            else rollout_weight_by_identifier[id(leaf_node)]
        )

        for path_node in reversed(search_path):
            backup_search_node(path_node, value, uncertainty_weight)
            value = -value


def run_search_simulations(
    root: SearchNode,
    evaluator: MokaEvaluator,
    simulation_count: int,
    exploration: float = SEARCH_PUCT_EXPLORATION,
    value_weight: float = SEARCH_PUCT_VALUE_WEIGHT,
    area_value_weight: float = SEARCH_AREA_VALUE_WEIGHT,
    rollout_depth: int = SEARCH_ROLLOUT_DEPTH,
    root_player_color: int | None = None,
    opponent_branch_count: int = SEARCH_OPPONENT_BRANCH_COUNT,
    first_play_urgency_reduction: float = (
        SEARCH_FIRST_PLAY_URGENCY_REDUCTION
    ),
    use_first_play_urgency_prior_mass: bool = (
        SEARCH_FIRST_PLAY_URGENCY_USE_PRIOR_MASS
    ),
    use_first_play_urgency_at_root_only: bool = (
        SEARCH_FIRST_PLAY_URGENCY_ROOT_ONLY
    ),
    q_value_normalization_weight: float = (
        SEARCH_Q_VALUE_NORMALIZATION_WEIGHT
    ),
    use_q_value_normalization_at_root_only: bool = (
        SEARCH_Q_VALUE_NORMALIZATION_ROOT_ONLY
    ),
    descendant_exploration: float | None = (
        SEARCH_DESCENDANT_PUCT_EXPLORATION
    ),
    child_q_pseudo_count: float = SEARCH_CHILD_Q_PSEUDO_COUNT,
    maximum_extra_simulation_count: int = (
        SEARCH_MAXIMUM_EXTRA_EVALUATION_BUDGET_SIMULATION_COUNT
    ),
    utility_stdev_prior: float = SEARCH_PUCT_UTILITY_STDEV_PRIOR,
    utility_stdev_prior_weight: float = (
        SEARCH_PUCT_UTILITY_STDEV_PRIOR_WEIGHT
    ),
    utility_stdev_scale: float = SEARCH_PUCT_UTILITY_STDEV_SCALE,
) -> None:
    if maximum_extra_simulation_count < 0:
        raise ValueError("Maximum extra simulation count must be nonnegative.")
    starting_evaluation_count = (
        evaluator.evaluation_count
        if maximum_extra_simulation_count > 0
        else 0
    )
    remaining_simulation_count = simulation_count

    if remaining_simulation_count > 0 and root.visit_count == 0:
        run_simulation(
            root,
            evaluator,
            exploration,
            value_weight,
            area_value_weight,
            rollout_depth,
            root_player_color,
            opponent_branch_count,
            first_play_urgency_reduction,
            use_first_play_urgency_prior_mass,
            use_first_play_urgency_at_root_only,
            q_value_normalization_weight,
            use_q_value_normalization_at_root_only,
            descendant_exploration=descendant_exploration,
            child_q_pseudo_count=child_q_pseudo_count,
            utility_stdev_prior=utility_stdev_prior,
            utility_stdev_prior_weight=utility_stdev_prior_weight,
            utility_stdev_scale=utility_stdev_scale,
        )
        remaining_simulation_count -= 1

    while remaining_simulation_count > 0:
        batch_simulation_count = min(
            SEARCH_SIMULATION_BATCH_SIZE,
            remaining_simulation_count,
        )
        run_simulation_batch(
            root,
            evaluator,
            batch_simulation_count,
            exploration,
            value_weight,
            area_value_weight,
            rollout_depth,
            root_player_color,
            opponent_branch_count,
            first_play_urgency_reduction,
            use_first_play_urgency_prior_mass,
            use_first_play_urgency_at_root_only,
            q_value_normalization_weight,
            use_q_value_normalization_at_root_only,
            descendant_exploration,
            child_q_pseudo_count,
            utility_stdev_prior,
            utility_stdev_prior_weight,
            utility_stdev_scale,
        )
        remaining_simulation_count -= batch_simulation_count

    extra_simulation_count = 0

    while (
        extra_simulation_count < maximum_extra_simulation_count
        and evaluator.evaluation_count - starting_evaluation_count
        < simulation_count
    ):
        run_simulation_batch(
            root,
            evaluator,
            1,
            exploration,
            value_weight,
            area_value_weight,
            rollout_depth,
            root_player_color,
            opponent_branch_count,
            first_play_urgency_reduction,
            use_first_play_urgency_prior_mass,
            use_first_play_urgency_at_root_only,
            q_value_normalization_weight,
            use_q_value_normalization_at_root_only,
            descendant_exploration,
            child_q_pseudo_count,
            utility_stdev_prior,
            utility_stdev_prior_weight,
            utility_stdev_scale,
        )
        extra_simulation_count += 1


def select_search_move(
    evaluator: MokaEvaluator,
    game_state: GameState,
    simulation_count: int,
    exploration: float = SEARCH_PUCT_EXPLORATION,
    value_weight: float = SEARCH_PUCT_VALUE_WEIGHT,
    area_value_weight: float = SEARCH_AREA_VALUE_WEIGHT,
    rollout_depth: int = SEARCH_ROLLOUT_DEPTH,
    first_play_urgency_reduction: float = (
        SEARCH_FIRST_PLAY_URGENCY_REDUCTION
    ),
    use_first_play_urgency_prior_mass: bool = (
        SEARCH_FIRST_PLAY_URGENCY_USE_PRIOR_MASS
    ),
    use_first_play_urgency_at_root_only: bool = (
        SEARCH_FIRST_PLAY_URGENCY_ROOT_ONLY
    ),
    q_value_normalization_weight: float = (
        SEARCH_Q_VALUE_NORMALIZATION_WEIGHT
    ),
    use_q_value_normalization_at_root_only: bool = (
        SEARCH_Q_VALUE_NORMALIZATION_ROOT_ONLY
    ),
    child_q_pseudo_count: float = SEARCH_CHILD_Q_PSEUDO_COUNT,
) -> int:
    root = SearchNode(game_state=game_state, prior=1)
    run_search_simulations(
        root,
        evaluator,
        simulation_count,
        exploration,
        value_weight,
        area_value_weight,
        rollout_depth,
        first_play_urgency_reduction=first_play_urgency_reduction,
        use_first_play_urgency_prior_mass=(
            use_first_play_urgency_prior_mass
        ),
        use_first_play_urgency_at_root_only=(
            use_first_play_urgency_at_root_only
        ),
        q_value_normalization_weight=q_value_normalization_weight,
        use_q_value_normalization_at_root_only=(
            use_q_value_normalization_at_root_only
        ),
        child_q_pseudo_count=child_q_pseudo_count,
    )

    if not root.children:
        return BOARD_AREA

    return max(
        root.children.items(),
        key=lambda move_and_child: (
            move_and_child[1].effective_visit_weight
        ),
    )[0]


class MokaSearchSession:
    def __init__(
        self,
        evaluator: MokaEvaluator,
        exploration: float = SEARCH_PUCT_EXPLORATION,
        descendant_exploration: float | None = (
            SEARCH_DESCENDANT_PUCT_EXPLORATION
        ),
        child_q_pseudo_count: float = SEARCH_CHILD_Q_PSEUDO_COUNT,
        maximum_extra_simulation_count: int = (
            SEARCH_MAXIMUM_EXTRA_EVALUATION_BUDGET_SIMULATION_COUNT
        ),
        use_saved_root_evaluation_budget: bool = False,
        value_weight: float = SEARCH_PUCT_VALUE_WEIGHT,
        area_value_weight: float = SEARCH_AREA_VALUE_WEIGHT,
        rollout_depth: int = SEARCH_ROLLOUT_DEPTH,
        adaptive_max_simulation_count: int = (
            SEARCH_ADAPTIVE_MAX_SIMULATION_COUNT
        ),
        adaptive_visit_margin_ratio: float = (
            SEARCH_ADAPTIVE_VISIT_MARGIN_RATIO
        ),
        adaptive_symmetry_value_spread_threshold: float = (
            SEARCH_ADAPTIVE_SYMMETRY_VALUE_SPREAD_THRESHOLD
        ),
        adaptive_uncertainty_threshold: float = (
            SEARCH_ADAPTIVE_UNCERTAINTY_THRESHOLD
        ),
        opponent_branch_count: int = SEARCH_OPPONENT_BRANCH_COUNT,
        root_evaluator: MokaEvaluator | None = None,
        root_selection_visit_slack: int = -1,
        root_capture_prior_bonus: float = 0,
        root_self_atari_prior_penalty: float = 0,
        first_play_urgency_reduction: float = (
            SEARCH_FIRST_PLAY_URGENCY_REDUCTION
        ),
        use_first_play_urgency_prior_mass: bool = (
            SEARCH_FIRST_PLAY_URGENCY_USE_PRIOR_MASS
        ),
        use_first_play_urgency_at_root_only: bool = (
            SEARCH_FIRST_PLAY_URGENCY_ROOT_ONLY
        ),
        root_branch_count: int = SEARCH_ROOT_BRANCH_COUNT,
        root_policy_temperature: float = SEARCH_ROOT_POLICY_TEMPERATURE,
        root_policy_temperature_end_move_count: int = (
            SEARCH_ROOT_POLICY_TEMPERATURE_END_MOVE_COUNT
        ),
        q_value_normalization_weight: float = (
            SEARCH_Q_VALUE_NORMALIZATION_WEIGHT
        ),
        use_q_value_normalization_at_root_only: bool = (
            SEARCH_Q_VALUE_NORMALIZATION_ROOT_ONLY
        ),
        utility_stdev_prior: float = SEARCH_PUCT_UTILITY_STDEV_PRIOR,
        utility_stdev_prior_weight: float = (
            SEARCH_PUCT_UTILITY_STDEV_PRIOR_WEIGHT
        ),
        utility_stdev_scale: float = SEARCH_PUCT_UTILITY_STDEV_SCALE,
        root_lcb_stdevs: float = SEARCH_ROOT_LCB_STDEVS,
        root_lcb_minimum_visit_proportion: float = (
            SEARCH_ROOT_LCB_MINIMUM_VISIT_PROPORTION
        ),
    ) -> None:
        self.evaluator = evaluator
        self.exploration = exploration
        self.descendant_exploration = descendant_exploration
        self.child_q_pseudo_count = child_q_pseudo_count
        self.maximum_extra_simulation_count = maximum_extra_simulation_count
        self.use_saved_root_evaluation_budget = (
            use_saved_root_evaluation_budget
        )
        self.value_weight = value_weight
        self.area_value_weight = area_value_weight
        self.rollout_depth = rollout_depth
        self.adaptive_max_simulation_count = adaptive_max_simulation_count
        self.adaptive_visit_margin_ratio = adaptive_visit_margin_ratio
        self.adaptive_symmetry_value_spread_threshold = (
            adaptive_symmetry_value_spread_threshold
        )
        self.adaptive_uncertainty_threshold = adaptive_uncertainty_threshold
        if (
            np.isfinite(adaptive_uncertainty_threshold)
            and root_evaluator is None
        ):
            raise ValueError(
                "Adaptive uncertainty requires a root evaluator."
            )
        self.adaptive_extension_count = 0
        self.adaptive_extra_simulation_count = 0
        self.opponent_branch_count = opponent_branch_count
        self.root_evaluator = root_evaluator
        self.root_selection_visit_slack = root_selection_visit_slack
        self.root_capture_prior_bonus = root_capture_prior_bonus
        self.root_self_atari_prior_penalty = root_self_atari_prior_penalty
        self.first_play_urgency_reduction = (
            first_play_urgency_reduction
        )
        self.use_first_play_urgency_prior_mass = (
            use_first_play_urgency_prior_mass
        )
        self.use_first_play_urgency_at_root_only = (
            use_first_play_urgency_at_root_only
        )
        self.root_branch_count = root_branch_count
        self.root_policy_temperature = root_policy_temperature
        self.root_policy_temperature_end_move_count = (
            root_policy_temperature_end_move_count
        )
        self.q_value_normalization_weight = q_value_normalization_weight
        self.use_q_value_normalization_at_root_only = (
            use_q_value_normalization_at_root_only
        )
        self.utility_stdev_prior = utility_stdev_prior
        self.utility_stdev_prior_weight = utility_stdev_prior_weight
        self.utility_stdev_scale = utility_stdev_scale
        self.root_lcb_stdevs = root_lcb_stdevs
        self.root_lcb_minimum_visit_proportion = (
            root_lcb_minimum_visit_proportion
        )
        self.root: SearchNode | None = None

    def align_root(self, game_state: GameState) -> SearchNode:
        if self.root is not None:
            if self.root.game_state.move_history == game_state.move_history:
                return self.root

            for child in self.root.children.values():
                if child.game_state.move_history == game_state.move_history:
                    self.root = child
                    return child

        self.root = SearchNode(game_state=game_state, prior=1)
        return self.root

    def refresh_root_evaluation(
        self,
        root: SearchNode,
        game_state: GameState,
    ) -> int:
        if self.root_evaluator is None:
            return 0

        policy, network_value = self.root_evaluator.evaluate(game_state)
        action_value_prior = (
            get_evaluator_action_value_prior(
                self.root_evaluator,
                game_state,
            )
        )
        policy = apply_search_policy_temperature(
            policy,
            resolve_root_policy_temperature(
                self.root_policy_temperature,
                self.root_policy_temperature_end_move_count,
                game_state.move_count,
            ),
        )

        if root.children:
            prior_sum = float(
                np.sum(policy[list(root.children)])
            )

            action_value_center = float(
                np.mean(action_value_prior[list(root.children)])
            )
            for move, child in root.children.items():
                child.prior = (
                    float(policy[move] / prior_sum)
                    if prior_sum > 0
                    else 1 / len(root.children)
                )
                child.action_value_prior = float(
                    action_value_prior[move] - action_value_center
                )

            adjust_root_tactical_priors(
                root,
                self.root_capture_prior_bonus,
                self.root_self_atari_prior_penalty,
            )
            prune_root_children(root, self.root_branch_count)
            return 0

        expand_node_with_evaluation(
            root,
            policy,
            game_state.next_color,
            self.opponent_branch_count,
            action_value_prior,
        )
        adjust_root_tactical_priors(
            root,
            self.root_capture_prior_bonus,
            self.root_self_atari_prior_penalty,
        )
        prune_root_children(root, self.root_branch_count)
        backup_search_node(
            root,
            blend_search_value(
                game_state,
                network_value,
                self.area_value_weight,
            ),
            get_evaluator_uncertainty_weight(
                self.root_evaluator,
                game_state,
            ),
        )
        return 1

    def select_move(
        self,
        game_state: GameState,
        simulation_count: int,
    ) -> int:
        move, _ = self.select_move_with_policy(game_state, simulation_count)
        return move

    def select_move_with_policy(
        self,
        game_state: GameState,
        simulation_count: int,
    ) -> tuple[int, np.ndarray]:
        move, policy, _, _ = self.select_move_with_search_targets(
            game_state,
            simulation_count,
        )
        return move, policy

    def select_move_with_search_targets(
        self,
        game_state: GameState,
        simulation_count: int,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        root = self.align_root(game_state)
        starting_root_evaluation_count = (
            self.root_evaluator.evaluation_count
            if self.use_saved_root_evaluation_budget
            and self.root_evaluator is not None
            else 0
        )
        root_evaluation_count = self.refresh_root_evaluation(
            root,
            game_state,
        )
        did_reuse_root_evaluation = (
            self.use_saved_root_evaluation_budget
            and self.root_evaluator is not None
            and self.root_evaluator.evaluation_count
            == starting_root_evaluation_count
        )
        run_search_simulations(
            root,
            self.evaluator,
            max(simulation_count - root_evaluation_count, 0)
            + int(did_reuse_root_evaluation),
            self.exploration,
            self.value_weight,
            self.area_value_weight,
            self.rollout_depth,
            game_state.next_color,
            self.opponent_branch_count,
            self.first_play_urgency_reduction,
            self.use_first_play_urgency_prior_mass,
            self.use_first_play_urgency_at_root_only,
            self.q_value_normalization_weight,
            self.use_q_value_normalization_at_root_only,
            self.descendant_exploration,
            self.child_q_pseudo_count,
            self.maximum_extra_simulation_count,
            self.utility_stdev_prior,
            self.utility_stdev_prior_weight,
            self.utility_stdev_scale,
        )
        ordered_visit_counts = sorted(
            (
                child.effective_visit_weight
                for child in root.children.values()
            ),
            reverse=True,
        )

        if self.adaptive_max_simulation_count > simulation_count:
            should_extend_search = False

            if len(ordered_visit_counts) >= 2:
                leading_visit_count = ordered_visit_counts[0]
                second_visit_count = ordered_visit_counts[1]
                visit_margin_ratio = (
                    leading_visit_count - second_visit_count
                ) / max(leading_visit_count + second_visit_count, 1)
                should_extend_search = (
                    visit_margin_ratio < self.adaptive_visit_margin_ratio
                )

            if self.root_evaluator is not None:
                symmetry_value_spread = (
                    self.root_evaluator.get_symmetry_value_spread(
                        game_state
                    )
                )
                should_extend_search = (
                    should_extend_search
                    or symmetry_value_spread
                    >= self.adaptive_symmetry_value_spread_threshold
                    or (
                        np.isfinite(self.adaptive_uncertainty_threshold)
                        and self.root_evaluator.get_uncertainty_prediction(
                            game_state
                        )
                        >= self.adaptive_uncertainty_threshold
                    )
                )

            if should_extend_search:
                extra_simulation_count = (
                    self.adaptive_max_simulation_count - simulation_count
                )
                self.adaptive_extension_count += 1
                self.adaptive_extra_simulation_count += (
                    extra_simulation_count
                )
                run_search_simulations(
                    root,
                    self.evaluator,
                    extra_simulation_count,
                    self.exploration,
                    self.value_weight,
                    self.area_value_weight,
                    self.rollout_depth,
                    game_state.next_color,
                    self.opponent_branch_count,
                    self.first_play_urgency_reduction,
                    self.use_first_play_urgency_prior_mass,
                    self.use_first_play_urgency_at_root_only,
                    self.q_value_normalization_weight,
                    self.use_q_value_normalization_at_root_only,
                    self.descendant_exploration,
                    self.child_q_pseudo_count,
                    self.maximum_extra_simulation_count,
                    self.utility_stdev_prior,
                    self.utility_stdev_prior_weight,
                    self.utility_stdev_scale,
                )

        if not root.children:
            policy = np.zeros(BOARD_AREA + 1, dtype=np.float32)
            policy[BOARD_AREA] = 1
            return (
                BOARD_AREA,
                policy,
                np.zeros(BOARD_AREA + 1, dtype=np.float32),
                np.zeros(BOARD_AREA + 1, dtype=np.float32),
            )

        selectable_root_children = list(root.children.items())

        if self.root_selection_visit_slack >= 0:
            maximum_visit_count = max(
                child.effective_visit_weight
                for child in root.children.values()
            )
            selectable_root_children = [
                move_and_child
                for move_and_child in selectable_root_children
                if move_and_child[1].effective_visit_weight
                >= maximum_visit_count - self.root_selection_visit_slack
            ]
            move, selected_child = max(
                selectable_root_children,
                key=lambda move_and_child: -move_and_child[1].mean_value,
            )
        else:
            move, selected_child = select_root_child_with_lcb(
                selectable_root_children,
                self.root_lcb_stdevs,
                self.root_lcb_minimum_visit_proportion,
            )
        policy = np.zeros(BOARD_AREA + 1, dtype=np.float32)
        q_values = np.zeros(BOARD_AREA + 1, dtype=np.float32)
        q_weights = np.zeros(BOARD_AREA + 1, dtype=np.float32)
        child_visit_sum = sum(
            child.effective_visit_weight
            for child in root.children.values()
        )

        for child_move, child in root.children.items():
            policy[child_move] = (
                child.effective_visit_weight / child_visit_sum
            )
            if child.effective_visit_weight > 0:
                q_values[child_move] = -child.mean_value
                q_weights[child_move] = child.effective_visit_weight

        self.root = selected_child
        return move, policy, q_values, q_weights


class MokaSequentialHalvingSearchSession(MokaSearchSession):
    def __init__(
        self,
        evaluator: MokaEvaluator,
        candidate_count: int,
        exploration: float = SEARCH_PUCT_EXPLORATION,
        descendant_exploration: float | None = (
            SEARCH_DESCENDANT_PUCT_EXPLORATION
        ),
        child_q_pseudo_count: float = SEARCH_CHILD_Q_PSEUDO_COUNT,
        maximum_extra_simulation_count: int = (
            SEARCH_MAXIMUM_EXTRA_EVALUATION_BUDGET_SIMULATION_COUNT
        ),
        use_saved_root_evaluation_budget: bool = False,
        value_weight: float = SEARCH_PUCT_VALUE_WEIGHT,
        area_value_weight: float = SEARCH_AREA_VALUE_WEIGHT,
        rollout_depth: int = SEARCH_ROLLOUT_DEPTH,
        adaptive_max_simulation_count: int = (
            SEARCH_ADAPTIVE_MAX_SIMULATION_COUNT
        ),
        adaptive_visit_margin_ratio: float = (
            SEARCH_ADAPTIVE_VISIT_MARGIN_RATIO
        ),
        adaptive_symmetry_value_spread_threshold: float = (
            SEARCH_ADAPTIVE_SYMMETRY_VALUE_SPREAD_THRESHOLD
        ),
        adaptive_uncertainty_threshold: float = (
            SEARCH_ADAPTIVE_UNCERTAINTY_THRESHOLD
        ),
        opponent_branch_count: int = SEARCH_OPPONENT_BRANCH_COUNT,
        root_evaluator: MokaEvaluator | None = None,
        root_selection_visit_slack: int = -1,
        root_capture_prior_bonus: float = 0,
        root_self_atari_prior_penalty: float = 0,
        first_play_urgency_reduction: float = (
            SEARCH_FIRST_PLAY_URGENCY_REDUCTION
        ),
        use_first_play_urgency_prior_mass: bool = (
            SEARCH_FIRST_PLAY_URGENCY_USE_PRIOR_MASS
        ),
        use_first_play_urgency_at_root_only: bool = (
            SEARCH_FIRST_PLAY_URGENCY_ROOT_ONLY
        ),
        root_branch_count: int = SEARCH_ROOT_BRANCH_COUNT,
        root_policy_temperature: float = SEARCH_ROOT_POLICY_TEMPERATURE,
        root_policy_temperature_end_move_count: int = (
            SEARCH_ROOT_POLICY_TEMPERATURE_END_MOVE_COUNT
        ),
        q_value_normalization_weight: float = (
            SEARCH_Q_VALUE_NORMALIZATION_WEIGHT
        ),
        use_q_value_normalization_at_root_only: bool = (
            SEARCH_Q_VALUE_NORMALIZATION_ROOT_ONLY
        ),
        utility_stdev_prior: float = SEARCH_PUCT_UTILITY_STDEV_PRIOR,
        utility_stdev_prior_weight: float = (
            SEARCH_PUCT_UTILITY_STDEV_PRIOR_WEIGHT
        ),
        utility_stdev_scale: float = SEARCH_PUCT_UTILITY_STDEV_SCALE,
        root_lcb_stdevs: float = SEARCH_ROOT_LCB_STDEVS,
        root_lcb_minimum_visit_proportion: float = (
            SEARCH_ROOT_LCB_MINIMUM_VISIT_PROPORTION
        ),
    ) -> None:
        super().__init__(
            evaluator=evaluator,
            exploration=exploration,
            descendant_exploration=descendant_exploration,
            child_q_pseudo_count=child_q_pseudo_count,
            maximum_extra_simulation_count=maximum_extra_simulation_count,
            use_saved_root_evaluation_budget=(
                use_saved_root_evaluation_budget
            ),
            value_weight=value_weight,
            area_value_weight=area_value_weight,
            rollout_depth=rollout_depth,
            adaptive_max_simulation_count=adaptive_max_simulation_count,
            adaptive_visit_margin_ratio=adaptive_visit_margin_ratio,
            adaptive_symmetry_value_spread_threshold=(
                adaptive_symmetry_value_spread_threshold
            ),
            adaptive_uncertainty_threshold=adaptive_uncertainty_threshold,
            opponent_branch_count=opponent_branch_count,
            root_evaluator=root_evaluator,
            root_selection_visit_slack=root_selection_visit_slack,
            root_capture_prior_bonus=root_capture_prior_bonus,
            root_self_atari_prior_penalty=root_self_atari_prior_penalty,
            first_play_urgency_reduction=first_play_urgency_reduction,
            use_first_play_urgency_prior_mass=(
                use_first_play_urgency_prior_mass
            ),
            use_first_play_urgency_at_root_only=(
                use_first_play_urgency_at_root_only
            ),
            root_branch_count=root_branch_count,
            root_policy_temperature=root_policy_temperature,
            root_policy_temperature_end_move_count=(
                root_policy_temperature_end_move_count
            ),
            q_value_normalization_weight=q_value_normalization_weight,
            use_q_value_normalization_at_root_only=(
                use_q_value_normalization_at_root_only
            ),
            utility_stdev_prior=utility_stdev_prior,
            utility_stdev_prior_weight=utility_stdev_prior_weight,
            utility_stdev_scale=utility_stdev_scale,
            root_lcb_stdevs=root_lcb_stdevs,
            root_lcb_minimum_visit_proportion=(
                root_lcb_minimum_visit_proportion
            ),
        )
        self.candidate_count = candidate_count

    def select_move_with_policy(
        self,
        game_state: GameState,
        simulation_count: int,
    ) -> tuple[int, np.ndarray]:
        root = self.align_root(game_state)
        root_evaluation_count = self.refresh_root_evaluation(
            root,
            game_state,
        )

        if not root.children:
            run_simulation(
                root,
                self.evaluator,
                self.exploration,
                self.value_weight,
                self.area_value_weight,
                self.rollout_depth,
                game_state.next_color,
                self.opponent_branch_count,
                self.first_play_urgency_reduction,
                self.use_first_play_urgency_prior_mass,
                self.use_first_play_urgency_at_root_only,
                self.q_value_normalization_weight,
                self.use_q_value_normalization_at_root_only,
                descendant_exploration=self.descendant_exploration,
                child_q_pseudo_count=self.child_q_pseudo_count,
                utility_stdev_prior=self.utility_stdev_prior,
                utility_stdev_prior_weight=(
                    self.utility_stdev_prior_weight
                ),
                utility_stdev_scale=self.utility_stdev_scale,
            )
            root_evaluation_count += 1

        if not root.children:
            policy = np.zeros(BOARD_AREA + 1, dtype=np.float32)
            policy[BOARD_AREA] = 1
            return BOARD_AREA, policy

        candidates = sorted(
            root.children.items(),
            key=lambda move_and_child: move_and_child[1].prior,
            reverse=True,
        )[: self.candidate_count]
        remaining_simulation_count = max(
            simulation_count - root_evaluation_count,
            0,
        )

        while len(candidates) > 1 and remaining_simulation_count > 0:
            remaining_round_count = max(
                1,
                int(
                    np.ceil(
                        np.log(len(candidates))
                        / np.log(SEARCH_SEQUENTIAL_HALVING_REDUCTION_FACTOR)
                    )
                ),
            )
            simulations_per_candidate = max(
                1,
                remaining_simulation_count
                // (len(candidates) * remaining_round_count),
            )

            for _, child in candidates:
                resolved_descendant_exploration = resolve_search_exploration(
                    self.exploration,
                    self.descendant_exploration,
                    False,
                )
                run_search_simulations(
                    child,
                    self.evaluator,
                    simulations_per_candidate,
                    resolved_descendant_exploration,
                    self.value_weight,
                    self.area_value_weight,
                    self.rollout_depth,
                    game_state.next_color,
                    self.opponent_branch_count,
                    self.first_play_urgency_reduction,
                    self.use_first_play_urgency_prior_mass,
                    self.use_first_play_urgency_at_root_only,
                    self.q_value_normalization_weight,
                    self.use_q_value_normalization_at_root_only,
                    resolved_descendant_exploration,
                    self.child_q_pseudo_count,
                    self.maximum_extra_simulation_count,
                    self.utility_stdev_prior,
                    self.utility_stdev_prior_weight,
                    self.utility_stdev_scale,
                )

            remaining_simulation_count -= (
                simulations_per_candidate * len(candidates)
            )
            survivor_count = max(
                1,
                int(
                    np.ceil(
                        len(candidates)
                        / SEARCH_SEQUENTIAL_HALVING_REDUCTION_FACTOR
                    )
                ),
            )
            candidates = sorted(
                candidates,
                key=lambda move_and_child: (
                    np.log(max(move_and_child[1].prior, SEARCH_POLICY_EPSILON))
                    - self.value_weight * move_and_child[1].mean_value
                ),
                reverse=True,
            )[:survivor_count]

        move, selected_child = candidates[0]
        policy = np.zeros(BOARD_AREA + 1, dtype=np.float32)
        child_visit_sum = sum(
            child.effective_visit_weight
            for child in root.children.values()
        )

        if child_visit_sum > 0:
            for child_move, child in root.children.items():
                policy[child_move] = (
                    child.effective_visit_weight / child_visit_sum
                )
        else:
            policy[move] = 1

        self.root = selected_child
        return move, policy


def select_policy_value_move(
    evaluator: MokaEvaluator,
    game_state: GameState,
    candidate_count: int,
) -> int:
    policy, _ = evaluator.evaluate(game_state)
    legal_moves = get_legal_moves(game_state)
    selectable_moves = (
        [move for move in legal_moves if move != BOARD_AREA]
        if game_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT
        else legal_moves
    )
    ordered_moves = sorted(
        selectable_moves,
        key=lambda move: float(policy[move]),
        reverse=True,
    )[:candidate_count]
    candidate_states = [
        next_state
        for move in ordered_moves
        if (next_state := play_move(game_state, move)) is not None
    ]
    candidate_evaluations = evaluator.evaluate_batch(candidate_states)
    best_candidate_index = max(
        range(len(candidate_states)),
        key=lambda candidate_index: np.log(
            max(
                float(policy[ordered_moves[candidate_index]]),
                SEARCH_POLICY_EPSILON,
            )
        )
        - SEARCH_VALUE_WEIGHT * candidate_evaluations[candidate_index][1],
    )
    return ordered_moves[best_candidate_index]


def select_rollout_move(
    evaluator: MokaEvaluator,
    game_state: GameState,
    candidate_count: int,
    rollout_count: int,
    random_seed: int,
) -> int:
    random_generator = np.random.default_rng(random_seed)
    policy, _ = evaluator.evaluate(game_state)
    legal_moves = get_legal_moves(game_state)
    selectable_moves = (
        [move for move in legal_moves if move != BOARD_AREA]
        if game_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT
        else legal_moves
    )
    candidate_moves = sorted(
        selectable_moves,
        key=lambda move: float(policy[move]),
        reverse=True,
    )[:candidate_count]
    rollout_states: list[GameState] = []
    rollout_candidate_indexes: list[int] = []

    for candidate_index, candidate_move in enumerate(candidate_moves):
        next_state = play_move(game_state, candidate_move)
        if next_state is None:
            continue
        for _ in range(rollout_count):
            rollout_states.append(next_state)
            rollout_candidate_indexes.append(candidate_index)

    candidate_value_sums = np.zeros(len(candidate_moves), dtype=np.float32)
    candidate_visit_counts = np.zeros(len(candidate_moves), dtype=np.int32)
    root_color = game_state.next_color

    while rollout_states:
        evaluations = evaluator.evaluate_batch(rollout_states)
        next_rollout_states: list[GameState] = []
        next_candidate_indexes: list[int] = []

        for rollout_index, rollout_state in enumerate(rollout_states):
            rollout_policy, _ = evaluations[rollout_index]
            legal_rollout_moves = get_legal_moves(rollout_state)
            selectable_rollout_moves = (
                [
                    move
                    for move in legal_rollout_moves
                    if move != BOARD_AREA
                ]
                if rollout_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT
                else legal_rollout_moves
            )
            selectable_probabilities = rollout_policy[
                selectable_rollout_moves
            ].astype(np.float64)
            selectable_probabilities /= np.sum(selectable_probabilities)
            move = int(
                random_generator.choice(
                    selectable_rollout_moves,
                    p=selectable_probabilities,
                )
            )
            next_state = play_move(rollout_state, move)

            if next_state is None:
                next_state = play_move(rollout_state, BOARD_AREA)

            candidate_index = rollout_candidate_indexes[rollout_index]
            if next_state is not None and not is_game_over(next_state):
                next_rollout_states.append(next_state)
                next_candidate_indexes.append(candidate_index)
                continue

            terminal_state = next_state or rollout_state
            did_black_win = get_area_score(terminal_state) > 0
            did_root_player_win = did_black_win == (root_color == 1)
            candidate_value_sums[candidate_index] += (
                1 if did_root_player_win else -1
            )
            candidate_visit_counts[candidate_index] += 1

        rollout_states = next_rollout_states
        rollout_candidate_indexes = next_candidate_indexes

    best_candidate_index = max(
        range(len(candidate_moves)),
        key=lambda candidate_index: (
            candidate_value_sums[candidate_index]
            / max(candidate_visit_counts[candidate_index], 1),
            policy[candidate_moves[candidate_index]],
        ),
    )
    return candidate_moves[best_candidate_index]
