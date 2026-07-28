from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from go_model.board import (
    GameState,
    get_area_score,
    get_group,
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
    SEARCH_ADAPTIVE_VISIT_MARGIN_RATIO,
    SEARCH_AREA_VALUE_SCALE_POINTS,
    SEARCH_AREA_VALUE_START_MOVE_COUNT,
    SEARCH_AREA_VALUE_RAMP_MOVE_COUNT,
    SEARCH_AREA_VALUE_WEIGHT,
    SEARCH_FIRST_PLAY_URGENCY_REDUCTION,
    SEARCH_FIRST_PLAY_URGENCY_ROOT_ONLY,
    SEARCH_FIRST_PLAY_URGENCY_USE_PRIOR_MASS,
    SEARCH_OPPONENT_BRANCH_COUNT,
    SEARCH_POLICY_EPSILON,
    SEARCH_PUCT_EXPLORATION,
    SEARCH_PUCT_VALUE_WEIGHT,
    SEARCH_ROLLOUT_DEPTH,
    SEARCH_ROOT_BRANCH_COUNT,
    SEARCH_ROOT_POLICY_TEMPERATURE,
    SEARCH_SEQUENTIAL_HALVING_REDUCTION_FACTOR,
    SEARCH_SIMULATION_BATCH_SIZE,
    SEARCH_VALUE_WEIGHT,
)
from go_model.features import encode_moka_features
from go_model.model import MokaNetwork
from go_model.symmetry import apply_board_symmetry, invert_policy_symmetry


@dataclass
class SearchNode:
    game_state: GameState
    prior: float
    children: dict[int, "SearchNode"] = field(default_factory=dict)
    value_sum: float = 0
    visit_count: int = 0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0


class MokaEvaluator:
    def __init__(
        self,
        model: MokaNetwork,
        use_symmetry_ensemble: bool = False,
        symmetry_rotation_count: int = 0,
        should_flip_symmetry: bool = False,
        use_symmetry_pair: bool = False,
    ) -> None:
        self.model = model
        self.use_symmetry_ensemble = use_symmetry_ensemble
        self.symmetry_rotation_count = symmetry_rotation_count
        self.should_flip_symmetry = should_flip_symmetry
        self.use_symmetry_pair = use_symmetry_pair
        self.cache: dict[
            tuple[bytes, int, int, tuple[int, ...]],
            tuple[np.ndarray, float],
        ] = {}

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

            policy_logits, values = self.model(mx.array(features, dtype=mx.float32))
            mx.eval(policy_logits, values)
            logits = np.asarray(policy_logits)
            value_array = np.asarray(values)
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
                    self.cache[self.get_cache_key(game_state)] = (
                        np.mean(aligned_policies, axis=0),
                        float(
                            np.mean(
                                value_array[symmetry_start:symmetry_end]
                            )
                        ),
                    )
            else:
                for missing_index, game_state in enumerate(missing_game_states):
                    self.cache[self.get_cache_key(game_state)] = (
                        policies[missing_index],
                        float(value_array[missing_index]),
                    )

        return [
            self.cache[self.get_cache_key(game_state)] for game_state in game_states
        ]


def get_terminal_value(game_state: GameState) -> float:
    did_black_win = get_area_score(game_state) > 0
    did_current_player_win = did_black_win == (game_state.next_color == 1)
    return 1 if did_current_player_win else -1


def blend_search_value(
    game_state: GameState,
    network_value: float,
    area_value_weight: float,
) -> float:
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
) -> None:
    legal_moves = get_legal_moves(node.game_state)
    selectable_moves = (
        [move for move in legal_moves if move != BOARD_AREA]
        if node.game_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT
        else legal_moves
    )

    if (
        root_player_color is not None
        and node.game_state.next_color != root_player_color
        and opponent_branch_count > 0
    ):
        selectable_moves = sorted(
            selectable_moves,
            key=lambda move: float(policy[move]),
            reverse=True,
        )[:opponent_branch_count]

    prior_sum = float(np.sum(policy[selectable_moves]))

    for move in selectable_moves:
        next_state = play_move(node.game_state, move)

        if next_state is None:
            continue

        prior = (
            float(policy[move] / prior_sum)
            if prior_sum > 0
            else 1 / len(selectable_moves)
        )
        node.children[move] = SearchNode(game_state=next_state, prior=prior)


def expand_node(node: SearchNode, evaluator: MokaEvaluator) -> float:
    policy, value = evaluator.evaluate(node.game_state)
    expand_node_with_evaluation(node, policy)
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
) -> SearchNode:
    reservation_counts = reservation_counts or {}
    parent_visit_scale = np.sqrt(
        max(node.visit_count + reservation_counts.get(id(node), 0), 1)
    )
    visited_prior_mass = sum(
        child.prior
        for child in node.children.values()
        if (
            child.visit_count + reservation_counts.get(id(child), 0)
        )
        > 0
    )
    effective_first_play_urgency_reduction = (
        first_play_urgency_reduction * np.sqrt(visited_prior_mass)
        if use_first_play_urgency_prior_mass
        else first_play_urgency_reduction
    )

    def get_child_score(child: SearchNode) -> float:
        child_reservation_count = reservation_counts.get(id(child), 0)
        effective_child_visit_count = (
            child.visit_count + child_reservation_count
        )
        parent_value = (
            node.mean_value - effective_first_play_urgency_reduction
            if (
                effective_child_visit_count == 0
                and first_play_urgency_reduction >= 0
            )
            else -child.mean_value
        )
        return (
            value_weight * parent_value
            + exploration
            * child.prior
            * parent_visit_scale
            / (effective_child_visit_count + 1)
        )

    return max(
        node.children.values(),
        key=get_child_score,
    )


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
    is_root: bool = True,
) -> float:
    if is_game_over(node.game_state):
        value = get_terminal_value(node.game_state)
    elif not node.children:
        policy, network_value = evaluator.evaluate(node.game_state)
        expand_node_with_evaluation(
            node,
            policy,
            root_player_color,
            opponent_branch_count,
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
    else:
        child = select_child(
            node,
            exploration=exploration,
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
        )
        value = -run_simulation(
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
            False,
        )

    node.visit_count += 1
    node.value_sum += value
    return value


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
                exploration,
                value_weight,
                resolve_first_play_urgency_reduction(
                    first_play_urgency_reduction,
                    use_first_play_urgency_at_root_only,
                    node is root,
                ),
                use_first_play_urgency_prior_mass,
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

    for search_path in search_paths:
        leaf_node = search_path[-1]
        value = (
            get_terminal_value(leaf_node.game_state)
            if is_game_over(leaf_node.game_state)
            else rollout_value_by_identifier[id(leaf_node)]
        )

        for path_node in reversed(search_path):
            path_node.visit_count += 1
            path_node.value_sum += value
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
) -> None:
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
        )
        remaining_simulation_count -= batch_simulation_count


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
    )

    if not root.children:
        return BOARD_AREA

    return max(
        root.children.items(),
        key=lambda move_and_child: move_and_child[1].visit_count,
    )[0]


class MokaSearchSession:
    def __init__(
        self,
        evaluator: MokaEvaluator,
        exploration: float = SEARCH_PUCT_EXPLORATION,
        value_weight: float = SEARCH_PUCT_VALUE_WEIGHT,
        area_value_weight: float = SEARCH_AREA_VALUE_WEIGHT,
        rollout_depth: int = SEARCH_ROLLOUT_DEPTH,
        adaptive_max_simulation_count: int = (
            SEARCH_ADAPTIVE_MAX_SIMULATION_COUNT
        ),
        adaptive_visit_margin_ratio: float = (
            SEARCH_ADAPTIVE_VISIT_MARGIN_RATIO
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
    ) -> None:
        self.evaluator = evaluator
        self.exploration = exploration
        self.value_weight = value_weight
        self.area_value_weight = area_value_weight
        self.rollout_depth = rollout_depth
        self.adaptive_max_simulation_count = adaptive_max_simulation_count
        self.adaptive_visit_margin_ratio = adaptive_visit_margin_ratio
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
        policy = apply_search_policy_temperature(
            policy,
            self.root_policy_temperature,
        )

        if root.children:
            prior_sum = float(
                np.sum(policy[list(root.children)])
            )

            for move, child in root.children.items():
                child.prior = (
                    float(policy[move] / prior_sum)
                    if prior_sum > 0
                    else 1 / len(root.children)
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
        )
        adjust_root_tactical_priors(
            root,
            self.root_capture_prior_bonus,
            self.root_self_atari_prior_penalty,
        )
        prune_root_children(root, self.root_branch_count)
        root.visit_count += 1
        root.value_sum += blend_search_value(
            game_state,
            network_value,
            self.area_value_weight,
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
        root_evaluation_count = self.refresh_root_evaluation(
            root,
            game_state,
        )
        run_search_simulations(
            root,
            self.evaluator,
            max(simulation_count - root_evaluation_count, 0),
            self.exploration,
            self.value_weight,
            self.area_value_weight,
            self.rollout_depth,
            game_state.next_color,
            self.opponent_branch_count,
            self.first_play_urgency_reduction,
            self.use_first_play_urgency_prior_mass,
            self.use_first_play_urgency_at_root_only,
        )
        ordered_visit_counts = sorted(
            (
                child.visit_count
                for child in root.children.values()
            ),
            reverse=True,
        )

        if (
            self.adaptive_max_simulation_count > simulation_count
            and len(ordered_visit_counts) >= 2
        ):
            leading_visit_count = ordered_visit_counts[0]
            second_visit_count = ordered_visit_counts[1]
            visit_margin_ratio = (
                leading_visit_count - second_visit_count
            ) / max(leading_visit_count + second_visit_count, 1)

            if visit_margin_ratio < self.adaptive_visit_margin_ratio:
                run_search_simulations(
                    root,
                    self.evaluator,
                    self.adaptive_max_simulation_count - simulation_count,
                    self.exploration,
                    self.value_weight,
                    self.area_value_weight,
                    self.rollout_depth,
                    game_state.next_color,
                    self.opponent_branch_count,
                    self.first_play_urgency_reduction,
                    self.use_first_play_urgency_prior_mass,
                    self.use_first_play_urgency_at_root_only,
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
                child.visit_count for child in root.children.values()
            )
            selectable_root_children = [
                move_and_child
                for move_and_child in selectable_root_children
                if move_and_child[1].visit_count
                >= maximum_visit_count - self.root_selection_visit_slack
            ]
            move, selected_child = max(
                selectable_root_children,
                key=lambda move_and_child: -move_and_child[1].mean_value,
            )
        else:
            move, selected_child = max(
                selectable_root_children,
                key=lambda move_and_child: move_and_child[1].visit_count,
            )
        policy = np.zeros(BOARD_AREA + 1, dtype=np.float32)
        q_values = np.zeros(BOARD_AREA + 1, dtype=np.float32)
        q_weights = np.zeros(BOARD_AREA + 1, dtype=np.float32)
        child_visit_sum = sum(
            child.visit_count for child in root.children.values()
        )

        for child_move, child in root.children.items():
            policy[child_move] = child.visit_count / child_visit_sum
            if child.visit_count > 0:
                q_values[child_move] = -child.mean_value
                q_weights[child_move] = child.visit_count

        self.root = selected_child
        return move, policy, q_values, q_weights


class MokaSequentialHalvingSearchSession(MokaSearchSession):
    def __init__(
        self,
        evaluator: MokaEvaluator,
        candidate_count: int,
        exploration: float = SEARCH_PUCT_EXPLORATION,
        value_weight: float = SEARCH_PUCT_VALUE_WEIGHT,
        area_value_weight: float = SEARCH_AREA_VALUE_WEIGHT,
        rollout_depth: int = SEARCH_ROLLOUT_DEPTH,
        adaptive_max_simulation_count: int = (
            SEARCH_ADAPTIVE_MAX_SIMULATION_COUNT
        ),
        adaptive_visit_margin_ratio: float = (
            SEARCH_ADAPTIVE_VISIT_MARGIN_RATIO
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
    ) -> None:
        super().__init__(
            evaluator=evaluator,
            exploration=exploration,
            value_weight=value_weight,
            area_value_weight=area_value_weight,
            rollout_depth=rollout_depth,
            adaptive_max_simulation_count=adaptive_max_simulation_count,
            adaptive_visit_margin_ratio=adaptive_visit_margin_ratio,
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
                run_search_simulations(
                    child,
                    self.evaluator,
                    simulations_per_candidate,
                    self.exploration,
                    self.value_weight,
                    self.area_value_weight,
                    self.rollout_depth,
                    game_state.next_color,
                    self.opponent_branch_count,
                    self.first_play_urgency_reduction,
                    self.use_first_play_urgency_prior_mass,
                    self.use_first_play_urgency_at_root_only,
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
            child.visit_count for child in root.children.values()
        )

        if child_visit_sum > 0:
            for child_move, child in root.children.items():
                policy[child_move] = child.visit_count / child_visit_sum
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
