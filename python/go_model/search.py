from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from go_model.board import (
    GameState,
    get_area_score,
    get_legal_moves,
    is_game_over,
    play_move,
)
from go_model.config import (
    BOARD_AREA,
    MINIMUM_TEACHER_PASS_MOVE_COUNT,
    SEARCH_PUCT_EXPLORATION,
    SEARCH_VALUE_WEIGHT,
)
from go_model.features import encode_student_features
from go_model.model import StudentNetwork


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
    def __init__(self, model: StudentNetwork) -> None:
        self.model = model
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
        cache_key = self.get_cache_key(game_state)
        cached_evaluation = self.cache.get(cache_key)

        if cached_evaluation:
            return cached_evaluation

        features = encode_student_features(game_state)
        policy_logits, values = self.model(mx.array(features[None], dtype=mx.float32))
        mx.eval(policy_logits, values)
        logits = np.asarray(policy_logits)[0]
        maximum_logit = float(np.max(logits))
        policy = np.exp(logits - maximum_logit)
        policy /= np.sum(policy)
        evaluation = (policy, float(np.asarray(values)[0]))
        self.cache[cache_key] = evaluation
        return evaluation

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
            features = np.stack(
                [encode_student_features(game_state) for game_state in missing_game_states]
            )
            policy_logits, values = self.model(mx.array(features, dtype=mx.float32))
            mx.eval(policy_logits, values)
            logits = np.asarray(policy_logits)
            value_array = np.asarray(values)
            maximum_logits = np.max(logits, axis=1, keepdims=True)
            policies = np.exp(logits - maximum_logits)
            policies /= np.sum(policies, axis=1, keepdims=True)

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


def expand_node(node: SearchNode, evaluator: MokaEvaluator) -> float:
    policy, value = evaluator.evaluate(node.game_state)
    legal_moves = get_legal_moves(node.game_state)
    selectable_moves = (
        [move for move in legal_moves if move != BOARD_AREA]
        if node.game_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT
        else legal_moves
    )
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

    return value


def select_child(node: SearchNode) -> SearchNode:
    parent_visit_scale = np.sqrt(max(node.visit_count, 1))
    return max(
        node.children.values(),
        key=lambda child: -child.mean_value
        + SEARCH_PUCT_EXPLORATION
        * child.prior
        * parent_visit_scale
        / (child.visit_count + 1),
    )


def run_simulation(node: SearchNode, evaluator: MokaEvaluator) -> float:
    if is_game_over(node.game_state):
        value = get_terminal_value(node.game_state)
    elif not node.children:
        value = expand_node(node, evaluator)
    else:
        child = select_child(node)
        value = -run_simulation(child, evaluator)

    node.visit_count += 1
    node.value_sum += value
    return value


def select_search_move(
    evaluator: MokaEvaluator,
    game_state: GameState,
    simulation_count: int,
) -> int:
    root = SearchNode(game_state=game_state, prior=1)

    for _ in range(simulation_count):
        run_simulation(root, evaluator)

    if not root.children:
        return BOARD_AREA

    return max(
        root.children.items(),
        key=lambda move_and_child: move_and_child[1].visit_count,
    )[0]


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
            max(float(policy[ordered_moves[candidate_index]]), 1e-8)
        )
        - SEARCH_VALUE_WEIGHT * candidate_evaluations[candidate_index][1],
    )
    return ordered_moves[best_candidate_index]
