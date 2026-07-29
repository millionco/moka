import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from go_model.arena import create_opening_game_state, select_teacher_move
from go_model.board import get_area_score, get_legal_moves, is_game_over, play_move
from go_model.config import (
    ARENA_DEFAULT_GAME_COUNT,
    ARENA_OPENING_PAIR_SIZE,
    BOARD_AREA,
    MID_GAME_MOVE_COUNT,
    MINIMUM_TEACHER_PASS_MOVE_COUNT,
    OPENING_MOVE_COUNT,
    SEARCH_ROOT_POLICY_TEMPERATURE,
    SEARCH_ROOT_SYMMETRY_GEOMETRIC_POLICY_WEIGHT,
)
from go_model.features import encode_moka_features
from go_model.model import MokaNestedNetwork
from go_model.search import MokaEvaluator, MokaSearchSession
from go_model.teacher import KataGoTeacher


def get_phase(move_count: int) -> str:
    if move_count < OPENING_MOVE_COUNT:
        return "opening"
    if move_count < MID_GAME_MOVE_COUNT:
        return "middle"
    return "endgame"


def normalize_policy(policy: np.ndarray, moves: list[int]) -> np.ndarray:
    selected_policy = policy[moves].astype(np.float64)
    policy_sum = float(np.sum(selected_policy))
    return (
        selected_policy / policy_sum
        if policy_sum > 0
        else np.full(len(moves), 1 / len(moves))
    )


def softmax_logits(logits: np.ndarray, moves: list[int]) -> np.ndarray:
    selected_logits = logits[moves].astype(np.float64)
    selected_logits -= np.max(selected_logits)
    probabilities = np.exp(selected_logits)
    return probabilities / np.sum(probabilities)


def diagnose_arena(
    checkpoint_path: Path,
    teacher_path: Path,
    game_count: int,
    opening_offset: int,
    simulation_count: int,
) -> dict:
    model = MokaNestedNetwork()
    model.load_weights(str(checkpoint_path))
    model.eval()
    teacher = KataGoTeacher(teacher_path)
    evaluator = MokaEvaluator(model)
    root_evaluator = MokaEvaluator(
        model,
        use_symmetry_ensemble=True,
        policy_temperature=SEARCH_ROOT_POLICY_TEMPERATURE,
        symmetry_geometric_policy_weight=(
            SEARCH_ROOT_SYMMETRY_GEOMETRIC_POLICY_WEIGHT
        ),
    )
    games: list[dict] = []

    for game_index in range(game_count):
        evaluator.clear_cache()
        root_evaluator.clear_cache()
        game_state = create_opening_game_state(game_index, opening_offset)
        is_moka_black = game_index % ARENA_OPENING_PAIR_SIZE == 0
        search_session = MokaSearchSession(
            evaluator,
            root_evaluator=root_evaluator,
        )
        decisions: list[dict] = []

        while not is_game_over(game_state):
            is_moka_turn = (game_state.next_color == 1) == is_moka_black

            if not is_moka_turn:
                move = select_teacher_move(teacher, game_state)
                next_state = play_move(game_state, move)
                if next_state is None:
                    raise RuntimeError("Teacher selected an illegal move.")
                game_state = next_state
                continue

            teacher_policy, teacher_value_before = teacher.evaluate(game_state)
            legal_moves = get_legal_moves(game_state)
            selectable_moves = (
                [move for move in legal_moves if move != BOARD_AREA]
                if game_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT
                else legal_moves
            )
            if simulation_count > 0:
                searched_root = search_session.align_root(game_state)
                move, searched_policy = search_session.select_move_with_policy(
                    game_state,
                    simulation_count,
                )
                moka_policy = normalize_policy(
                    searched_policy,
                    selectable_moves,
                )
                moka_value = searched_root.mean_value
                symmetry_value_spread = (
                    root_evaluator.get_symmetry_value_spread(game_state)
                )
            else:
                features = encode_moka_features(game_state)
                policy_logits, model_value = model(
                    mx.array(features[None], dtype=mx.float32)
                )
                mx.eval(policy_logits, model_value)
                logits = np.asarray(policy_logits)[0]
                moka_policy = softmax_logits(logits, selectable_moves)
                move = selectable_moves[int(np.argmax(moka_policy))]
                moka_value = float(np.asarray(model_value)[0])
                symmetry_value_spread = 0
            normalized_teacher_policy = normalize_policy(
                teacher_policy,
                selectable_moves,
            )
            selected_index = selectable_moves.index(move)
            teacher_index = int(np.argmax(normalized_teacher_policy))
            teacher_move = selectable_moves[teacher_index]
            next_state = play_move(game_state, move)

            if next_state is None:
                raise RuntimeError("Moka selected an illegal move.")

            _, child_teacher_value = teacher.evaluate(next_state)
            parent_value_after = -child_teacher_value
            teacher_next_state = (
                next_state
                if move == teacher_move
                else play_move(game_state, teacher_move)
            )

            if teacher_next_state is None:
                raise RuntimeError("Teacher selected an illegal move.")

            teacher_move_value_after = (
                parent_value_after
                if move == teacher_move
                else -teacher.evaluate(teacher_next_state)[1]
            )
            positive_teacher_mask = normalized_teacher_policy > 0
            policy_kl = float(
                np.sum(
                    normalized_teacher_policy[positive_teacher_mask]
                    * (
                        np.log(normalized_teacher_policy[positive_teacher_mask])
                        - np.log(
                            np.maximum(
                                moka_policy[positive_teacher_mask],
                                np.finfo(np.float64).tiny,
                            )
                        )
                    )
                )
            )
            decisions.append(
                {
                    "move_count": game_state.move_count,
                    "phase": get_phase(game_state.move_count),
                    "moka_move": move,
                    "teacher_move": teacher_move,
                    "did_match_teacher": move == teacher_move,
                    "policy_kl": policy_kl,
                    "teacher_probability": float(
                        normalized_teacher_policy[selected_index]
                    ),
                    "moka_value": float(moka_value),
                    "symmetry_value_spread": symmetry_value_spread,
                    "teacher_value_before": float(teacher_value_before),
                    "teacher_value_after": float(parent_value_after),
                    "teacher_parent_value_change": float(
                        teacher_value_before - parent_value_after
                    ),
                    "teacher_move_value_after": float(
                        teacher_move_value_after
                    ),
                    "teacher_move_value_regret": float(
                        teacher_move_value_after - parent_value_after
                    ),
                }
            )
            game_state = next_state

        score = float(get_area_score(game_state))
        did_moka_win = (score > 0) == is_moka_black
        games.append(
            {
                "game_index": game_index,
                "is_moka_black": is_moka_black,
                "did_moka_win": did_moka_win,
                "score": score,
                "move_count": game_state.move_count,
                "decisions": decisions,
            }
        )

    phase_summaries = {}

    for phase in ("opening", "middle", "endgame"):
        decisions = [
            decision
            for game in games
            for decision in game["decisions"]
            if decision["phase"] == phase
        ]
        phase_summaries[phase] = {
            "decision_count": len(decisions),
            "teacher_match_rate": float(
                np.mean(
                    [decision["did_match_teacher"] for decision in decisions]
                )
            ),
            "mean_policy_kl": float(
                np.mean([decision["policy_kl"] for decision in decisions])
            ),
            "mean_teacher_move_value_regret": float(
                np.mean(
                    [
                        decision["teacher_move_value_regret"]
                        for decision in decisions
                    ]
                )
            ),
            "maximum_teacher_move_value_regret": float(
                np.max(
                    [
                        decision["teacher_move_value_regret"]
                        for decision in decisions
                    ]
                )
            ),
        }

    return {
        "checkpoint": str(checkpoint_path),
        "game_count": game_count,
        "opening_offset": opening_offset,
        "simulation_count": simulation_count,
        "moka_wins": sum(game["did_moka_win"] for game in games),
        "phase_summaries": phase_summaries,
        "games": games,
    }


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument(
        "--teacher",
        type=Path,
        default=Path("../public/models/katago-b6c96.onnx"),
    )
    argument_parser.add_argument(
        "--games",
        type=int,
        default=ARENA_DEFAULT_GAME_COUNT,
    )
    argument_parser.add_argument("--opening-offset", type=int, default=0)
    argument_parser.add_argument("--simulations", type=int, default=0)
    argument_parser.add_argument("--output", required=True, type=Path)
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    diagnostics = diagnose_arena(
        arguments.checkpoint,
        arguments.teacher,
        arguments.games,
        arguments.opening_offset,
        arguments.simulations,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(diagnostics, indent="\t"))
    print(
        f"saved {arguments.output} "
        f"wins={diagnostics['moka_wins']}/{diagnostics['game_count']}"
    )


if __name__ == "__main__":
    main()
