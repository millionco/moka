import argparse
import time
from pathlib import Path

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
    ARENA_DEFAULT_GAME_COUNT,
    ARENA_OPENING_INDEX_MULTIPLIER,
    ARENA_OPENING_MOVE_COUNT,
    ARENA_OPENING_MOVE_MULTIPLIER,
    ARENA_OPENING_PAIR_SIZE,
    BOARD_AREA,
    MAXIMUM_GAME_MOVE_COUNT,
    MINIMUM_TEACHER_PASS_MOVE_COUNT,
)
from go_model.features import encode_student_features
from go_model.model import StudentNetwork
from go_model.search import MokaEvaluator, select_policy_value_move, select_search_move
from go_model.teacher import KataGoTeacher


def create_opening_game_state(game_index: int, opening_offset: int) -> GameState:
    opening_index = opening_offset + game_index // ARENA_OPENING_PAIR_SIZE
    game_state = GameState()

    for opening_move_index in range(ARENA_OPENING_MOVE_COUNT):
        legal_moves = [
            move for move in get_legal_moves(game_state) if move < BOARD_AREA
        ]
        selected_move_index = (
            opening_index * ARENA_OPENING_INDEX_MULTIPLIER
            + opening_move_index * ARENA_OPENING_MOVE_MULTIPLIER
        ) % len(legal_moves)
        next_state = play_move(game_state, legal_moves[selected_move_index])

        if next_state is None:
            break

        game_state = next_state

    return game_state


def select_student_move(
    model: StudentNetwork,
    evaluator: MokaEvaluator,
    game_state: GameState,
    simulation_count: int,
    lookahead_candidate_count: int,
) -> int:
    if lookahead_candidate_count > 0:
        return select_policy_value_move(
            evaluator,
            game_state,
            lookahead_candidate_count,
        )

    if simulation_count > 0:
        return select_search_move(evaluator, game_state, simulation_count)

    features = encode_student_features(game_state)
    policy_logits, _ = model(mx.array(features[None], dtype=mx.float32))
    mx.eval(policy_logits)
    logits = np.asarray(policy_logits)[0]
    legal_moves = get_legal_moves(game_state)
    selectable_moves = (
        [move for move in legal_moves if move != BOARD_AREA]
        if game_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT
        else legal_moves
    )
    return int(selectable_moves[int(np.argmax(logits[selectable_moves]))])


def select_teacher_move(teacher: KataGoTeacher, game_state: GameState) -> int:
    policy, _ = teacher.evaluate(game_state)
    legal_moves = get_legal_moves(game_state)
    selectable_moves = (
        [move for move in legal_moves if move != BOARD_AREA]
        if game_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT
        else legal_moves
    )
    return int(selectable_moves[int(np.argmax(policy[selectable_moves]))])


def run_arena(
    checkpoint_path: Path,
    teacher_path: Path,
    game_count: int,
    simulation_count: int,
    lookahead_candidate_count: int,
    opening_offset: int,
) -> tuple[int, int, int]:
    model = StudentNetwork()
    model.load_weights(str(checkpoint_path))
    model.eval()
    teacher = KataGoTeacher(teacher_path)
    evaluator = MokaEvaluator(model)
    moka_win_count = 0
    kata_go_win_count = 0
    move_cap_count = 0
    start_time = time.perf_counter()

    for game_index in range(game_count):
        game_state = create_opening_game_state(game_index, opening_offset)
        is_moka_black = game_index % ARENA_OPENING_PAIR_SIZE == 0

        while not is_game_over(game_state):
            is_moka_turn = (game_state.next_color == 1) == is_moka_black
            move = (
                select_student_move(
                    model,
                    evaluator,
                    game_state,
                    simulation_count,
                    lookahead_candidate_count,
                )
                if is_moka_turn
                else select_teacher_move(teacher, game_state)
            )
            next_state = play_move(game_state, move)

            if next_state is None:
                raise RuntimeError("Arena engine selected an illegal move.")

            game_state = next_state

        did_black_win = get_area_score(game_state) > 0
        did_moka_win = did_black_win == is_moka_black
        moka_win_count += int(did_moka_win)
        kata_go_win_count += int(not did_moka_win)
        move_cap_count += int(game_state.move_count >= MAXIMUM_GAME_MOVE_COUNT)

    duration_seconds = time.perf_counter() - start_time
    print(
        f"Moka={moka_win_count} "
        f"KataGo={kata_go_win_count} "
        f"caps={move_cap_count} "
        f"seconds={duration_seconds:.1f}"
    )
    return moka_win_count, kata_go_win_count, move_cap_count


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument(
        "--teacher",
        type=Path,
        default=Path("teachers/katago-b6c96.onnx"),
    )
    argument_parser.add_argument(
        "--games",
        type=int,
        default=ARENA_DEFAULT_GAME_COUNT,
    )
    argument_parser.add_argument("--simulations", type=int, default=0)
    argument_parser.add_argument("--lookahead-candidates", type=int, default=0)
    argument_parser.add_argument("--opening-offset", type=int, default=0)
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    run_arena(
        arguments.checkpoint,
        arguments.teacher,
        arguments.games,
        arguments.simulations,
        arguments.lookahead_candidates,
        arguments.opening_offset,
    )


if __name__ == "__main__":
    main()
