import argparse
import time
from pathlib import Path

import numpy as np

from go_model.arena import (
    create_opening_game_state,
    should_accept_opponent_pass,
    should_resign_selected_pass,
)
from go_model.board import get_area_score, get_legal_moves, is_game_over, play_move
from go_model.config import (
    BOARD_AREA,
    BOARD_SIZE,
    KATAGO_SIMPLE_AREA_RULES,
    KOMI_POINTS,
    MAXIMUM_GAME_MOVE_COUNT,
    SEARCH_RESIGNATION_AREA_MARGIN_POINTS,
)
from go_model.headtohead import SearchPlayer
from go_model.search_generate import coordinate_to_move, move_to_coordinate
from go_model.selfplay_generate import KataGoAnalysisProcess

KATAGO_RESTART_GAME_COUNT = 50


def create_query(query_id: str, moves: list[int], visit_count: int) -> dict:
    return {
        "id": query_id,
        "moves": [
            ["B" if move_index % 2 == 0 else "W", move_to_coordinate(move)]
            for move_index, move in enumerate(moves)
        ],
        "rules": KATAGO_SIMPLE_AREA_RULES,
        "komi": KOMI_POINTS,
        "boardXSize": BOARD_SIZE,
        "boardYSize": BOARD_SIZE,
        "analyzeTurns": [len(moves)],
        "maxVisits": visit_count,
        "includePolicy": True,
    }


def select_katago_move(analysis: KataGoAnalysisProcess, game_state, visit_count: int) -> int:
    analysis.send(create_query("move", list(game_state.move_history), visit_count))
    response = analysis.receive()
    legal_moves = set(get_legal_moves(game_state))
    ranked_moves = [
        coordinate_to_move(move_information["move"])
        for move_information in sorted(response["moveInfos"], key=lambda item: item["order"])
    ]
    policy = np.asarray(response["policy"], dtype=np.float32)
    ranked_moves += [int(move) for move in np.argsort(-policy) if policy[move] >= 0]
    for move in ranked_moves:
        if move in legal_moves:
            return move
    return BOARD_AREA


def run_match(arguments: argparse.Namespace) -> None:
    moka = SearchPlayer(arguments.checkpoint)
    wins = black_wins = moka_resigns = caps = 0
    start_time = time.perf_counter()
    analysis = None

    for game_index in range(arguments.games):
        if game_index % KATAGO_RESTART_GAME_COUNT == 0:
            if analysis is not None:
                analysis.close()
            analysis = KataGoAnalysisProcess(
                arguments.katago, arguments.config, arguments.model, arguments.threads
            )
        moka_is_black = game_index % 2 == 0
        game_state = create_opening_game_state(game_index, arguments.opening_offset)
        session = moka.create_session()
        moka_resigned = False

        while not is_game_over(game_state) and game_state.move_count < MAXIMUM_GAME_MOVE_COUNT:
            if (game_state.next_color == 1) == moka_is_black:
                if should_accept_opponent_pass(game_state):
                    move = BOARD_AREA
                else:
                    move = session.select_move(game_state, arguments.simulations)
                if should_resign_selected_pass(
                    game_state, move, SEARCH_RESIGNATION_AREA_MARGIN_POINTS
                ):
                    moka_resigned = True
                    break
            else:
                move = select_katago_move(analysis, game_state, arguments.visits)
            next_state = play_move(game_state, move)
            if next_state is None:
                raise RuntimeError("Match selected an illegal move.")
            game_state = next_state

        caps += int(not is_game_over(game_state) and not moka_resigned)
        if moka_resigned:
            moka_won = False
            moka_resigns += 1
        else:
            moka_won = (get_area_score(game_state) > 0) == moka_is_black
        wins += int(moka_won)
        black_wins += int(moka_won and moka_is_black)
        print(
            f"game={game_index + 1} moka_won={int(moka_won)} moka={wins} "
            f"katago={game_index + 1 - wins} moves={game_state.move_count}",
            flush=True,
        )

    analysis.close()
    print(
        f"Moka={wins} black={black_wins} white={wins - black_wins} "
        f"KataGo={arguments.games - wins} caps={caps} MokaResigns={moka_resigns} "
        f"visits={arguments.visits} model={arguments.model.name} "
        f"seconds={time.perf_counter() - start_time:.1f}"
    )


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--checkpoint", type=Path, required=True)
    argument_parser.add_argument("--games", type=int, default=100)
    argument_parser.add_argument("--simulations", type=int, default=64)
    argument_parser.add_argument("--visits", type=int, default=1)
    argument_parser.add_argument("--threads", type=int, default=4)
    argument_parser.add_argument("--opening-offset", type=int, required=True)
    argument_parser.add_argument(
        "--katago", type=Path, default=Path("/opt/homebrew/bin/katago")
    )
    argument_parser.add_argument(
        "--config",
        type=Path,
        default=Path("teachers/katago-source/cpp/configs/analysis_example.cfg"),
    )
    argument_parser.add_argument(
        "--model",
        type=Path,
        default=Path("teachers/kata9x9-native/kata9x9-b18c384nbt-20231025.bin.gz"),
    )
    return argument_parser


def main() -> None:
    run_match(create_argument_parser().parse_args())


if __name__ == "__main__":
    main()
