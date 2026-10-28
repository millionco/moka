import argparse
import time
from pathlib import Path

from go_model.arena import (
    create_opening_game_state,
    should_accept_opponent_pass,
    should_resign_selected_pass,
)
from go_model.board import get_area_score, is_game_over, play_move
from go_model.config import (
    BOARD_AREA,
    SEARCH_RESIGNATION_AREA_MARGIN_POINTS,
    SEARCH_ROOT_POLICY_TEMPERATURE,
    SEARCH_ROOT_SYMMETRY_GEOMETRIC_POLICY_WEIGHT,
)
from go_model.model import create_moka_network_for_checkpoint
from go_model.search import MokaEvaluator, MokaSearchSession


class SearchPlayer:
    def __init__(self, checkpoint_path: Path) -> None:
        self.model = create_moka_network_for_checkpoint(str(checkpoint_path))
        self.model.load_weights(str(checkpoint_path))
        self.model.eval()
        self.evaluator = MokaEvaluator(self.model)
        self.root_evaluator = MokaEvaluator(
            self.model,
            use_symmetry_ensemble=True,
            policy_temperature=SEARCH_ROOT_POLICY_TEMPERATURE,
            symmetry_geometric_policy_weight=(
                SEARCH_ROOT_SYMMETRY_GEOMETRIC_POLICY_WEIGHT
            ),
        )

    def create_session(self) -> MokaSearchSession:
        return MokaSearchSession(
            evaluator=self.evaluator,
            root_evaluator=self.root_evaluator,
        )


def run_match(arguments: argparse.Namespace) -> None:
    players = [SearchPlayer(arguments.first), SearchPlayer(arguments.second)]
    wins = [0, 0]
    black_wins = [0, 0]
    resignations = [0, 0]
    start_time = time.perf_counter()

    for game_index in range(arguments.games):
        first_is_black = game_index % 2 == 0
        game_state = create_opening_game_state(game_index, arguments.opening_offset)
        sessions = [player.create_session() for player in players]
        resigned_player = None

        while not is_game_over(game_state):
            is_black_turn = game_state.next_color == 1
            player_index = 0 if is_black_turn == first_is_black else 1
            if should_accept_opponent_pass(game_state):
                move = BOARD_AREA
            else:
                move = sessions[player_index].select_move(
                    game_state, arguments.simulations
                )
            if should_resign_selected_pass(
                game_state, move, SEARCH_RESIGNATION_AREA_MARGIN_POINTS
            ):
                resigned_player = player_index
                break
            next_state = play_move(game_state, move)
            if next_state is None:
                raise RuntimeError("Head-to-head engine selected an illegal move.")
            game_state = next_state

        if resigned_player is not None:
            winner = 1 - resigned_player
            resignations[resigned_player] += 1
        else:
            black_won = get_area_score(game_state) > 0
            winner = 0 if black_won == first_is_black else 1
        wins[winner] += 1
        black_wins[winner] += int((winner == 0) == first_is_black)
        print(
            f"game={game_index + 1} winner={'first' if winner == 0 else 'second'} "
            f"first={wins[0]} second={wins[1]} moves={game_state.move_count}",
            flush=True,
        )

    print(
        f"first={wins[0]} (black {black_wins[0]} / white {wins[0] - black_wins[0]}) "
        f"second={wins[1]} (black {black_wins[1]} / white {wins[1] - black_wins[1]}) "
        f"resigns={resignations} seconds={time.perf_counter() - start_time:.1f}"
    )


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--first", type=Path, required=True)
    argument_parser.add_argument("--second", type=Path, required=True)
    argument_parser.add_argument("--games", type=int, default=100)
    argument_parser.add_argument("--simulations", type=int, default=64)
    argument_parser.add_argument("--opening-offset", type=int, required=True)
    return argument_parser


def main() -> None:
    run_match(create_argument_parser().parse_args())


if __name__ == "__main__":
    main()
