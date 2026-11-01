import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from go_model.board import GameState, get_area_score, get_legal_moves, is_game_over, play_move
from go_model.config import BOARD_AREA, MAXIMUM_GAME_MOVE_COUNT, MINIMUM_TEACHER_PASS_MOVE_COUNT
from go_model.arena import should_accept_opponent_pass
from go_model.features import encode_moka_features
from go_model.model import create_moka_network_for_checkpoint
from go_model.selfplay_generate import (
    KataGoAnalysisProcess,
    create_argument_parser as create_selfplay_argument_parser,
    save_shard,
)
from go_model.search_generate import coordinate_to_move, move_to_coordinate
from go_model.config import BOARD_SIZE, KATAGO_SIMPLE_AREA_RULES, KOMI_POINTS, POLICY_MOVE_COUNT
from go_model.teacher import KataGoTeacher

ONPOLICY_MAXIMUM_OPENING_MOVE_COUNT = 4
ONPOLICY_EARLY_MOVE_COUNT = 16
ONPOLICY_EARLY_TEMPERATURE = 0.7
ONPOLICY_SAMPLE_PROBABILITY = 0.1


def sample_move(
    random_generator: np.random.Generator,
    logits: np.ndarray,
    selectable_moves: list[int],
    temperature: float,
) -> int:
    selected_logits = logits[selectable_moves] / temperature
    probabilities = np.exp(selected_logits - selected_logits.max())
    probabilities /= probabilities.sum()
    return int(selectable_moves[random_generator.choice(len(selectable_moves), p=probabilities)])


def get_selectable_moves(game_state: GameState) -> list[int]:
    legal_moves = get_legal_moves(game_state)
    if game_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT:
        return [move for move in legal_moves if move != BOARD_AREA]
    return legal_moves


def choose_move(
    random_generator: np.random.Generator,
    logits: np.ndarray,
    game_state: GameState,
    is_moka: bool,
) -> int:
    if is_moka and should_accept_opponent_pass(game_state):
        return BOARD_AREA
    selectable_moves = get_selectable_moves(game_state)
    if is_moka and game_state.move_count < ONPOLICY_EARLY_MOVE_COUNT:
        return sample_move(random_generator, logits, selectable_moves, ONPOLICY_EARLY_TEMPERATURE)
    if random_generator.random() < ONPOLICY_SAMPLE_PROBABILITY:
        return sample_move(random_generator, logits, selectable_moves, 1.0)
    return int(selectable_moves[int(np.argmax(logits[selectable_moves]))])


def play_game(
    random_generator: np.random.Generator,
    moka_model,
    teacher: KataGoTeacher,
    is_moka_black: bool,
) -> GameState:
    game_state = GameState()
    for _ in range(int(random_generator.integers(0, ONPOLICY_MAXIMUM_OPENING_MOVE_COUNT + 1))):
        policy, _ = teacher.evaluate(game_state)
        game_state = play_move(
            game_state,
            sample_move(
                random_generator,
                np.log(np.maximum(policy, 1e-8)),
                get_selectable_moves(game_state),
                1.0,
            ),
        )
    while not is_game_over(game_state) and game_state.move_count < MAXIMUM_GAME_MOVE_COUNT:
        is_moka = (game_state.next_color == 1) == is_moka_black
        if is_moka:
            logits, _ = moka_model(mx.array(encode_moka_features(game_state)[None], dtype=mx.float32))
            logits = np.asarray(logits)[0]
        else:
            policy, _ = teacher.evaluate(game_state)
            logits = np.log(np.maximum(policy, 1e-8))
        next_state = play_move(game_state, choose_move(random_generator, logits, game_state, is_moka))
        if next_state is None:
            break
        game_state = next_state
    return game_state


def create_query(game_id: int, moves: list[int], visit_count: int) -> dict:
    return {
        "id": str(game_id),
        "moves": [
            ["B" if move_index % 2 == 0 else "W", move_to_coordinate(move)]
            for move_index, move in enumerate(moves)
        ],
        "rules": KATAGO_SIMPLE_AREA_RULES,
        "komi": KOMI_POINTS,
        "boardXSize": BOARD_SIZE,
        "boardYSize": BOARD_SIZE,
        "analyzeTurns": list(range(len(moves))),
        "maxVisits": visit_count,
        "includePolicy": True,
        "includeOwnership": True,
    }


def create_records(game_id: int, final_state: GameState, responses: dict[int, dict]) -> list[dict]:
    black_outcome = 1.0 if get_area_score(final_state) > 0 else -1.0
    records = []
    game_state = GameState()
    for turn_number, move in enumerate(final_state.move_history):
        response = responses.get(turn_number)
        if response is not None and response["moveInfos"]:
            search_policy = np.zeros(POLICY_MOVE_COUNT, dtype=np.float32)
            for move_information in response["moveInfos"]:
                search_policy[coordinate_to_move(move_information["move"])] = float(
                    move_information["edgeVisits"]
                )
            raw_policy = np.maximum(np.asarray(response["policy"], dtype=np.float32), 0)
            root_information = response["rootInfo"]
            records.append(
                {
                    "features": encode_moka_features(game_state),
                    "search_policy": search_policy / search_policy.sum(),
                    "raw_policy": raw_policy / max(float(raw_policy.sum()), 1e-8),
                    "value": 2 * float(root_information["winrate"]) - 1,
                    "raw_value": 2 * float(root_information["rawWinrate"]) - 1,
                    "score": float(root_information["scoreLead"]),
                    "ownership": np.asarray(response["ownership"], dtype=np.float32).reshape(
                        BOARD_SIZE, BOARD_SIZE
                    ),
                    "move_number": game_state.move_count,
                    "outcome": black_outcome * game_state.next_color,
                    "game_id": game_id,
                }
            )
        game_state = play_move(game_state, move)
    return records


def run(arguments: argparse.Namespace) -> None:
    random_generator = np.random.default_rng(arguments.seed)
    moka_models = []
    for checkpoint in arguments.moka_checkpoints:
        model = create_moka_network_for_checkpoint(str(checkpoint))
        model.load_weights(str(checkpoint))
        model.eval()
        moka_models.append(model)
    teacher = KataGoTeacher(arguments.opponent)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    position_count = 0
    for shard_index in range(arguments.games // arguments.shard_games):
        first_game_id = arguments.first_game_id + shard_index * arguments.shard_games
        analysis = KataGoAnalysisProcess(
            arguments.katago, arguments.config, arguments.model, arguments.threads
        )
        final_states = {}
        for game_offset in range(arguments.shard_games):
            game_id = first_game_id + game_offset
            final_state = play_game(
                random_generator,
                moka_models[game_id % len(moka_models)],
                teacher,
                game_id % 2 == 0,
            )
            final_states[game_id] = final_state
            analysis.send(create_query(game_id, final_state.move_history, arguments.visits))
        responses: dict[int, dict[int, dict]] = {game_id: {} for game_id in final_states}
        expected_count = sum(len(state.move_history) for state in final_states.values())
        for _ in range(expected_count):
            response = analysis.receive()
            responses[int(response["id"])][int(response["turnNumber"])] = response
        analysis.close()
        records = [
            record
            for game_id, final_state in final_states.items()
            for record in create_records(game_id, final_state, responses[game_id])
        ]
        save_shard(records, arguments.output_directory / f"{arguments.prefix}-{shard_index:04d}.npz")
        position_count += len(records)
        print(
            f"shard={shard_index} positions={position_count:,} "
            f"rate={position_count / (time.time() - start_time):.1f}/s",
            flush=True,
        )


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = create_selfplay_argument_parser()
    argument_parser.add_argument("--moka-checkpoints", type=Path, nargs="+", required=True)
    argument_parser.add_argument("--opponent", type=Path, default=Path("teachers/katago-b6c96.onnx"))
    argument_parser.set_defaults(prefix="onpolicy", output_directory=Path("data/onpolicy"), shard_games=100)
    return argument_parser


def main() -> None:
    run(create_argument_parser().parse_args())


if __name__ == "__main__":
    main()
