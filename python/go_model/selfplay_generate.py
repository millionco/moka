import argparse
import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from go_model.board import GameState, get_legal_moves, play_move
from go_model.config import (
    BOARD_AREA,
    BOARD_SIZE,
    KATAGO_SIMPLE_AREA_RULES,
    KOMI_POINTS,
    MAXIMUM_GAME_MOVE_COUNT,
    MINIMUM_TEACHER_PASS_MOVE_COUNT,
    POLICY_MOVE_COUNT,
)
from go_model.features import encode_moka_features
from go_model.search_generate import coordinate_to_move, move_to_coordinate

SELFPLAY_MAXIMUM_OPENING_SAMPLE_MOVE_COUNT = 10
SELFPLAY_EXPLORATION_MOVE_COUNT = 24
SELFPLAY_EXPLORATION_TEMPERATURE = 0.6
SELFPLAY_DEVIATION_PROBABILITY = 0.06
SELFPLAY_RESIGNATION_PROBABILITY = 0.7
SELFPLAY_RESIGNATION_VALUE = 0.94
SELFPLAY_RESIGNATION_PLY_COUNT = 6
SELFPLAY_RESPONSE_TIMEOUT_SECONDS = 180


@dataclass
class SelfplayGame:
    game_id: int
    opening_sample_move_count: int
    allows_resignation: bool
    game_state: GameState = field(default_factory=GameState)
    records: list[dict] = field(default_factory=list)
    decisive_ply_count: int = 0
    black_value: float = 0.0


class KataGoAnalysisProcess:
    def __init__(
        self,
        executable_path: Path,
        config_path: Path,
        model_path: Path,
        thread_count: int,
    ) -> None:
        override_config = ",".join(
            [
                "reportAnalysisWinratesAs=SIDETOMOVE",
                f"numAnalysisThreads={thread_count}",
                "numSearchThreadsPerAnalysisThread=1",
                f"nnMaxBatchSize={thread_count}",
                "nnCacheSizePowerOfTwo=17",
                f"maxBoardXSizeForNNBuffer={BOARD_SIZE}",
                f"maxBoardYSizeForNNBuffer={BOARD_SIZE}",
                "logToStderr=false",
            ]
        )
        self.process = subprocess.Popen(
            [
                str(executable_path),
                "analysis",
                "-config",
                str(config_path),
                "-model",
                str(model_path),
                "-override-config",
                override_config,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.error_lines: list[str] = []
        self.output_lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self.capture_errors, daemon=True).start()
        threading.Thread(target=self.capture_output, daemon=True).start()

    def capture_output(self) -> None:
        for line in self.process.stdout:
            self.output_lines.put(line)
        self.output_lines.put("")

    def capture_errors(self) -> None:
        for line in self.process.stderr:
            self.error_lines.append(line.rstrip())

    def send(self, query: dict) -> None:
        self.process.stdin.write(json.dumps(query, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def receive(self) -> dict:
        while True:
            try:
                line = self.output_lines.get(timeout=SELFPLAY_RESPONSE_TIMEOUT_SECONDS)
            except queue.Empty:
                self.process.kill()
                raise RuntimeError("KataGo analysis stopped responding.")
            if not line:
                raise RuntimeError(
                    "KataGo analysis stopped unexpectedly.\n"
                    + "\n".join(self.error_lines[-20:])
                )
            response = json.loads(line)
            if "error" in response:
                raise RuntimeError(response["error"])
            if "warning" in response or response.get("isDuringSearch"):
                continue
            return response

    def close(self) -> None:
        self.process.stdin.close()
        self.process.wait(timeout=60)


def create_query(game: SelfplayGame, visit_count: int) -> dict:
    return {
        "id": str(game.game_id),
        "moves": [
            ["B" if move_index % 2 == 0 else "W", move_to_coordinate(move)]
            for move_index, move in enumerate(game.game_state.move_history)
        ],
        "rules": KATAGO_SIMPLE_AREA_RULES,
        "komi": KOMI_POINTS,
        "boardXSize": BOARD_SIZE,
        "boardYSize": BOARD_SIZE,
        "maxVisits": visit_count,
        "includePolicy": True,
        "includeOwnership": True,
    }


def sample_from_distribution(
    random_generator: np.random.Generator,
    weights: np.ndarray,
    temperature: float,
) -> int:
    positive_moves = np.flatnonzero(weights > 0)
    tempered = np.power(weights[positive_moves], 1 / temperature)
    return int(
        random_generator.choice(positive_moves, p=tempered / tempered.sum())
    )


def get_selectable_mask(game_state: GameState) -> np.ndarray:
    mask = np.zeros(POLICY_MOVE_COUNT, dtype=bool)
    mask[get_legal_moves(game_state)] = True
    if game_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT:
        mask[BOARD_AREA] = False
    return mask


def process_response(
    game: SelfplayGame,
    response: dict,
    random_generator: np.random.Generator,
) -> int | None:
    game_state = game.game_state
    raw_policy = np.maximum(
        np.asarray(response["policy"], dtype=np.float32), 0
    )
    search_policy = np.zeros(POLICY_MOVE_COUNT, dtype=np.float32)
    for move_information in response["moveInfos"]:
        search_policy[coordinate_to_move(move_information["move"])] = float(
            move_information["edgeVisits"]
        )
    if search_policy.sum() <= 0:
        return None
    search_policy /= search_policy.sum()
    root_information = response["rootInfo"]
    value = 2 * float(root_information["winrate"]) - 1
    game.black_value = value * game_state.next_color
    game.records.append(
        {
            "features": encode_moka_features(game_state),
            "search_policy": search_policy,
            "raw_policy": raw_policy / max(float(raw_policy.sum()), 1e-8),
            "value": value,
            "raw_value": 2 * float(root_information["rawWinrate"]) - 1,
            "score": float(root_information["scoreLead"]),
            "ownership": np.asarray(
                response["ownership"], dtype=np.float32
            ).reshape(BOARD_SIZE, BOARD_SIZE),
            "color": game_state.next_color,
            "move_number": game_state.move_count,
        }
    )

    if abs(value) >= SELFPLAY_RESIGNATION_VALUE:
        game.decisive_ply_count += 1
    else:
        game.decisive_ply_count = 0
    if (
        game.allows_resignation
        and game.decisive_ply_count >= SELFPLAY_RESIGNATION_PLY_COUNT
    ):
        return None

    selectable_mask = get_selectable_mask(game_state)
    if game_state.move_count < game.opening_sample_move_count or (
        random_generator.random() < SELFPLAY_DEVIATION_PROBABILITY
    ):
        candidate_weights = np.where(selectable_mask, raw_policy, 0)
        if candidate_weights.sum() > 0:
            return sample_from_distribution(
                random_generator, candidate_weights, 1.0
            )

    candidate_weights = np.where(selectable_mask, search_policy, 0)
    if candidate_weights.sum() <= 0:
        return BOARD_AREA if selectable_mask[BOARD_AREA] else None
    if game_state.move_count < SELFPLAY_EXPLORATION_MOVE_COUNT:
        return sample_from_distribution(
            random_generator,
            candidate_weights,
            SELFPLAY_EXPLORATION_TEMPERATURE,
        )
    return int(np.argmax(candidate_weights))


def finish_game(game: SelfplayGame) -> list[dict]:
    black_outcome = 1.0 if game.black_value > 0 else -1.0
    for record in game.records:
        record["outcome"] = black_outcome * record["color"]
        record["game_id"] = game.game_id
    return game.records


def save_shard(records: list[dict], output_path: Path) -> None:
    np.savez_compressed(
        output_path,
        features=np.stack([record["features"] for record in records]).astype(
            np.float16
        ),
        search_policies=np.stack(
            [record["search_policy"] for record in records]
        ).astype(np.float16),
        raw_policies=np.stack(
            [record["raw_policy"] for record in records]
        ).astype(np.float16),
        values=np.array([record["value"] for record in records], np.float16),
        raw_values=np.array(
            [record["raw_value"] for record in records], np.float16
        ),
        outcomes=np.array([record["outcome"] for record in records], np.int8),
        scores=np.array([record["score"] for record in records], np.float16),
        ownerships=np.stack(
            [record["ownership"] for record in records]
        ).astype(np.float16),
        move_numbers=np.array(
            [record["move_number"] for record in records], np.int16
        ),
        game_ids=np.array([record["game_id"] for record in records], np.int32),
    )


def run_selfplay(arguments: argparse.Namespace) -> None:
    random_generator = np.random.default_rng(arguments.seed)
    analysis = KataGoAnalysisProcess(
        arguments.katago,
        arguments.config,
        arguments.model,
        arguments.threads,
    )
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    next_game_id = arguments.first_game_id
    final_game_id = arguments.first_game_id + arguments.games
    active_games: dict[int, SelfplayGame] = {}
    shard_records: list[dict] = []
    shard_game_count = 0
    shard_index = 0
    finished_game_count = 0
    position_count = 0
    start_time = time.time()

    def start_game() -> None:
        nonlocal next_game_id
        game = SelfplayGame(
            game_id=next_game_id,
            opening_sample_move_count=int(
                random_generator.integers(
                    0, SELFPLAY_MAXIMUM_OPENING_SAMPLE_MOVE_COUNT + 1
                )
            ),
            allows_resignation=bool(
                random_generator.random() < SELFPLAY_RESIGNATION_PROBABILITY
            ),
        )
        next_game_id += 1
        active_games[game.game_id] = game
        analysis.send(create_query(game, arguments.visits))

    while len(active_games) < arguments.concurrent_games and (
        next_game_id < final_game_id
    ):
        start_game()

    while active_games:
        response = analysis.receive()
        game = active_games[int(response["id"])]
        move = process_response(game, response, random_generator)
        next_state = None if move is None else play_move(game.game_state, move)
        if next_state is not None:
            game.game_state = next_state
        if (
            next_state is None
            or next_state.consecutive_pass_count >= 2
            or next_state.move_count >= MAXIMUM_GAME_MOVE_COUNT
        ):
            del active_games[game.game_id]
            records = finish_game(game)
            shard_records.extend(records)
            position_count += len(records)
            finished_game_count += 1
            shard_game_count += 1
            if shard_game_count >= arguments.shard_games:
                save_shard(
                    shard_records,
                    arguments.output_directory
                    / f"{arguments.prefix}-{shard_index:04d}.npz",
                )
                shard_index += 1
                shard_records = []
                shard_game_count = 0
            if finished_game_count % 16 == 0:
                elapsed = time.time() - start_time
                print(
                    f"games={finished_game_count} positions={position_count:,} "
                    f"rate={position_count / elapsed:.1f}/s",
                    flush=True,
                )
            if next_game_id < final_game_id:
                start_game()
        else:
            analysis.send(create_query(game, arguments.visits))

    if shard_records:
        save_shard(
            shard_records,
            arguments.output_directory
            / f"{arguments.prefix}-{shard_index:04d}.npz",
        )
    analysis.close()


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--games", type=int, default=1_000)
    argument_parser.add_argument("--first-game-id", type=int, default=0)
    argument_parser.add_argument("--visits", type=int, default=64)
    argument_parser.add_argument("--threads", type=int, default=32)
    argument_parser.add_argument("--concurrent-games", type=int, default=48)
    argument_parser.add_argument("--shard-games", type=int, default=200)
    argument_parser.add_argument("--seed", type=int, default=0)
    argument_parser.add_argument("--prefix", default="b18-selfplay")
    argument_parser.add_argument(
        "--output-directory", type=Path, default=Path("data/selfplay")
    )
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
        default=Path(
            "teachers/kata9x9-native/kata9x9-b18c384nbt-20231025.bin.gz"
        ),
    )
    return argument_parser


def main() -> None:
    run_selfplay(create_argument_parser().parse_args())


if __name__ == "__main__":
    main()
