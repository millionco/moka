import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from mlx.utils import tree_flatten
import mlx.core as mx

from go_model.arena import run_arena
from go_model.config import (
    ARENA_DEFAULT_GAME_COUNT,
    RESEARCH_BOOTSTRAP_TRIAL_COUNT,
    RESEARCH_CONFIDENCE,
    RESEARCH_DEFAULT_SEED_COUNT,
    RESEARCH_INCUMBENT_DEV_WINS,
    RESEARCH_MAX_CHECKPOINT_BYTES,
    RESEARCH_TIMEOUT_SECONDS,
)
from go_model.model import MokaGlobalPoolNetwork, MokaNestedNetwork

ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
AUTORESEARCH_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENT_PATH = AUTORESEARCH_DIRECTORY / "experiment.py"
LEDGER_PATH = AUTORESEARCH_DIRECTORY / "rigor-ledger.jsonl"
RUN_DIRECTORY = AUTORESEARCH_DIRECTORY / "runs"
TEACHER_PATH = ROOT_DIRECTORY.parent / "public/models/katago-b6c96.onnx"


def calculate_experiment_hash() -> str:
    return hashlib.sha256(EXPERIMENT_PATH.read_bytes()).hexdigest()[:12]


def load_ledger() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    return [
        json.loads(line)
        for line in LEDGER_PATH.read_text().splitlines()
        if line.strip()
    ]


def get_best_entry(ledger: list[dict]) -> dict | None:
    kept_entries = [entry for entry in ledger if entry["status"] == "keep"]
    return max(
        kept_entries,
        key=lambda entry: entry["mean_wins"],
        default=None,
    )


def calculate_improvement_probability(
    best_samples: list[int],
    candidate_samples: list[int],
) -> float:
    random_generator = random.Random(1_234)
    improvement_count = 0

    for _ in range(RESEARCH_BOOTSTRAP_TRIAL_COUNT):
        best_mean = statistics.mean(
            random_generator.choice(best_samples)
            for _ in best_samples
        )
        candidate_mean = statistics.mean(
            random_generator.choice(candidate_samples)
            for _ in candidate_samples
        )
        improvement_count += int(candidate_mean > best_mean)

    return improvement_count / RESEARCH_BOOTSTRAP_TRIAL_COUNT


def validate_checkpoint(checkpoint_path: Path) -> tuple[int, bool]:
    checkpoint_size = checkpoint_path.stat().st_size

    if checkpoint_size > RESEARCH_MAX_CHECKPOINT_BYTES:
        raise ValueError(
            f"Checkpoint is {checkpoint_size:,} bytes, exceeding "
            f"{RESEARCH_MAX_CHECKPOINT_BYTES:,}."
        )

    checkpoint_weights = mx.load(str(checkpoint_path))
    use_global_pool_network = any(
        name.startswith("global_") for name in checkpoint_weights
    )
    model = (
        MokaGlobalPoolNetwork()
        if use_global_pool_network
        else MokaNestedNetwork()
    )
    model.load_weights(str(checkpoint_path))
    parameter_count = sum(
        parameter.size
        for _, parameter in tree_flatten(model.parameters())
    )
    return parameter_count, use_global_pool_network


def run_experiment_once(
    experiment_hash: str,
    seed: int,
) -> tuple[int, int, int, int]:
    checkpoint_path = (
        RUN_DIRECTORY / f"{experiment_hash}-seed-{seed}.safetensors"
    )
    subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_PATH),
            "--seed",
            str(seed),
            "--output",
            str(checkpoint_path),
        ],
        cwd=ROOT_DIRECTORY,
        check=True,
        timeout=RESEARCH_TIMEOUT_SECONDS,
    )
    parameter_count, use_global_pool_network = validate_checkpoint(
        checkpoint_path
    )
    moka_wins, kata_go_wins, move_caps = run_arena(
        checkpoint_path,
        TEACHER_PATH,
        ARENA_DEFAULT_GAME_COUNT,
        0,
        0,
        0,
        True,
        False,
        False,
        False,
        False,
        4,
        0,
        use_global_pool_network=use_global_pool_network,
    )
    return moka_wins, kata_go_wins, move_caps, parameter_count


def record_result(
    experiment_hash: str,
    description: str,
    samples: list[int],
    move_caps: list[int],
    parameter_count: int,
    status: str,
    improvement_probability: float,
) -> dict:
    entry = {
        "created_at": datetime.now(UTC).isoformat(),
        "description": description,
        "hash": experiment_hash,
        "improvement_probability": round(improvement_probability, 4),
        "mean_wins": statistics.mean(samples) if samples else 0,
        "move_caps": move_caps,
        "parameter_count": parameter_count,
        "samples": samples,
        "status": status,
    }
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a") as ledger_file:
        ledger_file.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def score_experiment(
    description: str,
    seed_count: int,
    confidence: float,
) -> None:
    ledger = load_ledger()
    experiment_hash = calculate_experiment_hash()
    previous_entry = next(
        (
            entry
            for entry in ledger
            if entry["hash"] == experiment_hash
        ),
        None,
    )

    if previous_entry:
        print(
            f"already scored {experiment_hash}: "
            f"{previous_entry['status']} "
            f"mean={previous_entry['mean_wins']:.2f}"
        )
        return

    best_entry = get_best_entry(ledger)
    samples: list[int] = []
    move_caps: list[int] = []
    parameter_count = 0

    try:
        for seed_index in range(seed_count):
            seed = seed_index + 1
            moka_wins, _, caps, parameter_count = run_experiment_once(
                experiment_hash,
                seed,
            )
            samples.append(moka_wins)
            move_caps.append(caps)
            print(
                f"seed {seed}/{seed_count}: "
                f"wins={moka_wins} caps={caps}"
            )

            if (
                best_entry
                and seed_index == 0
                and moka_wins < best_entry["mean_wins"]
            ):
                record_result(
                    experiment_hash,
                    description,
                    samples,
                    move_caps,
                    parameter_count,
                    "discard",
                    0,
                )
                print("DISCARD: clear first-seed regression")
                return
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as error:
        record_result(
            experiment_hash,
            description,
            samples,
            move_caps,
            parameter_count,
            "crash",
            0,
        )
        print(f"CRASH: {error}")
        return

    if best_entry is None:
        entry = record_result(
            experiment_hash,
            description,
            samples,
            move_caps,
            parameter_count,
            "keep",
            1,
        )
        print(
            f"BASELINE: mean={entry['mean_wins']:.2f} "
            f"expected={RESEARCH_INCUMBENT_DEV_WINS}"
        )
        return

    improvement_probability = calculate_improvement_probability(
        best_entry["samples"],
        samples,
    )
    status = (
        "keep"
        if improvement_probability >= confidence
        else "discard"
    )
    entry = record_result(
        experiment_hash,
        description,
        samples,
        move_caps,
        parameter_count,
        status,
        improvement_probability,
    )
    print(
        f"{status.upper()}: mean={entry['mean_wins']:.2f} "
        f"best={best_entry['mean_wins']:.2f} "
        f"P(better)={improvement_probability:.3f}"
    )


def print_best() -> None:
    best_entry = get_best_entry(load_ledger())
    if best_entry is None:
        print("no baseline")
        return
    print(json.dumps(best_entry, indent=2, sort_keys=True))


def print_log() -> None:
    for entry in load_ledger():
        print(json.dumps(entry, sort_keys=True))


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    subparsers = argument_parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("description")
    run_parser.add_argument(
        "--seeds",
        type=int,
        default=RESEARCH_DEFAULT_SEED_COUNT,
    )
    run_parser.add_argument(
        "--confidence",
        type=float,
        default=RESEARCH_CONFIDENCE,
    )
    subparsers.add_parser("best")
    subparsers.add_parser("log")
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    if arguments.command == "run":
        score_experiment(
            arguments.description,
            arguments.seeds,
            arguments.confidence,
        )
    elif arguments.command == "best":
        print_best()
    else:
        print_log()


if __name__ == "__main__":
    main()
