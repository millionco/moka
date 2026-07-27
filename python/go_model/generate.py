import argparse
from pathlib import Path

import numpy as np

from go_model.board import GameState, get_legal_moves, is_game_over, play_move
from go_model.config import (
    BOARD_AREA,
    DEFAULT_POSITION_COUNT,
    DEFAULT_RANDOM_SEED,
    MINIMUM_TEACHER_PASS_MOVE_COUNT,
    ON_POLICY_PROGRESS_INTERVAL,
    STRONG_TEACHER_BATCH_SIZE,
)
from go_model.features import encode_student_features
from go_model.teacher import KataGoTeacher


def sample_move(
    game_state: GameState,
    policy_probabilities: np.ndarray,
    random_generator: np.random.Generator,
) -> int:
    legal_moves = get_legal_moves(game_state)
    selectable_moves = (
        [move for move in legal_moves if move != BOARD_AREA]
        if game_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT
        else legal_moves
    )
    temperature = max(0.35, 1.15 - game_state.move_count / 100)
    tempered_probabilities = np.power(
        policy_probabilities[selectable_moves],
        1 / temperature,
    )
    probability_sum = float(np.sum(tempered_probabilities))

    if probability_sum == 0:
        return int(random_generator.choice(selectable_moves))

    tempered_probabilities /= probability_sum
    return int(random_generator.choice(selectable_moves, p=tempered_probabilities))


def generate_dataset(
    teacher: KataGoTeacher,
    position_count: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    random_generator = np.random.default_rng(random_seed)
    student_features: list[np.ndarray] = []
    teacher_policies: list[np.ndarray] = []
    teacher_values: list[float] = []
    game_ids: list[int] = []
    game_state = GameState()
    game_id = 0

    while len(student_features) < position_count:
        policy_probabilities, perspective_value = teacher.evaluate(game_state)
        student_features.append(encode_student_features(game_state))
        teacher_policies.append(policy_probabilities)
        teacher_values.append(perspective_value)
        game_ids.append(game_id)
        move = sample_move(game_state, policy_probabilities, random_generator)
        next_state = play_move(game_state, move)

        if next_state is None:
            next_state = play_move(game_state, BOARD_AREA)

        if next_state is None or is_game_over(next_state):
            game_state = GameState()
            game_id += 1
        else:
            game_state = next_state

        generated_count = len(student_features)

        if generated_count % 1_000 == 0 or generated_count == position_count:
            print(f"generated {generated_count:,}/{position_count:,} positions")

    return (
        np.asarray(student_features, dtype=np.float16),
        np.asarray(teacher_policies, dtype=np.float16),
        np.asarray(teacher_values, dtype=np.float16),
        np.asarray(game_ids, dtype=np.int32),
    )


def generate_batched_dataset(
    teacher,
    position_count: int,
    random_seed: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    random_generator = np.random.default_rng(random_seed)
    student_features: list[np.ndarray] = []
    teacher_policies: list[np.ndarray] = []
    teacher_values: list[float] = []
    game_ids: list[int] = []
    game_states = [GameState() for _ in range(batch_size)]
    active_game_ids = list(range(batch_size))
    next_game_id = batch_size

    while len(student_features) < position_count:
        active_count = min(batch_size, position_count - len(student_features))
        active_game_states = game_states[:active_count]
        evaluations = teacher.evaluate_batch(active_game_states)

        for game_index, game_state in enumerate(active_game_states):
            policy_probabilities, perspective_value = evaluations[game_index]
            student_features.append(encode_student_features(game_state))
            teacher_policies.append(policy_probabilities)
            teacher_values.append(perspective_value)
            game_ids.append(active_game_ids[game_index])
            move = sample_move(
                game_state,
                policy_probabilities,
                random_generator,
            )
            next_state = play_move(game_state, move)

            if next_state is None or is_game_over(next_state):
                game_states[game_index] = GameState()
                active_game_ids[game_index] = next_game_id
                next_game_id += 1
            else:
                game_states[game_index] = next_state

        generated_count = len(student_features)

        if (
            generated_count % ON_POLICY_PROGRESS_INTERVAL < batch_size
            or generated_count == position_count
        ):
            print(f"generated {generated_count:,}/{position_count:,} positions")

    return (
        np.asarray(student_features, dtype=np.float16),
        np.asarray(teacher_policies, dtype=np.float16),
        np.asarray(teacher_values, dtype=np.float16),
        np.asarray(game_ids, dtype=np.int32),
    )


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--teacher",
        type=Path,
        default=Path("teachers/katago-b6c96.onnx"),
    )
    argument_parser.add_argument("--strong-teacher-checkpoint", type=Path)
    argument_parser.add_argument(
        "--katago-source",
        type=Path,
        default=Path("teachers/katago-source"),
    )
    argument_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/katago-distillation.npz"),
    )
    argument_parser.add_argument("--positions", type=int, default=DEFAULT_POSITION_COUNT)
    argument_parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    argument_parser.add_argument(
        "--teacher-batch-size",
        type=int,
        default=STRONG_TEACHER_BATCH_SIZE,
    )
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.strong_teacher_checkpoint:
        from go_model.strong_teacher import StrongKataGoTeacher

        teacher = StrongKataGoTeacher(
            arguments.strong_teacher_checkpoint,
            arguments.katago_source,
        )
        features, policies, values, game_ids = generate_batched_dataset(
            teacher,
            arguments.positions,
            arguments.seed,
            arguments.teacher_batch_size,
        )
    else:
        teacher = KataGoTeacher(arguments.teacher)
        features, policies, values, game_ids = generate_dataset(
            teacher,
            arguments.positions,
            arguments.seed,
        )
    np.savez_compressed(
        arguments.output,
        features=features,
        game_ids=game_ids,
        policies=policies,
        values=values,
    )
    print(f"saved {arguments.output} ({arguments.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
