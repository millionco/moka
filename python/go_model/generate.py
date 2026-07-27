import argparse
from pathlib import Path

import numpy as np

from go_model.arena import create_opening_game_state
from go_model.board import (
    GameState,
    get_area_score,
    get_legal_moves,
    is_game_over,
    play_move,
)
from go_model.collect import select_greedy_rollout_move
from go_model.config import (
    BOARD_AREA,
    DEFAULT_POSITION_COUNT,
    DEFAULT_RANDOM_SEED,
    ELITE_DEFAULT_GAME_COUNT,
    ELITE_OPENING_OFFSET,
    ELITE_PROGRESS_INTERVAL_GAMES,
    MINIMUM_TEACHER_PASS_MOVE_COUNT,
    ON_POLICY_PROGRESS_INTERVAL,
    STRONG_TEACHER_BATCH_SIZE,
)
from go_model.features import encode_moka_context_features, encode_moka_features
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
    use_context_features: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    random_generator = np.random.default_rng(random_seed)
    moka_features: list[np.ndarray] = []
    teacher_policies: list[np.ndarray] = []
    teacher_values: list[float] = []
    game_ids: list[int] = []
    game_state = GameState()
    game_id = 0

    while len(moka_features) < position_count:
        policy_probabilities, perspective_value = teacher.evaluate(game_state)
        moka_features.append(
            encode_moka_context_features(game_state)
            if use_context_features
            else encode_moka_features(game_state)
        )
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

        generated_count = len(moka_features)

        if generated_count % 1_000 == 0 or generated_count == position_count:
            print(f"generated {generated_count:,}/{position_count:,} positions")

    return (
        np.asarray(moka_features, dtype=np.float16),
        np.asarray(teacher_policies, dtype=np.float16),
        np.asarray(teacher_values, dtype=np.float16),
        np.asarray(game_ids, dtype=np.int32),
    )


def generate_batched_dataset(
    teacher,
    position_count: int,
    random_seed: int,
    batch_size: int,
    use_context_features: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    random_generator = np.random.default_rng(random_seed)
    moka_features: list[np.ndarray] = []
    teacher_policies: list[np.ndarray] = []
    teacher_values: list[float] = []
    game_ids: list[int] = []
    game_states = [GameState() for _ in range(batch_size)]
    active_game_ids = list(range(batch_size))
    next_game_id = batch_size

    while len(moka_features) < position_count:
        active_count = min(batch_size, position_count - len(moka_features))
        active_game_states = game_states[:active_count]
        evaluations = teacher.evaluate_batch(active_game_states)

        for game_index, game_state in enumerate(active_game_states):
            policy_probabilities, perspective_value = evaluations[game_index]
            moka_features.append(
                encode_moka_context_features(game_state)
                if use_context_features
                else encode_moka_features(game_state)
            )
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

        generated_count = len(moka_features)

        if (
            generated_count % ON_POLICY_PROGRESS_INTERVAL < batch_size
            or generated_count == position_count
        ):
            print(f"generated {generated_count:,}/{position_count:,} positions")

    return (
        np.asarray(moka_features, dtype=np.float16),
        np.asarray(teacher_policies, dtype=np.float16),
        np.asarray(teacher_values, dtype=np.float16),
        np.asarray(game_ids, dtype=np.int32),
    )


def generate_elite_dataset(
    teacher,
    opponent: KataGoTeacher,
    game_count: int,
    batch_size: int,
    opening_offset: int,
    use_context_features: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    accepted_features: list[np.ndarray] = []
    accepted_policies: list[np.ndarray] = []
    accepted_values: list[float] = []
    accepted_game_ids: list[int] = []
    active_game_count = min(batch_size, game_count)
    game_states = [
        create_opening_game_state(game_id, opening_offset)
        for game_id in range(active_game_count)
    ]
    active_game_ids = list(range(active_game_count))
    trajectory_features: list[list[np.ndarray]] = [
        [] for _ in range(active_game_count)
    ]
    trajectory_policies: list[list[np.ndarray]] = [
        [] for _ in range(active_game_count)
    ]
    trajectory_values: list[list[float]] = [[] for _ in range(active_game_count)]
    next_game_id = active_game_count
    completed_game_count = 0
    accepted_black_game_count = 0
    accepted_white_game_count = 0
    last_reported_game_count = 0

    while game_states:
        teacher_evaluations = teacher.evaluate_batch(game_states)
        opponent_evaluations = opponent.evaluate_batch(game_states)

        for game_index in reversed(range(len(game_states))):
            game_state = game_states[game_index]
            game_id = active_game_ids[game_index]
            teacher_policy, teacher_value = teacher_evaluations[game_index]
            opponent_policy, _ = opponent_evaluations[game_index]
            trajectory_features[game_index].append(
                encode_moka_context_features(game_state)
                if use_context_features
                else encode_moka_features(game_state)
            )
            trajectory_policies[game_index].append(teacher_policy)
            trajectory_values[game_index].append(teacher_value)
            is_teacher_black = game_id % 2 == 0
            is_teacher_turn = (game_state.next_color == 1) == is_teacher_black
            controller_policy = teacher_policy if is_teacher_turn else opponent_policy
            move = select_greedy_rollout_move(game_state, controller_policy)
            next_state = play_move(game_state, move)

            if next_state is None:
                next_state = play_move(game_state, BOARD_AREA)

            if next_state is not None and not is_game_over(next_state):
                game_states[game_index] = next_state
                continue

            terminal_state = next_state if next_state is not None else game_state
            did_black_win = get_area_score(terminal_state) > 0
            did_teacher_win = did_black_win == is_teacher_black

            if did_teacher_win:
                accepted_features.extend(trajectory_features[game_index])
                accepted_policies.extend(trajectory_policies[game_index])
                accepted_values.extend(trajectory_values[game_index])
                accepted_game_ids.extend(
                    [game_id] * len(trajectory_features[game_index])
                )
                accepted_black_game_count += int(is_teacher_black)
                accepted_white_game_count += int(not is_teacher_black)

            completed_game_count += 1

            if next_game_id < game_count:
                game_states[game_index] = create_opening_game_state(
                    next_game_id,
                    opening_offset,
                )
                active_game_ids[game_index] = next_game_id
                trajectory_features[game_index] = []
                trajectory_policies[game_index] = []
                trajectory_values[game_index] = []
                next_game_id += 1
            else:
                game_states.pop(game_index)
                active_game_ids.pop(game_index)
                trajectory_features.pop(game_index)
                trajectory_policies.pop(game_index)
                trajectory_values.pop(game_index)

        if (
            completed_game_count - last_reported_game_count
            >= ELITE_PROGRESS_INTERVAL_GAMES
            or completed_game_count == game_count
        ):
            print(
                f"completed {completed_game_count:,}/{game_count:,} games "
                f"accepted={accepted_black_game_count + accepted_white_game_count:,} "
                f"black={accepted_black_game_count:,} "
                f"white={accepted_white_game_count:,}"
            )
            last_reported_game_count = completed_game_count

    feature_array = np.asarray(accepted_features, dtype=np.float16)
    policy_array = np.asarray(accepted_policies, dtype=np.float16)
    value_array = np.asarray(accepted_values, dtype=np.float16)
    game_id_array = np.asarray(accepted_game_ids, dtype=np.int32)
    accepted_black_game_ids = sorted(
        game_id for game_id in set(accepted_game_ids) if game_id % 2 == 0
    )
    accepted_white_game_ids = sorted(
        game_id for game_id in set(accepted_game_ids) if game_id % 2 == 1
    )
    balanced_game_count_per_color = min(
        len(accepted_black_game_ids),
        len(accepted_white_game_ids),
    )
    balanced_game_ids = (
        accepted_black_game_ids[:balanced_game_count_per_color]
        + accepted_white_game_ids[:balanced_game_count_per_color]
    )
    balanced_mask = np.isin(game_id_array, balanced_game_ids)
    print(
        f"balanced {balanced_game_count_per_color * 2:,} elite games "
        f"into {int(np.count_nonzero(balanced_mask)):,} positions"
    )
    return (
        feature_array[balanced_mask],
        policy_array[balanced_mask],
        value_array[balanced_mask],
        game_id_array[balanced_mask],
    )


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--teacher",
        type=Path,
        default=Path("../public/models/katago-b6c96.onnx"),
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
    argument_parser.add_argument("--context", action="store_true")
    argument_parser.add_argument(
        "--elite-games",
        type=int,
        nargs="?",
        const=ELITE_DEFAULT_GAME_COUNT,
        default=0,
    )
    argument_parser.add_argument(
        "--elite-opponent",
        type=Path,
        default=Path("../public/models/katago-b6c96.onnx"),
    )
    argument_parser.add_argument(
        "--opening-offset",
        type=int,
        default=ELITE_OPENING_OFFSET,
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
        if arguments.elite_games:
            opponent = KataGoTeacher(arguments.elite_opponent)
            features, policies, values, game_ids = generate_elite_dataset(
                teacher,
                opponent,
                arguments.elite_games,
                arguments.teacher_batch_size,
                arguments.opening_offset,
                arguments.context,
            )
        else:
            features, policies, values, game_ids = generate_batched_dataset(
                teacher,
                arguments.positions,
                arguments.seed,
                arguments.teacher_batch_size,
                arguments.context,
            )
    else:
        teacher = KataGoTeacher(arguments.teacher)
        features, policies, values, game_ids = generate_dataset(
            teacher,
            arguments.positions,
            arguments.seed,
            arguments.context,
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
