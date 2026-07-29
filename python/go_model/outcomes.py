import argparse
from pathlib import Path

import numpy as np

from go_model.board import (
    GameState,
    get_area_score,
    get_legal_moves,
    is_game_over,
    play_move,
)
from go_model.collect import evaluate_moka_batch, sample_rollout_move
from go_model.config import (
    DEFAULT_RANDOM_SEED,
    GRPO_AREA_MARGIN_SCALE_POINTS,
    GRPO_AREA_MARGIN_WEIGHT,
    GRPO_OPENING_MOVE_COUNTS,
    OUTCOME_BATCH_SIZE,
    OUTCOME_DEFAULT_GAME_COUNT,
    POLICY_MOVE_COUNT,
    PPO_PROBABILITY_EPSILON,
)
from go_model.features import encode_moka_features
from go_model.model import create_moka_network_for_checkpoint
from go_model.teacher import KataGoTeacher


def create_group_opening_states(
    opponent: KataGoTeacher,
    game_count: int,
    group_size: int,
    random_generator: np.random.Generator,
) -> list[GameState]:
    if group_size == 0:
        return [GameState() for _ in range(game_count)]

    games_per_opening = group_size * 2
    opening_count = (game_count + games_per_opening - 1) // games_per_opening
    opening_states = [GameState() for _ in range(opening_count)]
    opening_move_counts = random_generator.choice(
        GRPO_OPENING_MOVE_COUNTS,
        size=opening_count,
    )

    for opening_move_index in range(max(GRPO_OPENING_MOVE_COUNTS)):
        active_opening_indexes = np.flatnonzero(
            opening_move_counts > opening_move_index
        )

        if len(active_opening_indexes) == 0:
            break

        active_states = [
            opening_states[int(opening_index)]
            for opening_index in active_opening_indexes
        ]
        evaluations = opponent.evaluate_batch(active_states)

        for evaluation_index, opening_index in enumerate(active_opening_indexes):
            game_state = opening_states[int(opening_index)]
            move = sample_rollout_move(
                game_state,
                evaluations[evaluation_index][0],
                random_generator,
            )
            next_state = play_move(game_state, move)

            if next_state is not None and not is_game_over(next_state):
                opening_states[int(opening_index)] = next_state

    return [
        opening_states[game_id // games_per_opening].copy()
        for game_id in range(game_count)
    ]


def generate_outcome_dataset(
    checkpoint_path: Path,
    opponent_path: Path,
    game_count: int,
    batch_size: int,
    random_seed: int,
    group_size: int,
    guidance_teacher_checkpoint_path: Path | None,
    katago_source_path: Path,
    use_wide_network: bool,
) -> dict[str, np.ndarray]:
    random_generator = np.random.default_rng(random_seed)
    model = create_moka_network_for_checkpoint(
        str(checkpoint_path),
        not use_wide_network,
        False,
        False,
        False,
        use_wide_network,
    )
    model.load_weights(str(checkpoint_path))
    model.eval()
    opponent = KataGoTeacher(opponent_path)
    if guidance_teacher_checkpoint_path:
        from go_model.strong_teacher import StrongKataGoTeacher

        guidance_teacher = StrongKataGoTeacher(
            guidance_teacher_checkpoint_path,
            katago_source_path,
        )
    else:
        guidance_teacher = opponent
    initial_game_states = create_group_opening_states(
        opponent,
        game_count,
        group_size,
        random_generator,
    )
    features: list[np.ndarray] = []
    game_ids: list[int] = []
    next_colors: list[int] = []
    values: list[float] = []
    actions: list[int] = []
    baselines: list[float] = []
    moka_action_masks: list[bool] = []
    legal_masks: list[np.ndarray] = []
    old_log_probabilities: list[float] = []
    teacher_action_advantages: list[float] = []
    teacher_policies: list[np.ndarray] = []
    teacher_values: list[float] = []
    pending_indexes: dict[int, list[int]] = {}
    previous_position_indexes: dict[int, int] = {}
    active_game_ids = list(range(min(batch_size, game_count)))
    game_states = [
        initial_game_states[game_id].copy() for game_id in active_game_ids
    ]
    next_game_id = len(active_game_ids)
    completed_game_count = 0
    last_reported_game_count = -1

    while game_states:
        moka_evaluations = evaluate_moka_batch(
            model,
            game_states,
            False,
        )
        opponent_evaluations = opponent.evaluate_batch(game_states)
        guidance_evaluations = (
            opponent_evaluations
            if guidance_teacher is opponent
            else guidance_teacher.evaluate_batch(game_states)
        )

        for game_index in range(len(game_states) - 1, -1, -1):
            game_state = game_states[game_index]
            game_id = active_game_ids[game_index]
            position_index = len(features)
            features.append(encode_moka_features(game_state))
            game_ids.append(game_id)
            next_colors.append(game_state.next_color)
            values.append(0)
            legal_mask = np.zeros(POLICY_MOVE_COUNT, dtype=np.bool_)
            legal_mask[get_legal_moves(game_state)] = True
            legal_masks.append(legal_mask)
            pending_indexes.setdefault(game_id, []).append(position_index)
            is_moka_black = game_id % 2 == 0
            is_moka_turn = (game_state.next_color == 1) == is_moka_black
            moka_policy, moka_value = moka_evaluations[game_index]
            teacher_policy, teacher_value = guidance_evaluations[game_index]
            teacher_policies.append(teacher_policy)
            teacher_values.append(teacher_value)
            teacher_action_advantages.append(0)
            previous_position_index = previous_position_indexes.get(game_id)

            if previous_position_index is not None:
                teacher_action_advantages[previous_position_index] = (
                    -teacher_value - teacher_values[previous_position_index]
                )

            previous_position_indexes[game_id] = position_index
            policy = (
                moka_policy
                if is_moka_turn
                else opponent_evaluations[game_index][0]
            )
            move = sample_rollout_move(game_state, policy, random_generator)
            actions.append(move)
            baselines.append(moka_value if is_moka_turn else 0)
            moka_action_masks.append(is_moka_turn)
            old_log_probabilities.append(
                float(np.log(max(moka_policy[move], PPO_PROBABILITY_EPSILON)))
                if is_moka_turn
                else 0
            )
            next_state = play_move(game_state, move)

            if next_state is not None and not is_game_over(next_state):
                game_states[game_index] = next_state
                continue

            terminal_state = next_state or game_state
            area_score = get_area_score(terminal_state)
            winning_color = 1 if area_score > 0 else -1
            for pending_index in pending_indexes.pop(game_id):
                perspective_margin = (
                    area_score * next_colors[pending_index]
                )
                outcome_reward = (
                    1 if next_colors[pending_index] == winning_color else -1
                )
                values[pending_index] = (
                    outcome_reward
                    + GRPO_AREA_MARGIN_WEIGHT
                    * np.tanh(
                        perspective_margin
                        / GRPO_AREA_MARGIN_SCALE_POINTS
                    )
                )

            teacher_action_advantages[position_index] = (
                values[position_index] - teacher_value
            )
            previous_position_indexes.pop(game_id)
            completed_game_count += 1
            if next_game_id < game_count:
                game_states[game_index] = initial_game_states[next_game_id].copy()
                active_game_ids[game_index] = next_game_id
                next_game_id += 1
            else:
                del game_states[game_index]
                del active_game_ids[game_index]

        should_report_progress = (
            completed_game_count > 0
            and completed_game_count != last_reported_game_count
            and completed_game_count % OUTCOME_BATCH_SIZE == 0
        )
        if should_report_progress or not game_states:
            print(f"completed {completed_game_count:,}/{game_count:,} games")
            last_reported_game_count = completed_game_count

    return {
        "actions": np.asarray(actions, dtype=np.int16),
        "baselines": np.asarray(baselines, dtype=np.float16),
        "features": np.asarray(features, dtype=np.float16),
        "game_ids": np.asarray(game_ids, dtype=np.int32),
        "legal_masks": np.asarray(legal_masks, dtype=np.bool_),
        "moka_action_masks": np.asarray(moka_action_masks, dtype=np.bool_),
        "old_log_probabilities": np.asarray(
            old_log_probabilities,
            dtype=np.float16,
        ),
        "teacher_action_advantages": np.asarray(
            teacher_action_advantages,
            dtype=np.float16,
        ),
        "teacher_policies": np.asarray(teacher_policies, dtype=np.float16),
        "values": np.asarray(values, dtype=np.float16),
    }


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument(
        "--opponent",
        type=Path,
        default=Path("../public/models/katago-b6c96.onnx"),
    )
    argument_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/moka-outcomes.npz"),
    )
    argument_parser.add_argument(
        "--games",
        type=int,
        default=OUTCOME_DEFAULT_GAME_COUNT,
    )
    argument_parser.add_argument("--batch-size", type=int, default=OUTCOME_BATCH_SIZE)
    argument_parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    argument_parser.add_argument("--grpo-group-size", type=int, default=0)
    argument_parser.add_argument("--guidance-teacher-checkpoint", type=Path)
    argument_parser.add_argument("--wide", action="store_true")
    argument_parser.add_argument(
        "--katago-source",
        type=Path,
        default=Path("teachers/katago-source"),
    )
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    dataset = generate_outcome_dataset(
        arguments.checkpoint,
        arguments.opponent,
        arguments.games,
        arguments.batch_size,
        arguments.seed,
        arguments.grpo_group_size,
        arguments.guidance_teacher_checkpoint,
        arguments.katago_source,
        arguments.wide,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, **dataset)
    print(
        f"saved {arguments.output} "
        f"positions={len(dataset['features']):,} "
        f"bytes={arguments.output.stat().st_size:,}"
    )


if __name__ == "__main__":
    main()
