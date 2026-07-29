import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from go_model.arena import create_opening_game_state
from go_model.board import GameState, get_legal_moves, is_game_over, play_move
from go_model.config import (
    BOARD_AREA,
    DEFAULT_POSITION_COUNT,
    DEFAULT_RANDOM_SEED,
    DISTILLATION_OPENING_OFFSET,
    MINIMUM_TEACHER_PASS_MOVE_COUNT,
    ON_POLICY_GOLDILOCKS_BASE_WEIGHT,
    ON_POLICY_GOLDILOCKS_DIFFICULTY_TARGET,
    ON_POLICY_GOLDILOCKS_EXPONENT_SCALE,
    ON_POLICY_GOLDILOCKS_PEAK_WEIGHT,
    ON_POLICY_GOLDILOCKS_WIDTH,
    ON_POLICY_BATCH_SIZE,
    ON_POLICY_PROGRESS_INTERVAL,
    ON_POLICY_MOKA_TEMPERATURE,
    ON_POLICY_TEACHER_INTERVENTION_PROBABILITY,
    POLICY_MOVE_COUNT,
)
from go_model.features import (
    encode_moka_context_features,
    encode_moka_features,
)
from go_model.model import (
    MokaNetwork,
    create_moka_network_for_checkpoint,
)
from go_model.teacher import KataGoTeacher


def calculate_goldilocks_weight(moka_teacher_move_probability: float) -> float:
    difficulty = 1 - moka_teacher_move_probability
    normalized_distance = (
        difficulty - ON_POLICY_GOLDILOCKS_DIFFICULTY_TARGET
    ) / ON_POLICY_GOLDILOCKS_WIDTH
    return ON_POLICY_GOLDILOCKS_BASE_WEIGHT + ON_POLICY_GOLDILOCKS_PEAK_WEIGHT * (
        np.exp(ON_POLICY_GOLDILOCKS_EXPONENT_SCALE * normalized_distance**2)
    )


def evaluate_moka_batch(
    model: MokaNetwork,
    game_states: list[GameState],
    use_context_features: bool,
) -> list[tuple[np.ndarray, float]]:
    feature_encoder = (
        encode_moka_context_features
        if use_context_features
        else encode_moka_features
    )
    features = np.asarray(
        [feature_encoder(game_state) for game_state in game_states],
        dtype=np.float32,
    )
    policy_logits, values = model(mx.array(features, dtype=mx.float32))
    mx.eval(policy_logits, values)
    moka_logits = np.asarray(policy_logits)
    moka_values = np.asarray(values)
    evaluations: list[tuple[np.ndarray, float]] = []

    for game_index, game_state in enumerate(game_states):
        legal_moves = get_legal_moves(game_state)
        legal_logits = (
            moka_logits[game_index, legal_moves] / ON_POLICY_MOKA_TEMPERATURE
        )
        maximum_logit = float(np.max(legal_logits))
        legal_probabilities = np.exp(legal_logits - maximum_logit)
        legal_probabilities /= np.sum(legal_probabilities)
        probabilities = np.zeros(POLICY_MOVE_COUNT, dtype=np.float32)
        probabilities[legal_moves] = legal_probabilities
        evaluations.append((probabilities, float(moka_values[game_index])))

    return evaluations


def sample_rollout_move(
    game_state: GameState,
    probabilities: np.ndarray,
    random_generator: np.random.Generator,
) -> int:
    legal_moves = get_legal_moves(game_state)
    selectable_moves = (
        [move for move in legal_moves if move != BOARD_AREA]
        if game_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT
        else legal_moves
    )
    selectable_probabilities = probabilities[selectable_moves].astype(np.float64)
    probability_sum = float(np.sum(selectable_probabilities))

    if probability_sum == 0:
        return int(random_generator.choice(selectable_moves))

    selectable_probabilities /= probability_sum
    return int(random_generator.choice(selectable_moves, p=selectable_probabilities))


def select_greedy_rollout_move(
    game_state: GameState,
    probabilities: np.ndarray,
) -> int:
    legal_moves = get_legal_moves(game_state)
    selectable_moves = (
        [move for move in legal_moves if move != BOARD_AREA]
        if game_state.move_count < MINIMUM_TEACHER_PASS_MOVE_COUNT
        else legal_moves
    )
    return int(
        selectable_moves[
            int(np.argmax(probabilities[selectable_moves]))
        ]
    )


def collect_on_policy_dataset(
    teacher: KataGoTeacher,
    model: MokaNetwork,
    position_count: int,
    random_seed: int,
    batch_size: int,
    opponent: KataGoTeacher | None,
    include_auxiliary_targets: bool,
    use_context_features: bool,
    use_deterministic_rollouts: bool,
    opening_offset: int | None,
) -> dict[str, np.ndarray]:
    random_generator = np.random.default_rng(random_seed)
    moka_features: list[np.ndarray] = []
    teacher_policies: list[np.ndarray] = []
    teacher_values: list[float] = []
    teacher_ownerships: list[np.ndarray] = []
    teacher_attentions: list[np.ndarray] = []
    teacher_scores: list[float] = []
    teacher_short_values: list[float] = []
    sample_weights: list[float] = []
    game_ids: list[int] = []
    game_states = [
        (
            create_opening_game_state(game_id, opening_offset)
            if opening_offset is not None
            else GameState()
        )
        for game_id in range(batch_size)
    ]
    active_game_ids = list(range(batch_size))
    next_game_id = batch_size
    teacher_intervention_count = 0
    teacher_move_agreement_count = 0

    while len(moka_features) < position_count:
        active_count = min(batch_size, position_count - len(moka_features))
        active_game_states = game_states[:active_count]
        teacher_evaluations = (
            teacher.evaluate_batch_with_auxiliary(active_game_states)
            if include_auxiliary_targets
            else teacher.evaluate_batch(active_game_states)
        )
        moka_evaluations = evaluate_moka_batch(
            model,
            active_game_states,
            use_context_features,
        )
        opponent_evaluations = (
            opponent.evaluate_batch(active_game_states) if opponent else None
        )

        for game_index, game_state in enumerate(active_game_states):
            teacher_policy = teacher_evaluations[game_index][0]
            teacher_value = teacher_evaluations[game_index][1]
            moka_policy, _ = moka_evaluations[game_index]
            teacher_move = int(np.argmax(teacher_policy))
            teacher_move_agreement_count += int(
                teacher_move == int(np.argmax(moka_policy))
            )
            moka_features.append(
                encode_moka_context_features(game_state)
                if use_context_features
                else encode_moka_features(game_state)
            )
            teacher_policies.append(teacher_policy)
            teacher_values.append(teacher_value)
            if include_auxiliary_targets:
                teacher_ownerships.append(teacher_evaluations[game_index][2])
                teacher_short_values.append(teacher_evaluations[game_index][3])
                teacher_scores.append(teacher_evaluations[game_index][4])
                teacher_attentions.append(teacher_evaluations[game_index][5])
            sample_weights.append(
                calculate_goldilocks_weight(float(moka_policy[teacher_move]))
            )
            game_ids.append(active_game_ids[game_index])

            is_moka_black = active_game_ids[game_index] % 2 == 0
            is_moka_turn = (game_state.next_color == 1) == is_moka_black
            should_intervene = (
                not use_deterministic_rollouts
                and is_moka_turn
                and random_generator.random()
                < ON_POLICY_TEACHER_INTERVENTION_PROBABILITY
            )
            if opponent_evaluations and not is_moka_turn:
                rollout_policy = opponent_evaluations[game_index][0]
            else:
                rollout_policy = teacher_policy if should_intervene else moka_policy
            teacher_intervention_count += int(should_intervene)
            move = (
                select_greedy_rollout_move(
                    game_state,
                    rollout_policy,
                )
                if use_deterministic_rollouts
                else sample_rollout_move(
                    game_state,
                    rollout_policy,
                    random_generator,
                )
            )
            next_state = play_move(game_state, move)

            if next_state is None:
                next_state = play_move(game_state, BOARD_AREA)

            if next_state is None or is_game_over(next_state):
                game_states[game_index] = (
                    create_opening_game_state(next_game_id, opening_offset)
                    if opening_offset is not None
                    else GameState()
                )
                active_game_ids[game_index] = next_game_id
                next_game_id += 1
            else:
                game_states[game_index] = next_state

        generated_count = len(moka_features)

        if (
            generated_count % ON_POLICY_PROGRESS_INTERVAL == 0
            or generated_count == position_count
        ):
            agreement = teacher_move_agreement_count / generated_count
            intervention_rate = teacher_intervention_count / generated_count
            mean_weight = float(np.mean(sample_weights))
            print(
                f"collected {generated_count:,}/{position_count:,} "
                f"agreement={agreement:.1%} "
                f"interventions={intervention_rate:.1%} "
                f"weight={mean_weight:.3f}"
            )

    dataset = {
        "features": np.asarray(moka_features, dtype=np.float16),
        "game_ids": np.asarray(game_ids, dtype=np.int32),
        "policies": np.asarray(teacher_policies, dtype=np.float16),
        "sample_weights": np.asarray(sample_weights, dtype=np.float16),
        "values": np.asarray(teacher_values, dtype=np.float16),
    }
    if include_auxiliary_targets:
        dataset["ownerships"] = np.asarray(teacher_ownerships, dtype=np.float16)
        dataset["attentions"] = np.asarray(teacher_attentions, dtype=np.float16)
        dataset["scores"] = np.asarray(teacher_scores, dtype=np.float16)
        dataset["short_values"] = np.asarray(
            teacher_short_values,
            dtype=np.float16,
        )
    return dataset


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/moka-model.safetensors"),
    )
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
        default=Path("data/on-policy.npz"),
    )
    argument_parser.add_argument("--positions", type=int, default=DEFAULT_POSITION_COUNT)
    argument_parser.add_argument("--batch-size", type=int, default=ON_POLICY_BATCH_SIZE)
    argument_parser.add_argument("--nested", action="store_true")
    argument_parser.add_argument("--spatial", action="store_true")
    argument_parser.add_argument("--recurrent", action="store_true")
    argument_parser.add_argument("--context", action="store_true")
    argument_parser.add_argument("--wide", action="store_true")
    argument_parser.add_argument("--opponent", type=Path)
    argument_parser.add_argument("--auxiliary-targets", action="store_true")
    argument_parser.add_argument("--deterministic-rollouts", action="store_true")
    argument_parser.add_argument(
        "--diverse-openings",
        action="store_true",
    )
    argument_parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    model = create_moka_network_for_checkpoint(
        str(arguments.checkpoint),
        arguments.nested,
        arguments.spatial,
        arguments.recurrent,
        arguments.context,
        arguments.wide,
    )
    model.load_weights(
        str(arguments.checkpoint),
        strict=not arguments.spatial,
    )
    model.eval()
    if arguments.strong_teacher_checkpoint:
        from go_model.strong_teacher import StrongKataGoTeacher

        teacher = StrongKataGoTeacher(
            arguments.strong_teacher_checkpoint,
            arguments.katago_source,
        )
    else:
        teacher = KataGoTeacher(arguments.teacher)
    opponent = KataGoTeacher(arguments.opponent) if arguments.opponent else None
    dataset = collect_on_policy_dataset(
        teacher,
        model,
        arguments.positions,
        arguments.seed,
        arguments.batch_size,
        opponent,
        arguments.auxiliary_targets,
        arguments.context,
        arguments.deterministic_rollouts,
        DISTILLATION_OPENING_OFFSET if arguments.diverse_openings else None,
    )
    np.savez_compressed(
        arguments.output,
        **dataset,
    )
    print(f"saved {arguments.output} ({arguments.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
