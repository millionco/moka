import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from go_model.board import GameState, get_legal_moves, is_game_over, play_move
from go_model.config import (
    BOARD_AREA,
    DEFAULT_POSITION_COUNT,
    DEFAULT_RANDOM_SEED,
    MINIMUM_TEACHER_PASS_MOVE_COUNT,
    ON_POLICY_GOLDILOCKS_BASE_WEIGHT,
    ON_POLICY_GOLDILOCKS_DIFFICULTY_TARGET,
    ON_POLICY_GOLDILOCKS_EXPONENT_SCALE,
    ON_POLICY_GOLDILOCKS_PEAK_WEIGHT,
    ON_POLICY_GOLDILOCKS_WIDTH,
    ON_POLICY_PROGRESS_INTERVAL,
    ON_POLICY_STUDENT_TEMPERATURE,
    ON_POLICY_TEACHER_INTERVENTION_PROBABILITY,
    POLICY_MOVE_COUNT,
)
from go_model.features import encode_student_features
from go_model.model import StudentNetwork
from go_model.teacher import KataGoTeacher


def calculate_goldilocks_weight(student_teacher_move_probability: float) -> float:
    difficulty = 1 - student_teacher_move_probability
    normalized_distance = (
        difficulty - ON_POLICY_GOLDILOCKS_DIFFICULTY_TARGET
    ) / ON_POLICY_GOLDILOCKS_WIDTH
    return ON_POLICY_GOLDILOCKS_BASE_WEIGHT + ON_POLICY_GOLDILOCKS_PEAK_WEIGHT * (
        np.exp(ON_POLICY_GOLDILOCKS_EXPONENT_SCALE * normalized_distance**2)
    )


def evaluate_student(
    model: StudentNetwork,
    game_state: GameState,
) -> tuple[np.ndarray, float]:
    features = encode_student_features(game_state)
    policy_logits, values = model(mx.array(features[None], dtype=mx.float32))
    mx.eval(policy_logits, values)
    student_logits = np.asarray(policy_logits)[0]
    legal_moves = get_legal_moves(game_state)
    legal_logits = student_logits[legal_moves] / ON_POLICY_STUDENT_TEMPERATURE
    maximum_logit = float(np.max(legal_logits))
    legal_probabilities = np.exp(legal_logits - maximum_logit)
    legal_probabilities /= np.sum(legal_probabilities)
    probabilities = np.zeros(POLICY_MOVE_COUNT, dtype=np.float32)
    probabilities[legal_moves] = legal_probabilities
    return probabilities, float(np.asarray(values)[0])


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


def collect_on_policy_dataset(
    teacher: KataGoTeacher,
    model: StudentNetwork,
    position_count: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    random_generator = np.random.default_rng(random_seed)
    student_features: list[np.ndarray] = []
    teacher_policies: list[np.ndarray] = []
    teacher_values: list[float] = []
    sample_weights: list[float] = []
    game_ids: list[int] = []
    game_state = GameState()
    game_id = 0
    teacher_intervention_count = 0
    teacher_move_agreement_count = 0

    while len(student_features) < position_count:
        teacher_policy, teacher_value = teacher.evaluate(game_state)
        student_policy, _ = evaluate_student(model, game_state)
        teacher_move = int(np.argmax(teacher_policy))
        teacher_move_agreement_count += int(
            teacher_move == int(np.argmax(student_policy))
        )
        student_features.append(encode_student_features(game_state))
        teacher_policies.append(teacher_policy)
        teacher_values.append(teacher_value)
        sample_weights.append(
            calculate_goldilocks_weight(float(student_policy[teacher_move]))
        )
        game_ids.append(game_id)

        should_intervene = (
            random_generator.random()
            < ON_POLICY_TEACHER_INTERVENTION_PROBABILITY
        )
        rollout_policy = teacher_policy if should_intervene else student_policy
        teacher_intervention_count += int(should_intervene)
        move = sample_rollout_move(game_state, rollout_policy, random_generator)
        next_state = play_move(game_state, move)

        if next_state is None:
            next_state = play_move(game_state, BOARD_AREA)

        if next_state is None or is_game_over(next_state):
            game_state = GameState()
            game_id += 1
        else:
            game_state = next_state

        generated_count = len(student_features)

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

    return (
        np.asarray(student_features, dtype=np.float16),
        np.asarray(teacher_policies, dtype=np.float16),
        np.asarray(teacher_values, dtype=np.float16),
        np.asarray(game_ids, dtype=np.int32),
        np.asarray(sample_weights, dtype=np.float16),
    )


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/go-model.safetensors"),
    )
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
        default=Path("data/on-policy.npz"),
    )
    argument_parser.add_argument("--positions", type=int, default=DEFAULT_POSITION_COUNT)
    argument_parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    model = StudentNetwork()
    model.load_weights(str(arguments.checkpoint))
    model.eval()
    if arguments.strong_teacher_checkpoint:
        from go_model.strong_teacher import StrongKataGoTeacher

        teacher = StrongKataGoTeacher(
            arguments.strong_teacher_checkpoint,
            arguments.katago_source,
        )
    else:
        teacher = KataGoTeacher(arguments.teacher)
    features, policies, values, game_ids, sample_weights = (
        collect_on_policy_dataset(
            teacher,
            model,
            arguments.positions,
            arguments.seed,
        )
    )
    np.savez_compressed(
        arguments.output,
        features=features,
        game_ids=game_ids,
        policies=policies,
        sample_weights=sample_weights,
        values=values,
    )
    print(f"saved {arguments.output} ({arguments.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
