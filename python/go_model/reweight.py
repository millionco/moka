import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from go_model.config import (
    COUNTERFACTUAL_REGRET_BASE_WEIGHT,
    COUNTERFACTUAL_REGRET_MAXIMUM,
    COUNTERFACTUAL_REGRET_MAXIMUM_WEIGHT,
    COUNTERFACTUAL_REGRET_SIGNIFICANCE_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    DISAGREEMENT_CORRECT_SAMPLE_WEIGHT,
    DISAGREEMENT_ERROR_SAMPLE_WEIGHT,
    POLICY_SURPRISE_MAXIMUM_WEIGHT,
    POLICY_SURPRISE_UNIFORM_WEIGHT,
    REWEIGHT_DISAGREEMENT_MODE,
    REWEIGHT_COUNTERFACTUAL_MODE,
    REWEIGHT_COUNTERFACTUAL_CRITICAL_MODE,
    REWEIGHT_MODES,
    REWEIGHT_ROLLOUT_REGRET_MODE,
)
from go_model.model import MokaNetwork, create_moka_network


def calculate_policy_surprise_weights(
    model: MokaNetwork,
    features: np.ndarray,
    policies: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    policy_surprises: list[np.ndarray] = []

    for batch_start in range(0, len(features), batch_size):
        batch_end = batch_start + batch_size
        feature_batch = mx.array(features[batch_start:batch_end], dtype=mx.float32)
        policy_logits, _ = model(feature_batch)
        log_probabilities = policy_logits - mx.logsumexp(
            policy_logits,
            axis=1,
            keepdims=True,
        )
        moka_log_probabilities = np.asarray(log_probabilities)
        teacher_probabilities = policies[batch_start:batch_end]
        teacher_log_probabilities = np.log(np.maximum(teacher_probabilities, 1e-8))
        policy_surprises.append(
            np.sum(
                teacher_probabilities
                * (teacher_log_probabilities - moka_log_probabilities),
                axis=1,
            )
        )

    surprises = np.maximum(np.concatenate(policy_surprises), 0)
    proportional_weights = surprises / max(float(np.mean(surprises)), 1e-8)
    weights = (
        POLICY_SURPRISE_UNIFORM_WEIGHT
        + (1 - POLICY_SURPRISE_UNIFORM_WEIGHT) * proportional_weights
    )
    return np.minimum(weights, POLICY_SURPRISE_MAXIMUM_WEIGHT).astype(np.float32)


def calculate_disagreement_weights(
    model: MokaNetwork,
    features: np.ndarray,
    policies: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    did_disagree_batches: list[np.ndarray] = []

    for batch_start in range(0, len(features), batch_size):
        batch_end = batch_start + batch_size
        policy_logits, _ = model(
            mx.array(features[batch_start:batch_end], dtype=mx.float32)
        )
        mx.eval(policy_logits)
        predicted_moves = np.asarray(mx.argmax(policy_logits, axis=1))
        teacher_moves = np.argmax(policies[batch_start:batch_end], axis=1)
        did_disagree_batches.append(predicted_moves != teacher_moves)

    did_disagree = np.concatenate(did_disagree_batches)
    weights = np.where(
        did_disagree,
        DISAGREEMENT_ERROR_SAMPLE_WEIGHT,
        DISAGREEMENT_CORRECT_SAMPLE_WEIGHT,
    ).astype(np.float32)
    weights /= np.mean(weights)
    return weights


def calculate_counterfactual_regret_weights(
    policies: np.ndarray,
    q_values: np.ndarray,
    counterfactual_values: np.ndarray,
) -> np.ndarray:
    teacher_moves = np.argmax(policies, axis=1)
    teacher_values = q_values[
        np.arange(len(teacher_moves)),
        teacher_moves,
    ]
    regrets = np.where(
        np.isfinite(counterfactual_values),
        teacher_values - counterfactual_values,
        0,
    )
    normalized_regrets = np.clip(
        regrets / COUNTERFACTUAL_REGRET_MAXIMUM,
        0,
        1,
    )
    return (
        COUNTERFACTUAL_REGRET_BASE_WEIGHT
        + (
            COUNTERFACTUAL_REGRET_MAXIMUM_WEIGHT
            - COUNTERFACTUAL_REGRET_BASE_WEIGHT
        )
        * normalized_regrets
    ).astype(np.float32)


def calculate_counterfactual_critical_weights(
    policies: np.ndarray,
    q_values: np.ndarray,
    counterfactual_values: np.ndarray,
) -> np.ndarray:
    teacher_moves = np.argmax(policies, axis=1)
    teacher_values = q_values[
        np.arange(len(teacher_moves)),
        teacher_moves,
    ]
    is_available = np.isfinite(counterfactual_values)
    regrets = teacher_values - counterfactual_values
    is_significant = (
        is_available
        & (regrets >= COUNTERFACTUAL_REGRET_SIGNIFICANCE_THRESHOLD)
    )
    is_winner_flipping = (
        is_significant
        & (teacher_values > 0)
        & (counterfactual_values < 0)
    )
    return np.where(
        is_winner_flipping,
        COUNTERFACTUAL_REGRET_MAXIMUM_WEIGHT,
        is_significant.astype(np.float32),
    ).astype(np.float32)


def calculate_rollout_regret_weights(
    policies: np.ndarray,
    q_values: np.ndarray,
    q_weights: np.ndarray,
    rollout_moves: np.ndarray,
) -> np.ndarray:
    row_indexes = np.arange(len(policies))
    teacher_moves = np.argmax(policies, axis=1)
    teacher_values = q_values[row_indexes, teacher_moves]
    rollout_values = q_values[row_indexes, rollout_moves]
    is_rollout_value_available = (
        q_weights[row_indexes, rollout_moves] > 0
    )
    regrets = np.where(
        is_rollout_value_available,
        teacher_values - rollout_values,
        0,
    )
    normalized_regrets = np.clip(
        regrets / COUNTERFACTUAL_REGRET_MAXIMUM,
        0,
        1,
    )
    return (
        COUNTERFACTUAL_REGRET_BASE_WEIGHT
        + (
            COUNTERFACTUAL_REGRET_MAXIMUM_WEIGHT
            - COUNTERFACTUAL_REGRET_BASE_WEIGHT
        )
        * normalized_regrets
    ).astype(np.float32)


def reweight_dataset(
    dataset_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int,
    use_nested_network: bool,
    use_spatial_network: bool,
    use_recurrent_network: bool,
    use_context_network: bool,
    use_wide_network: bool,
    mode: str,
) -> None:
    dataset = np.load(dataset_path)
    features = dataset["features"].astype(np.float32)
    policies = dataset["policies"].astype(np.float32)
    model = create_moka_network(
        use_nested_network,
        use_spatial_network,
        use_recurrent_network,
        use_context_network,
        use_wide_network,
    )
    model.load_weights(
        str(checkpoint_path),
        strict=not use_spatial_network,
    )
    model.eval()
    if mode == REWEIGHT_COUNTERFACTUAL_CRITICAL_MODE:
        weights = calculate_counterfactual_critical_weights(
            policies,
            dataset["q_values"].astype(np.float32),
            dataset["counterfactual_values"].astype(np.float32),
        )
    elif mode == REWEIGHT_ROLLOUT_REGRET_MODE:
        weights = calculate_rollout_regret_weights(
            policies,
            dataset["q_values"].astype(np.float32),
            dataset["q_weights"].astype(np.float32),
            dataset["rollout_moves"].astype(np.int64),
        )
    elif mode == REWEIGHT_COUNTERFACTUAL_MODE:
        weights = calculate_counterfactual_regret_weights(
            policies,
            dataset["q_values"].astype(np.float32),
            dataset["counterfactual_values"].astype(np.float32),
        )
    elif mode == REWEIGHT_DISAGREEMENT_MODE:
        weights = calculate_disagreement_weights(
            model,
            features,
            policies,
            batch_size,
        )
    else:
        weights = calculate_policy_surprise_weights(
            model,
            features,
            policies,
            batch_size,
        )
    output_values = {name: dataset[name] for name in dataset.files}
    if mode == REWEIGHT_COUNTERFACTUAL_CRITICAL_MODE:
        hard_policies = np.zeros_like(policies)
        hard_policies[
            np.arange(len(policies)),
            np.argmax(policies, axis=1),
        ] = 1
        output_values["policies"] = hard_policies.astype(np.float16)
    output_values["sample_weights"] = weights.astype(np.float16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output_values)
    print(
        f"saved {output_path} "
        f"mean={float(np.mean(weights)):.3f} "
        f"p95={float(np.percentile(weights, 95)):.3f} "
        f"max={float(np.max(weights)):.3f}"
    )


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data", required=True, type=Path)
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument("--output", required=True, type=Path)
    argument_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    argument_parser.add_argument("--nested", action="store_true")
    argument_parser.add_argument("--spatial", action="store_true")
    argument_parser.add_argument("--recurrent", action="store_true")
    argument_parser.add_argument("--context", action="store_true")
    argument_parser.add_argument("--wide", action="store_true")
    argument_parser.add_argument("--mode", choices=REWEIGHT_MODES, default=REWEIGHT_MODES[0])
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    reweight_dataset(
        arguments.data,
        arguments.checkpoint,
        arguments.output,
        arguments.batch_size,
        arguments.nested,
        arguments.spatial,
        arguments.recurrent,
        arguments.context,
        arguments.wide,
        arguments.mode,
    )


if __name__ == "__main__":
    main()
