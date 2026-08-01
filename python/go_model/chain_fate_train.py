import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from go_model.chain_fate import (
    ChainFateExamples,
    ChainFateModel,
    extract_chain_fate_examples,
    fit_chain_fate_value_coefficient_from_signals,
    train_chain_fate_model,
)
from go_model.config import (
    CHAIN_FATE_RIDGE_WEIGHT,
    CHAIN_FATE_VALUE_LIMIT,
    DEFAULT_BATCH_SIZE,
)
from go_model.model import create_moka_network_for_checkpoint


@dataclass
class ChainFateDomain:
    name: str
    examples: ChainFateExamples
    encoded_positions: np.ndarray
    teacher_values: np.ndarray
    moka_values: np.ndarray


def evaluate_moka_values(
    checkpoint_path: Path,
    encoded_position_domains: list[np.ndarray],
) -> list[np.ndarray]:
    model = create_moka_network_for_checkpoint(str(checkpoint_path))
    model.load_weights(str(checkpoint_path))
    model.eval()
    domain_values: list[np.ndarray] = []

    for encoded_positions in encoded_position_domains:
        batches = []
        for start_index in range(
            0,
            len(encoded_positions),
            DEFAULT_BATCH_SIZE,
        ):
            _, values, *_ = model(
                mx.array(
                    encoded_positions[
                        start_index : start_index + DEFAULT_BATCH_SIZE
                    ]
                )
            )
            mx.eval(values)
            batches.append(np.asarray(values).reshape(-1))
        domain_values.append(np.concatenate(batches))

    return domain_values


def load_chain_fate_domains(
    dataset_paths: list[Path],
    checkpoint_path: Path,
) -> list[ChainFateDomain]:
    raw_domains = []
    for dataset_path in dataset_paths:
        with np.load(dataset_path) as dataset:
            if "ownerships" not in dataset:
                raise ValueError(
                    f"Dataset lacks ownerships: {dataset_path}"
                )
            encoded_positions = dataset["features"].astype(np.float32)
            ownerships = dataset["ownerships"].astype(np.float32)
            teacher_values = dataset["values"].astype(np.float32)
        raw_domains.append(
            (
                dataset_path.name,
                encoded_positions,
                teacher_values,
                extract_chain_fate_examples(
                    encoded_positions,
                    ownerships,
                ),
            )
        )

    moka_value_domains = evaluate_moka_values(
        checkpoint_path,
        [domain[1] for domain in raw_domains],
    )
    return [
        ChainFateDomain(
            name=name,
            encoded_positions=encoded_positions,
            teacher_values=teacher_values,
            examples=examples,
            moka_values=moka_values,
        )
        for (
            name,
            encoded_positions,
            teacher_values,
            examples,
        ), moka_values in zip(
            raw_domains,
            moka_value_domains,
            strict=True,
        )
    ]


def fit_phase_baseline(
    training_features: np.ndarray,
    training_targets: np.ndarray,
) -> np.ndarray:
    selected_features = training_features[:, [0, 13]]
    design = np.column_stack(
        [selected_features, np.ones(len(selected_features))]
    )
    return np.linalg.solve(
        design.T @ design
        + CHAIN_FATE_RIDGE_WEIGHT * np.eye(design.shape[1]),
        design.T @ training_targets,
    )


def predict_phase_baseline(
    baseline_weights: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    selected_features = features[:, [0, 13]]
    design = np.column_stack(
        [selected_features, np.ones(len(selected_features))]
    )
    return np.clip(design @ baseline_weights, 0, 1)


def get_value_residuals(
    moka_values: np.ndarray,
    teacher_values: np.ndarray,
) -> np.ndarray:
    return np.arctanh(
        np.clip(
            teacher_values,
            -CHAIN_FATE_VALUE_LIMIT,
            CHAIN_FATE_VALUE_LIMIT,
        )
    ) - np.arctanh(
        np.clip(
            moka_values,
            -CHAIN_FATE_VALUE_LIMIT,
            CHAIN_FATE_VALUE_LIMIT,
        )
    )


def evaluate_cross_domain_fold(
    training_domains: list[ChainFateDomain],
    test_domain: ChainFateDomain,
) -> tuple[dict[str, float | int | str], ChainFateModel]:
    training_features = np.concatenate(
        [domain.examples.features for domain in training_domains]
    )
    training_targets = np.concatenate(
        [domain.examples.targets for domain in training_domains]
    )
    model = train_chain_fate_model(
        training_features,
        training_targets,
    )
    training_signals = np.concatenate(
        [model.get_position_signals(domain.examples) for domain in training_domains]
    )
    training_moka_values = np.concatenate(
        [domain.moka_values for domain in training_domains]
    )
    training_teacher_values = np.concatenate(
        [domain.teacher_values for domain in training_domains]
    )
    model = fit_chain_fate_value_coefficient_from_signals(
        model,
        training_signals,
        training_moka_values,
        training_teacher_values,
    )
    chain_predictions = model.predict_chain_survival(
        test_domain.examples.features
    )
    baseline_weights = fit_phase_baseline(
        training_features,
        training_targets,
    )
    baseline_predictions = predict_phase_baseline(
        baseline_weights,
        test_domain.examples.features,
    )
    position_signals = model.get_position_signals(test_domain.examples)
    corrected_values = model.correct_values(
        test_domain.moka_values,
        position_signals,
    )
    value_residuals = get_value_residuals(
        test_domain.moka_values,
        test_domain.teacher_values,
    )
    raw_errors = test_domain.moka_values - test_domain.teacher_values
    corrected_errors = corrected_values - test_domain.teacher_values
    raw_sign_matches = int(
        np.sum(
            np.sign(test_domain.moka_values)
            == np.sign(test_domain.teacher_values)
        )
    )
    corrected_sign_matches = int(
        np.sum(
            np.sign(corrected_values)
            == np.sign(test_domain.teacher_values)
        )
    )
    return (
        {
            "domain": test_domain.name,
            "chain_count": len(test_domain.examples.features),
            "position_count": len(test_domain.encoded_positions),
            "ownership_brier": float(
                np.mean(
                    (
                        chain_predictions
                        - test_domain.examples.targets
                    )
                    ** 2
                )
            ),
            "baseline_ownership_brier": float(
                np.mean(
                    (
                        baseline_predictions
                        - test_domain.examples.targets
                    )
                    ** 2
                )
            ),
            "signal_residual_correlation": float(
                np.corrcoef(position_signals, value_residuals)[0, 1]
            ),
            "value_coefficient": model.value_coefficient,
            "raw_value_mse": float(np.mean(raw_errors**2)),
            "corrected_value_mse": float(
                np.mean(corrected_errors**2)
            ),
            "raw_value_mae": float(np.mean(np.abs(raw_errors))),
            "corrected_value_mae": float(
                np.mean(np.abs(corrected_errors))
            ),
            "raw_sign_matches": raw_sign_matches,
            "corrected_sign_matches": corrected_sign_matches,
        },
        model,
    )


def train_final_model(domains: list[ChainFateDomain]) -> ChainFateModel:
    model = train_chain_fate_model(
        np.concatenate([domain.examples.features for domain in domains]),
        np.concatenate([domain.examples.targets for domain in domains]),
    )
    return fit_chain_fate_value_coefficient_from_signals(
        model,
        np.concatenate(
            [model.get_position_signals(domain.examples) for domain in domains]
        ),
        np.concatenate([domain.moka_values for domain in domains]),
        np.concatenate([domain.teacher_values for domain in domains]),
    )


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        required=True,
    )
    argument_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    argument_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    if len(arguments.dataset) < 2:
        raise ValueError("Chain-fate training requires at least two domains.")
    domains = load_chain_fate_domains(
        arguments.dataset,
        arguments.checkpoint,
    )
    fold_metrics = []
    for test_index, test_domain in enumerate(domains):
        training_domains = [
            domain
            for domain_index, domain in enumerate(domains)
            if domain_index != test_index
        ]
        metrics, _ = evaluate_cross_domain_fold(
            training_domains,
            test_domain,
        )
        fold_metrics.append(metrics)

    final_model = train_final_model(domains)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(arguments.output)
    print(
        json.dumps(
            {
                "folds": fold_metrics,
                "artifact": str(arguments.output),
                "artifact_bytes": arguments.output.stat().st_size,
                "value_coefficient": final_model.value_coefficient,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
