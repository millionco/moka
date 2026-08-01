import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from go_model.chain_fate import load_chain_fate_model
from go_model.config import DEFAULT_BATCH_SIZE
from go_model.model import create_moka_network_for_checkpoint


def evaluate_moka_values(
    checkpoint_path: Path,
    features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model = create_moka_network_for_checkpoint(str(checkpoint_path))
    model.load_weights(str(checkpoint_path))
    model.eval()
    value_batches: list[np.ndarray] = []

    for batch_start in range(0, len(features), batch_size):
        _, values = model(
            mx.array(
                features[batch_start : batch_start + batch_size],
                dtype=mx.float32,
            )
        )
        mx.eval(values)
        value_batches.append(np.asarray(values, dtype=np.float32))

    return np.concatenate(value_batches)


def create_chain_fate_distillation_dataset(
    source_path: Path,
    checkpoint_path: Path,
    chain_fate_model_path: Path,
    output_path: Path,
    correction_scale: float,
    batch_size: int,
) -> dict[str, float | int | str]:
    if not 0 <= correction_scale <= 1:
        raise ValueError("Correction scale must be between zero and one.")
    with np.load(source_path) as source:
        features = source["features"].astype(np.float32)
        game_ids = source["game_ids"]
        sample_weights = (
            source["weights"].astype(np.float32)
            if "weights" in source
            else np.ones(len(features), dtype=np.float32)
        )

    raw_values = evaluate_moka_values(
        checkpoint_path,
        features,
        batch_size,
    )
    chain_fate_model = load_chain_fate_model(chain_fate_model_path)
    chain_fate_model.value_coefficient *= correction_scale
    corrected_values = chain_fate_model.correct_encoded_values(
        features,
        raw_values,
    ).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features,
        game_ids=game_ids,
        values=corrected_values,
        weights=sample_weights,
    )
    value_changes = corrected_values - raw_values
    metrics: dict[str, float | int | str] = {
        "source": str(source_path),
        "output": str(output_path),
        "position_count": len(features),
        "correction_scale": correction_scale,
        "mean_absolute_correction": float(np.mean(np.abs(value_changes))),
        "maximum_absolute_correction": float(np.max(np.abs(value_changes))),
        "changed_value_count": int(np.count_nonzero(value_changes)),
        "sign_change_count": int(
            np.count_nonzero(np.sign(raw_values) != np.sign(corrected_values))
        ),
        "artifact_bytes": output_path.stat().st_size,
    }
    return metrics


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--source", required=True, type=Path)
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument(
        "--chain-fate-model",
        required=True,
        type=Path,
    )
    argument_parser.add_argument("--output", required=True, type=Path)
    argument_parser.add_argument(
        "--correction-scale",
        type=float,
        default=1,
    )
    argument_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    metrics = create_chain_fate_distillation_dataset(
        arguments.source,
        arguments.checkpoint,
        arguments.chain_fate_model,
        arguments.output,
        arguments.correction_scale,
        arguments.batch_size,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
