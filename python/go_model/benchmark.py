import argparse
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from go_model.config import BOARD_SIZE, INPUT_PLANE_COUNT
from go_model.model import create_moka_network_for_checkpoint


def percentile(values: list[float], percentile_value: float) -> float:
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, round((len(sorted_values) - 1) * percentile_value))
    return sorted_values[index]


def benchmark(checkpoint_path: Path, iteration_count: int) -> None:
    model = create_moka_network_for_checkpoint(str(checkpoint_path))

    if checkpoint_path.exists():
        model.load_weights(str(checkpoint_path))

    features = mx.array(
        np.zeros((1, BOARD_SIZE, BOARD_SIZE, INPUT_PLANE_COUNT), dtype=np.float32)
    )
    compiled_model = mx.compile(lambda inputs: model(inputs))

    for _ in range(10):
        outputs = compiled_model(features)
        mx.eval(outputs)

    durations_ms: list[float] = []

    for _ in range(iteration_count):
        start_time = time.perf_counter()
        outputs = compiled_model(features)
        mx.eval(outputs)
        durations_ms.append((time.perf_counter() - start_time) * 1_000)

    parameter_count = sum(
        parameter.size
        for _, parameter in tree_flatten(model.parameters())
    )
    print(f"parameters: {parameter_count:,}")
    print(f"float32 weights: {parameter_count * 4:,} bytes")
    print(f"mean inference: {statistics.mean(durations_ms):.3f} ms")
    print(f"p50 inference: {percentile(durations_ms, 0.5):.3f} ms")
    print(f"p95 inference: {percentile(durations_ms, 0.95):.3f} ms")


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/moka-model.safetensors"),
    )
    argument_parser.add_argument("--iterations", type=int, default=200)
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    benchmark(arguments.checkpoint, arguments.iterations)


if __name__ == "__main__":
    main()
