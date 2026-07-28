import argparse
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from go_model.config import INT8_MAXIMUM_VALUE


def fake_quantize_int8_weight(values: mx.array) -> mx.array:
    flattened_values = values.reshape(values.shape[0], -1)
    scales = mx.maximum(
        mx.max(mx.abs(flattened_values), axis=1) / INT8_MAXIMUM_VALUE,
        mx.finfo(mx.float32).eps,
    )
    scale_shape = (values.shape[0],) + (1,) * (values.ndim - 1)
    expanded_scales = scales.reshape(scale_shape)
    quantized_values = (
        mx.clip(
            mx.round(values / expanded_scales),
            -INT8_MAXIMUM_VALUE,
            INT8_MAXIMUM_VALUE,
        )
        * expanded_scales
    )
    return values + mx.stop_gradient(quantized_values - values)


def fake_quantize_int8_parameters(parameters: dict) -> dict:
    quantized_parameters = [
        (
            name,
            fake_quantize_int8_weight(values)
            if name.endswith(".weight")
            else values,
        )
        for name, values in tree_flatten(parameters)
    ]
    return tree_unflatten(quantized_parameters)


def materialize_int8_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
) -> None:
    parameters = mx.load(str(checkpoint_path))
    quantized_parameters = fake_quantize_int8_parameters(parameters)
    mx.eval(quantized_parameters)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(output_path),
        dict(tree_flatten(quantized_parameters)),
    )
    print(f"saved {output_path} bytes={output_path.stat().st_size:,}")


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--checkpoint", required=True, type=Path)
    argument_parser.add_argument("--output", required=True, type=Path)
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    materialize_int8_checkpoint(
        arguments.checkpoint,
        arguments.output,
    )


if __name__ == "__main__":
    main()
