import argparse
import hashlib
import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

from go_model.config import (
    BOARD_SIZE,
    INPUT_PLANE_COUNT,
    MAXIMUM_MODEL_BYTES,
    POLICY_CHANNEL_COUNT,
    POLICY_MOVE_COUNT,
    RESIDUAL_BLOCK_COUNT,
    SCORE_HIDDEN_CHANNEL_COUNT,
    TRUNK_CHANNEL_COUNT,
    VALUE_CHANNEL_COUNT,
)
from go_model.model import StudentNetwork


def get_named_parameters(model: StudentNetwork) -> list[tuple[str, mx.array]]:
    named_parameters = [
        ("stem.weight", model.stem.weight),
        ("stem.bias", model.stem.bias),
    ]

    for block_index, residual_block in enumerate(model.residual_blocks):
        prefix = f"residual.{block_index}"
        named_parameters.extend(
            [
                (
                    f"{prefix}.first.weight",
                    residual_block.first_convolution.weight,
                ),
                (
                    f"{prefix}.first.bias",
                    residual_block.first_convolution.bias,
                ),
                (
                    f"{prefix}.second.weight",
                    residual_block.second_convolution.weight,
                ),
                (
                    f"{prefix}.second.bias",
                    residual_block.second_convolution.bias,
                ),
            ]
        )

    named_parameters.extend(
        [
            ("policy.convolution.weight", model.policy_convolution.weight),
            ("policy.convolution.bias", model.policy_convolution.bias),
            ("policy.linear.weight", model.policy_linear.weight),
            ("policy.linear.bias", model.policy_linear.bias),
            ("value.convolution.weight", model.value_convolution.weight),
            ("value.convolution.bias", model.value_convolution.bias),
            ("value.hidden.weight", model.value_hidden.weight),
            ("value.hidden.bias", model.value_hidden.bias),
            ("value.output.weight", model.value_output.weight),
            ("value.output.bias", model.value_output.bias),
        ]
    )
    return named_parameters


def append_aligned(chunks: bytearray, values: bytes) -> int:
    while len(chunks) % 4:
        chunks.append(0)

    offset = len(chunks)
    chunks.extend(values)
    return offset


def export_model(
    checkpoint_path: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    model = StudentNetwork()
    model.load_weights(str(checkpoint_path))
    mx.eval(model.parameters())
    binary = bytearray()
    tensors: dict[str, dict[str, object]] = {}

    for name, parameter in get_named_parameters(model):
        values = np.asarray(parameter, dtype=np.float32)

        if name.endswith(".weight"):
            output_channel_count = values.shape[0]
            flattened_values = values.reshape(output_channel_count, -1)
            scales = np.max(np.abs(flattened_values), axis=1) / 127
            scales = np.maximum(scales, np.finfo(np.float32).eps).astype(np.float32)
            reshape_dimensions = (output_channel_count,) + (1,) * (values.ndim - 1)
            quantized_values = np.rint(values / scales.reshape(reshape_dimensions))
            quantized_values = np.clip(quantized_values, -127, 127).astype(np.int8)
            data_offset = append_aligned(binary, quantized_values.tobytes())
            scale_offset = append_aligned(binary, scales.tobytes())
            tensors[name] = {
                "dataOffset": data_offset,
                "dtype": "int8",
                "scaleOffset": scale_offset,
                "shape": list(values.shape),
            }
        else:
            data_offset = append_aligned(binary, values.tobytes())
            tensors[name] = {
                "dataOffset": data_offset,
                "dtype": "float32",
                "shape": list(values.shape),
            }

    output_directory.mkdir(parents=True, exist_ok=True)
    weights_path = output_directory / "go-model.bin"
    manifest_path = output_directory / "go-model.json"
    weights_path.write_bytes(binary)
    manifest = {
        "architecture": {
            "boardSize": BOARD_SIZE,
            "inputPlaneCount": INPUT_PLANE_COUNT,
            "policyChannelCount": POLICY_CHANNEL_COUNT,
            "policyMoveCount": POLICY_MOVE_COUNT,
            "residualBlockCount": RESIDUAL_BLOCK_COUNT,
            "scoreHiddenChannelCount": SCORE_HIDDEN_CHANNEL_COUNT,
            "trunkChannelCount": TRUNK_CHANNEL_COUNT,
            "valueChannelCount": VALUE_CHANNEL_COUNT,
        },
        "format": "million-go-int8",
        "sha256": hashlib.sha256(binary).hexdigest(),
        "tensors": tensors,
        "version": 1,
        "weightsBytes": len(binary),
    }
    serialized_manifest = json.dumps(manifest, indent=2, sort_keys=True)
    formatted_manifest = re.sub(
        r"\[\n\s+((?:-?\d+,?\n\s*)+)\]",
        lambda match: "[" + ", ".join(re.findall(r"-?\d+", match.group(1))) + "]",
        serialized_manifest,
    )
    manifest_path.write_text(f"{formatted_manifest}\n", encoding="utf-8")

    if len(binary) > MAXIMUM_MODEL_BYTES:
        raise RuntimeError(
            f"artifact is {len(binary):,} bytes; budget is {MAXIMUM_MODEL_BYTES:,}"
        )

    print(f"exported {weights_path} ({len(binary):,} bytes)")
    print(f"exported {manifest_path} ({manifest_path.stat().st_size:,} bytes)")
    return weights_path, manifest_path


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/go-model.safetensors"),
    )
    argument_parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist"),
    )
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    export_model(arguments.checkpoint, arguments.output)


if __name__ == "__main__":
    main()
