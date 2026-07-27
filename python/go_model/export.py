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
    INT4_GROUP_SIZE,
    INT4_MAXIMUM_VALUE,
    INT4_ZERO_POINT,
    INT8_MAXIMUM_VALUE,
    MAXIMUM_MODEL_BYTES,
    MODEL_QUANTIZATION_BITS,
    NESTED_BOTTLENECK_CHANNEL_COUNT,
    NESTED_RESIDUAL_BLOCK_COUNT,
    POLICY_CHANNEL_COUNT,
    POLICY_MOVE_COUNT,
    RESIDUAL_BLOCK_COUNT,
    SCORE_HIDDEN_CHANNEL_COUNT,
    TRUNK_CHANNEL_COUNT,
    VALUE_CHANNEL_COUNT,
)
from go_model.model import MokaNestedNetwork, MokaNetwork


def get_named_parameters(
    model: MokaNetwork,
    use_nested_network: bool,
) -> list[tuple[str, mx.array]]:
    named_parameters = [
        ("stem.weight", model.stem.weight),
        ("stem.bias", model.stem.bias),
    ]

    for block_index, residual_block in enumerate(model.residual_blocks):
        prefix = f"residual.{block_index}"
        if use_nested_network:
            named_parameters.extend(
                [
                    (
                        f"{prefix}.reduce.weight",
                        residual_block.reduce_convolution.weight,
                    ),
                    (
                        f"{prefix}.reduce.bias",
                        residual_block.reduce_convolution.bias,
                    ),
                    (
                        f"{prefix}.first.weight",
                        residual_block.first_spatial_convolution.weight,
                    ),
                    (
                        f"{prefix}.first.bias",
                        residual_block.first_spatial_convolution.bias,
                    ),
                    (
                        f"{prefix}.second.weight",
                        residual_block.second_spatial_convolution.weight,
                    ),
                    (
                        f"{prefix}.second.bias",
                        residual_block.second_spatial_convolution.bias,
                    ),
                    (
                        f"{prefix}.expand.weight",
                        residual_block.expand_convolution.weight,
                    ),
                    (
                        f"{prefix}.expand.bias",
                        residual_block.expand_convolution.bias,
                    ),
                ]
            )
        else:
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


def pack_int4(values: np.ndarray) -> bytes:
    encoded_values = values.astype(np.int16) + INT4_ZERO_POINT
    if len(encoded_values) % 2:
        encoded_values = np.append(encoded_values, INT4_ZERO_POINT)
    packed_values = (
        encoded_values[::2]
        | (encoded_values[1::2] << 4)
    ).astype(np.uint8)
    return packed_values.tobytes()


def export_model(
    checkpoint_path: Path,
    output_directory: Path,
    quantization_bits: int,
    use_nested_network: bool = False,
) -> tuple[Path, Path]:
    model = MokaNestedNetwork() if use_nested_network else MokaNetwork()
    model.load_weights(str(checkpoint_path))
    mx.eval(model.parameters())
    binary = bytearray()
    tensors: dict[str, dict[str, object]] = {}

    for name, parameter in get_named_parameters(model, use_nested_network):
        values = np.asarray(parameter, dtype=np.float32)

        if name.endswith(".weight"):
            output_channel_count = values.shape[0]
            flattened_values = values.reshape(output_channel_count, -1)
            maximum_quantized_value = (
                INT4_MAXIMUM_VALUE
                if quantization_bits == 4
                else INT8_MAXIMUM_VALUE
            )
            if quantization_bits == 4:
                values_per_output_channel = flattened_values.shape[1]
                group_count = int(
                    np.ceil(values_per_output_channel / INT4_GROUP_SIZE)
                )
                padded_value_count = group_count * INT4_GROUP_SIZE
                padded_values = np.pad(
                    flattened_values,
                    (
                        (0, 0),
                        (0, padded_value_count - values_per_output_channel),
                    ),
                )
                grouped_values = padded_values.reshape(
                    output_channel_count,
                    group_count,
                    INT4_GROUP_SIZE,
                )
                scales = (
                    np.max(np.abs(grouped_values), axis=2)
                    / maximum_quantized_value
                )
                scales = np.maximum(
                    scales,
                    np.finfo(np.float32).eps,
                ).astype(np.float32)
                expanded_scales = np.repeat(
                    scales,
                    INT4_GROUP_SIZE,
                    axis=1,
                )[:, :values_per_output_channel]
                quantized_values = np.rint(
                    flattened_values / expanded_scales
                ).reshape(values.shape)
            else:
                scales = (
                    np.max(np.abs(flattened_values), axis=1)
                    / maximum_quantized_value
                )
                scales = np.maximum(
                    scales,
                    np.finfo(np.float32).eps,
                ).astype(np.float32)
                reshape_dimensions = (output_channel_count,) + (1,) * (
                    values.ndim - 1
                )
                quantized_values = np.rint(
                    values / scales.reshape(reshape_dimensions)
                )
            quantized_values = np.clip(
                quantized_values,
                -maximum_quantized_value,
                maximum_quantized_value,
            ).astype(np.int8)
            serialized_values = (
                pack_int4(quantized_values.reshape(-1))
                if quantization_bits == 4
                else quantized_values.tobytes()
            )
            data_offset = append_aligned(binary, serialized_values)
            scale_offset = append_aligned(binary, scales.tobytes())
            tensors[name] = {
                "dataOffset": data_offset,
                "dtype": f"int{quantization_bits}",
                "scaleOffset": scale_offset,
                "shape": list(values.shape),
            }
            if quantization_bits == 4:
                tensors[name]["quantizationGroupSize"] = INT4_GROUP_SIZE
        else:
            data_offset = append_aligned(binary, values.tobytes())
            tensors[name] = {
                "dataOffset": data_offset,
                "dtype": "float32",
                "shape": list(values.shape),
            }

    output_directory.mkdir(parents=True, exist_ok=True)
    weights_path = output_directory / "moka-model.bin"
    manifest_path = output_directory / "moka-model.json"
    weights_path.write_bytes(binary)
    architecture = {
            "boardSize": BOARD_SIZE,
            "inputPlaneCount": INPUT_PLANE_COUNT,
            "policyChannelCount": POLICY_CHANNEL_COUNT,
            "policyMoveCount": POLICY_MOVE_COUNT,
            "residualBlockCount": (
                NESTED_RESIDUAL_BLOCK_COUNT
                if use_nested_network
                else RESIDUAL_BLOCK_COUNT
            ),
            "residualBlockKind": (
                "nested-bottleneck"
                if use_nested_network
                else "standard"
            ),
            "scoreHiddenChannelCount": SCORE_HIDDEN_CHANNEL_COUNT,
            "trunkChannelCount": TRUNK_CHANNEL_COUNT,
            "valueChannelCount": VALUE_CHANNEL_COUNT,
    }
    if use_nested_network:
        architecture["bottleneckChannelCount"] = (
            NESTED_BOTTLENECK_CHANNEL_COUNT
        )
    manifest = {
        "architecture": architecture,
        "format": f"million-go-int{quantization_bits}",
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
        default=Path("checkpoints/moka-model.safetensors"),
    )
    argument_parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist"),
    )
    argument_parser.add_argument(
        "--quantization-bits",
        choices=MODEL_QUANTIZATION_BITS,
        default=8,
        type=int,
    )
    argument_parser.add_argument("--nested", action="store_true")
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    export_model(
        arguments.checkpoint,
        arguments.output,
        arguments.quantization_bits,
        arguments.nested,
    )


if __name__ == "__main__":
    main()
