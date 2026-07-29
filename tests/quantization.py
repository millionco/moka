import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from go_model.config import (
    BOARD_SIZE,
    GLOBAL_RESIDUAL_BLOCK_INTERVAL,
    GLOBAL_RESIDUAL_HIDDEN_CHANNEL_COUNT,
    INPUT_PLANE_COUNT,
)
from go_model.export import export_model
from go_model.model import (
    MokaGlobalResidualNetwork,
    MokaNestedNetwork,
    checkpoint_uses_global_residual_network,
    checkpoint_uses_nested_network,
    create_moka_network_for_checkpoint,
)
from go_model.quantization import (
    fake_quantize_int8_parameters,
    fake_quantize_int8_weight,
    materialize_int8_checkpoint,
)


class QuantizationTest(unittest.TestCase):
    def test_global_residual_initialization_preserves_nested_output(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = (
                Path(temporary_directory) / "nested.safetensors"
            )
            nested_model = MokaNestedNetwork()
            nested_model.save_weights(str(checkpoint_path))
            global_model = MokaGlobalResidualNetwork()
            global_model.load_weights(
                str(checkpoint_path),
                strict=False,
            )
            features = mx.zeros(
                (
                    2,
                    BOARD_SIZE,
                    BOARD_SIZE,
                    INPUT_PLANE_COUNT,
                )
            )
            nested_policy, nested_value = nested_model(features)
            global_policy, global_value = global_model(features)
            mx.eval(
                nested_policy,
                nested_value,
                global_policy,
                global_value,
            )

            np.testing.assert_array_equal(
                np.asarray(global_policy),
                np.asarray(nested_policy),
            )
            np.testing.assert_array_equal(
                np.asarray(global_value),
                np.asarray(nested_value),
            )

    def test_global_residual_export_declares_adapter_tensors(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            checkpoint_path = temporary_path / "global.safetensors"
            output_directory = temporary_path / "export"
            MokaGlobalResidualNetwork().save_weights(
                str(checkpoint_path)
            )

            _, manifest_path = export_model(
                checkpoint_path,
                output_directory,
                8,
                use_global_residual_network=True,
            )

            manifest = json.loads(manifest_path.read_text())
            architecture = manifest["architecture"]
            self.assertEqual(
                architecture["globalResidualBlockInterval"],
                GLOBAL_RESIDUAL_BLOCK_INTERVAL,
            )
            self.assertEqual(
                architecture["globalResidualHiddenChannelCount"],
                GLOBAL_RESIDUAL_HIDDEN_CHANNEL_COUNT,
            )
            self.assertIn(
                "residual.3.global.hidden.weight",
                manifest["tensors"],
            )
            self.assertIn(
                "residual.11.global.output.bias",
                manifest["tensors"],
            )

    def test_global_residual_checkpoint_selects_matching_network(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            nested_checkpoint_path = temporary_path / "nested.safetensors"
            global_checkpoint_path = temporary_path / "global.safetensors"
            MokaNestedNetwork().save_weights(str(nested_checkpoint_path))
            MokaGlobalResidualNetwork().save_weights(
                str(global_checkpoint_path)
            )

            self.assertFalse(
                checkpoint_uses_global_residual_network(
                    str(nested_checkpoint_path)
                )
            )
            self.assertTrue(
                checkpoint_uses_nested_network(str(nested_checkpoint_path))
            )
            self.assertTrue(
                checkpoint_uses_global_residual_network(
                    str(global_checkpoint_path)
                )
            )
            self.assertIsInstance(
                create_moka_network_for_checkpoint(
                    str(nested_checkpoint_path)
                ),
                MokaNestedNetwork,
            )
            self.assertIsInstance(
                create_moka_network_for_checkpoint(
                    str(global_checkpoint_path)
                ),
                MokaGlobalResidualNetwork,
            )

    def test_materialized_checkpoint_matches_fake_quantized_parameters(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "source.safetensors"
            output_path = Path(temporary_directory) / "quantized.safetensors"
            parameters = {
                "layer.weight": mx.array(
                    [[-1.5, -0.25, 0.5, 1.25]],
                    dtype=mx.float32,
                ),
                "layer.bias": mx.array([0.125], dtype=mx.float32),
            }
            mx.save_safetensors(str(checkpoint_path), parameters)

            materialize_int8_checkpoint(checkpoint_path, output_path)

            materialized_parameters = mx.load(str(output_path))
            expected_parameters = dict(
                tree_flatten(fake_quantize_int8_parameters(parameters))
            )
            for parameter_name in parameters:
                np.testing.assert_array_equal(
                    np.asarray(materialized_parameters[parameter_name]),
                    np.asarray(expected_parameters[parameter_name]),
                )

    def test_fake_quantization_matches_per_output_channel_int8(self) -> None:
        values = mx.array(
            [
                [0.0, 0.5, -1.0],
                [0.25, -0.25, 0.125],
            ],
            dtype=mx.float32,
        )
        quantized_values = np.asarray(fake_quantize_int8_weight(values))
        expected_scales = np.array([1.0 / 127, 0.25 / 127], dtype=np.float32)
        expected_values = (
            np.clip(
                np.rint(np.asarray(values) / expected_scales[:, None]),
                -127,
                127,
            )
            * expected_scales[:, None]
        )
        np.testing.assert_allclose(quantized_values, expected_values)

    def test_fake_quantization_uses_straight_through_gradient(self) -> None:
        values = mx.array([[0.2, -0.4, 0.7]], dtype=mx.float32)
        gradients = mx.grad(
            lambda weights: mx.sum(fake_quantize_int8_weight(weights))
        )(values)
        np.testing.assert_array_equal(
            np.asarray(gradients),
            np.ones_like(np.asarray(values)),
        )

    def test_parameter_quantization_leaves_biases_unchanged(self) -> None:
        parameters = {
            "layer": {
                "weight": mx.array([[0.2, -0.4]], dtype=mx.float32),
                "bias": mx.array([0.3], dtype=mx.float32),
            }
        }
        quantized_parameters = dict(
            tree_flatten(fake_quantize_int8_parameters(parameters))
        )
        np.testing.assert_array_equal(
            np.asarray(quantized_parameters["layer.bias"]),
            np.asarray(parameters["layer"]["bias"]),
        )
        self.assertFalse(
            np.array_equal(
                np.asarray(quantized_parameters["layer.weight"]),
                np.asarray(parameters["layer"]["weight"]),
            )
        )


if __name__ == "__main__":
    unittest.main()
