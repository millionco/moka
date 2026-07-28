import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from go_model.quantization import (
    fake_quantize_int8_parameters,
    fake_quantize_int8_weight,
    materialize_int8_checkpoint,
)


class QuantizationTest(unittest.TestCase):
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
