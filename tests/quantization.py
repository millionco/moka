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
    MOKA_CURRENT_PLAYER_FEATURE_INDEX,
    MOKA_CURRENT_PLAYER_ONE_LIBERTY_FEATURE_INDEX,
    MOKA_CURRENT_PLAYER_TWO_LIBERTY_FEATURE_INDEX,
    MOKA_OPPONENT_FEATURE_INDEX,
)
from go_model.export import export_model
from go_model.model import (
    MokaActionValueNetwork,
    MokaAuxiliaryNetwork,
    MokaGatedGlobalResidualNetwork,
    MokaGlobalResidualNetwork,
    MokaGlobalScoreNetwork,
    MokaGlobalValueNetwork,
    MokaHeuristicAdapterNetwork,
    MokaNestedNetwork,
    checkpoint_uses_global_residual_network,
    checkpoint_uses_global_score_network,
    checkpoint_uses_global_value_network,
    checkpoint_uses_gated_global_residual_network,
    checkpoint_uses_heuristic_adapter_network,
    checkpoint_uses_action_value_network,
    checkpoint_uses_nested_network,
    create_moka_network_for_checkpoint,
    get_checkpoint_global_residual_block_interval,
)
from go_model.quantization import (
    fake_quantize_int8_parameters,
    fake_quantize_int8_weight,
    materialize_int8_checkpoint,
)


class QuantizationTest(unittest.TestCase):
    def test_action_value_initialization_preserves_outputs(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "global.safetensors"
            action_checkpoint_path = (
                Path(temporary_directory) / "action.safetensors"
            )
            global_model = MokaGlobalResidualNetwork()
            global_model.save_weights(str(checkpoint_path))
            action_model = MokaActionValueNetwork()
            action_model.load_weights(str(checkpoint_path), strict=False)
            features = mx.random.normal(
                (
                    2,
                    BOARD_SIZE,
                    BOARD_SIZE,
                    INPUT_PLANE_COUNT,
                )
            )
            global_outputs = global_model(features)
            action_outputs = action_model.get_action_value_outputs(features)
            mx.eval(global_outputs, action_outputs)

            np.testing.assert_array_equal(
                np.asarray(action_outputs[0]),
                np.asarray(global_outputs[0]),
            )
            np.testing.assert_array_equal(
                np.asarray(action_outputs[1]),
                np.asarray(global_outputs[1]),
            )
            np.testing.assert_array_equal(
                np.asarray(action_outputs[2]),
                np.zeros((2, 82), dtype=np.float32),
            )

            action_model.save_weights(str(action_checkpoint_path))
            self.assertTrue(
                checkpoint_uses_action_value_network(
                    str(action_checkpoint_path)
                )
            )
            self.assertIsInstance(
                create_moka_network_for_checkpoint(
                    str(action_checkpoint_path)
                ),
                MokaActionValueNetwork,
            )

    def test_heuristic_adapter_features_encode_high_liberties_and_phase(
        self,
    ) -> None:
        model = MokaHeuristicAdapterNetwork()
        features = np.zeros(
            (1, BOARD_SIZE, BOARD_SIZE, INPUT_PLANE_COUNT),
            dtype=np.float32,
        )
        features[0, 0, 0, MOKA_CURRENT_PLAYER_FEATURE_INDEX] = 1
        features[0, 0, 1, MOKA_CURRENT_PLAYER_FEATURE_INDEX] = 1
        features[
            0, 0, 1, MOKA_CURRENT_PLAYER_ONE_LIBERTY_FEATURE_INDEX
        ] = 1
        features[0, 0, 2, MOKA_CURRENT_PLAYER_FEATURE_INDEX] = 1
        features[
            0, 0, 2, MOKA_CURRENT_PLAYER_TWO_LIBERTY_FEATURE_INDEX
        ] = 1
        features[0, 1, 0, MOKA_OPPONENT_FEATURE_INDEX] = 1

        heuristic_features = np.asarray(
            model.get_heuristic_features(mx.array(features))
        )

        self.assertEqual(heuristic_features[0, 0, 0, 0], 1)
        self.assertEqual(heuristic_features[0, 0, 1, 0], 0)
        self.assertEqual(heuristic_features[0, 0, 2, 0], 0)
        self.assertEqual(heuristic_features[0, 1, 0, 1], 1)
        np.testing.assert_allclose(
            heuristic_features[0, :, :, 2],
            4 / (BOARD_SIZE * BOARD_SIZE),
        )

    def test_heuristic_adapter_initialization_preserves_outputs(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "global.safetensors"
            adapter_checkpoint_path = (
                Path(temporary_directory) / "adapter.safetensors"
            )
            global_model = MokaGlobalResidualNetwork()
            global_model.save_weights(str(checkpoint_path))
            adapter_model = MokaHeuristicAdapterNetwork()
            adapter_model.load_weights(str(checkpoint_path), strict=False)
            features = mx.random.normal(
                (
                    2,
                    BOARD_SIZE,
                    BOARD_SIZE,
                    INPUT_PLANE_COUNT,
                )
            )
            global_outputs = global_model(features)
            adapter_outputs = adapter_model(features)
            mx.eval(global_outputs, adapter_outputs)

            for global_output, adapter_output in zip(
                global_outputs,
                adapter_outputs,
                strict=True,
            ):
                np.testing.assert_array_equal(
                    np.asarray(adapter_output),
                    np.asarray(global_output),
                )

            adapter_model.save_weights(str(adapter_checkpoint_path))
            self.assertTrue(
                checkpoint_uses_heuristic_adapter_network(
                    str(adapter_checkpoint_path)
                )
            )
            self.assertIsInstance(
                create_moka_network_for_checkpoint(
                    str(adapter_checkpoint_path)
                ),
                MokaHeuristicAdapterNetwork,
            )

    def test_global_residual_auxiliary_initialization_preserves_outputs(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = (
                Path(temporary_directory) / "global.safetensors"
            )
            global_model = MokaGlobalResidualNetwork()
            global_model.save_weights(str(checkpoint_path))
            auxiliary_model = MokaAuxiliaryNetwork(
                GLOBAL_RESIDUAL_BLOCK_INTERVAL
            )
            auxiliary_model.load_weights(
                str(checkpoint_path),
                strict=False,
            )
            features = mx.random.normal(
                (
                    2,
                    BOARD_SIZE,
                    BOARD_SIZE,
                    INPUT_PLANE_COUNT,
                )
            )
            global_policy, global_value = global_model(features)
            auxiliary_outputs = auxiliary_model.get_training_outputs(
                features
            )
            mx.eval(global_policy, global_value, auxiliary_outputs)

            np.testing.assert_array_equal(
                np.asarray(auxiliary_outputs[0]),
                np.asarray(global_policy),
            )
            np.testing.assert_array_equal(
                np.asarray(auxiliary_outputs[1]),
                np.asarray(global_value),
            )

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
            self.assertEqual(
                get_checkpoint_global_residual_block_interval(
                    str(global_checkpoint_path)
                ),
                GLOBAL_RESIDUAL_BLOCK_INTERVAL,
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

    def test_denser_global_residual_checkpoint_preserves_and_exports(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_checkpoint_path = temporary_path / "source.safetensors"
            dense_checkpoint_path = temporary_path / "dense.safetensors"
            output_directory = temporary_path / "export"
            source_model = MokaGlobalResidualNetwork()
            source_model.save_weights(str(source_checkpoint_path))
            dense_model = MokaGlobalResidualNetwork(2)
            dense_model.load_weights(
                str(source_checkpoint_path),
                strict=False,
            )
            features = mx.random.normal(
                (
                    2,
                    BOARD_SIZE,
                    BOARD_SIZE,
                    INPUT_PLANE_COUNT,
                )
            )
            source_outputs = source_model(features)
            dense_outputs = dense_model(features)
            mx.eval(source_outputs, dense_outputs)

            for source_output, dense_output in zip(
                source_outputs,
                dense_outputs,
                strict=True,
            ):
                np.testing.assert_array_equal(
                    np.asarray(source_output),
                    np.asarray(dense_output),
                )

            dense_model.save_weights(str(dense_checkpoint_path))
            self.assertEqual(
                get_checkpoint_global_residual_block_interval(
                    str(dense_checkpoint_path)
                ),
                2,
            )
            loaded_model = create_moka_network_for_checkpoint(
                str(dense_checkpoint_path)
            )
            self.assertEqual(
                loaded_model.global_residual_block_interval,
                2,
            )
            _, manifest_path = export_model(
                dense_checkpoint_path,
                output_directory,
                8,
            )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(
                manifest["architecture"]["globalResidualBlockInterval"],
                2,
            )
            self.assertIn(
                "residual.1.global.hidden.weight",
                manifest["tensors"],
            )

    def test_global_value_initialization_preserves_deployed_outputs(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = (
                Path(temporary_directory) / "source.safetensors"
            )
            global_value_checkpoint_path = (
                Path(temporary_directory) / "global-value.safetensors"
            )
            source_model = MokaGlobalResidualNetwork()
            source_model.save_weights(str(checkpoint_path))
            global_value_model = MokaGlobalValueNetwork()
            global_value_model.load_weights(
                str(checkpoint_path),
                strict=False,
            )
            features = mx.random.normal(
                (
                    2,
                    BOARD_SIZE,
                    BOARD_SIZE,
                    INPUT_PLANE_COUNT,
                )
            )
            source_outputs = source_model(features)
            global_value_outputs = global_value_model(features)
            mx.eval(source_outputs, global_value_outputs)

            for source_output, global_value_output in zip(
                source_outputs,
                global_value_outputs,
                strict=True,
            ):
                np.testing.assert_array_equal(
                    np.asarray(source_output),
                    np.asarray(global_value_output),
                )

            global_value_model.save_weights(
                str(global_value_checkpoint_path)
            )
            self.assertTrue(
                checkpoint_uses_global_value_network(
                    str(global_value_checkpoint_path)
                )
            )
            self.assertIsInstance(
                create_moka_network_for_checkpoint(
                    str(global_value_checkpoint_path)
                ),
                MokaGlobalValueNetwork,
            )
            with self.assertRaisesRegex(
                ValueError,
                "not supported by the browser runtime",
            ):
                export_model(
                    global_value_checkpoint_path,
                    Path(temporary_directory) / "export",
                    8,
                )

    def test_global_score_checkpoint_preserves_deployed_outputs(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = (
                Path(temporary_directory) / "source.safetensors"
            )
            global_score_checkpoint_path = (
                Path(temporary_directory) / "global-score.safetensors"
            )
            source_model = MokaGlobalResidualNetwork()
            source_model.save_weights(str(checkpoint_path))
            global_score_model = MokaGlobalScoreNetwork()
            global_score_model.load_weights(
                str(checkpoint_path),
                strict=False,
            )
            features = mx.random.normal(
                (
                    2,
                    BOARD_SIZE,
                    BOARD_SIZE,
                    INPUT_PLANE_COUNT,
                )
            )
            source_outputs = source_model(features)
            global_score_outputs = global_score_model(features)
            mx.eval(source_outputs, global_score_outputs)

            for source_output, global_score_output in zip(
                source_outputs,
                global_score_outputs,
                strict=True,
            ):
                np.testing.assert_array_equal(
                    np.asarray(source_output),
                    np.asarray(global_score_output),
                )

            global_score_model.save_weights(
                str(global_score_checkpoint_path)
            )
            self.assertTrue(
                checkpoint_uses_global_score_network(
                    str(global_score_checkpoint_path)
                )
            )
            self.assertIsInstance(
                create_moka_network_for_checkpoint(
                    str(global_score_checkpoint_path)
                ),
                MokaGlobalScoreNetwork,
            )
            with self.assertRaisesRegex(
                ValueError,
                "not supported by the browser runtime",
            ):
                export_model(
                    global_score_checkpoint_path,
                    Path(temporary_directory) / "export",
                    8,
                )

    def test_gated_global_checkpoint_preserves_deployed_outputs(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = (
                Path(temporary_directory) / "source.safetensors"
            )
            gated_checkpoint_path = (
                Path(temporary_directory) / "gated.safetensors"
            )
            source_model = MokaGlobalResidualNetwork()
            source_model.save_weights(str(checkpoint_path))
            gated_model = MokaGatedGlobalResidualNetwork()
            gated_model.load_weights(
                str(checkpoint_path),
                strict=False,
            )
            features = mx.random.normal(
                (
                    2,
                    BOARD_SIZE,
                    BOARD_SIZE,
                    INPUT_PLANE_COUNT,
                )
            )
            source_outputs = source_model(features)
            gated_outputs = gated_model(features)
            mx.eval(source_outputs, gated_outputs)

            for source_output, gated_output in zip(
                source_outputs,
                gated_outputs,
                strict=True,
            ):
                np.testing.assert_array_equal(
                    np.asarray(source_output),
                    np.asarray(gated_output),
                )

            gated_model.save_weights(str(gated_checkpoint_path))
            self.assertTrue(
                checkpoint_uses_gated_global_residual_network(
                    str(gated_checkpoint_path)
                )
            )
            self.assertIsInstance(
                create_moka_network_for_checkpoint(
                    str(gated_checkpoint_path)
                ),
                MokaGatedGlobalResidualNetwork,
            )
            with self.assertRaisesRegex(
                ValueError,
                "not supported by the browser runtime",
            ):
                export_model(
                    gated_checkpoint_path,
                    Path(temporary_directory) / "export",
                    8,
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

    def test_materialized_checkpoint_can_merge_selected_prefixes(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory_path = Path(temporary_directory)
            checkpoint_path = directory_path / "source.safetensors"
            base_checkpoint_path = directory_path / "base.safetensors"
            output_path = directory_path / "merged.safetensors"
            source_parameters = {
                "value.weight": mx.array(
                    [[0.2, -0.4]],
                    dtype=mx.float32,
                ),
                "policy.weight": mx.array(
                    [[0.3, -0.7]],
                    dtype=mx.float32,
                ),
            }
            base_parameters = {
                "value.weight": mx.zeros((1, 2)),
                "policy.weight": mx.ones((1, 2)),
            }
            mx.save_safetensors(str(checkpoint_path), source_parameters)
            mx.save_safetensors(
                str(base_checkpoint_path),
                base_parameters,
            )

            materialize_int8_checkpoint(
                checkpoint_path,
                output_path,
                base_checkpoint_path,
                ["value."],
            )

            output_parameters = mx.load(str(output_path))
            expected_value_parameters = fake_quantize_int8_parameters(
                {"value": {"weight": source_parameters["value.weight"]}}
            )
            np.testing.assert_array_equal(
                np.asarray(output_parameters["value.weight"]),
                np.asarray(expected_value_parameters["value"]["weight"]),
            )
            np.testing.assert_array_equal(
                np.asarray(output_parameters["policy.weight"]),
                np.asarray(base_parameters["policy.weight"]),
            )

    def test_materialized_checkpoint_can_add_selected_prefixes(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory_path = Path(temporary_directory)
            checkpoint_path = directory_path / "source.safetensors"
            base_checkpoint_path = directory_path / "base.safetensors"
            output_path = directory_path / "merged.safetensors"
            source_parameters = {
                "adapter.weight": mx.array(
                    [[0.2, -0.4]],
                    dtype=mx.float32,
                ),
                "policy.weight": mx.zeros((1, 2)),
            }
            base_parameters = {
                "policy.weight": mx.ones((1, 2)),
            }
            mx.save_safetensors(str(checkpoint_path), source_parameters)
            mx.save_safetensors(
                str(base_checkpoint_path),
                base_parameters,
            )

            materialize_int8_checkpoint(
                checkpoint_path,
                output_path,
                base_checkpoint_path,
                ["adapter."],
            )

            output_parameters = mx.load(str(output_path))
            expected_adapter_parameters = fake_quantize_int8_parameters(
                {"adapter": {"weight": source_parameters["adapter.weight"]}}
            )
            np.testing.assert_array_equal(
                np.asarray(output_parameters["adapter.weight"]),
                np.asarray(
                    expected_adapter_parameters["adapter"]["weight"]
                ),
            )
            np.testing.assert_array_equal(
                np.asarray(output_parameters["policy.weight"]),
                np.asarray(base_parameters["policy.weight"]),
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
