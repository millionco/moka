import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np

from go_model.chain_fate import ChainFateModel
from go_model.chain_fate_distill import (
    create_chain_fate_distillation_dataset,
)
from go_model.config import CHAIN_FATE_FEATURE_COUNT
from go_model.model import create_moka_network


class ChainFateDistillTest(unittest.TestCase):
    def test_distillation_preserves_games_and_value_signs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.npz"
            checkpoint_path = directory_path / "checkpoint.safetensors"
            chain_fate_path = directory_path / "chain-fate.npz"
            output_path = directory_path / "distilled.npz"
            model = create_moka_network(
                False,
                False,
                False,
                False,
                False,
            )
            model.save_weights(str(checkpoint_path))
            features = np.zeros((2, 9, 9, 12), dtype=np.float32)
            features[0, 0, 0, 0] = 1
            features[1, 8, 8, 1] = 1
            game_ids = np.asarray([10, 11], dtype=np.int32)
            np.savez_compressed(
                source_path,
                features=features,
                game_ids=game_ids,
            )
            chain_fate_model = ChainFateModel(
                feature_means=np.zeros(
                    CHAIN_FATE_FEATURE_COUNT,
                    dtype=np.float32,
                ),
                feature_standard_deviations=np.ones(
                    CHAIN_FATE_FEATURE_COUNT,
                    dtype=np.float32,
                ),
                hidden_weights=np.zeros(
                    (CHAIN_FATE_FEATURE_COUNT, 4),
                    dtype=np.float32,
                ),
                hidden_biases=np.zeros(4, dtype=np.float32),
                output_weights=np.zeros(4, dtype=np.float32),
                output_bias=np.zeros(1, dtype=np.float32),
                value_coefficient=1,
            )
            chain_fate_model.save(chain_fate_path)

            metrics = create_chain_fate_distillation_dataset(
                source_path,
                checkpoint_path,
                chain_fate_path,
                output_path,
                1,
                2,
            )

            with np.load(output_path) as output:
                np.testing.assert_array_equal(output["features"], features)
                np.testing.assert_array_equal(output["game_ids"], game_ids)
                _, raw_values = model(mx.array(features))
                mx.eval(raw_values)
                np.testing.assert_array_equal(
                    np.sign(output["values"]),
                    np.sign(np.asarray(raw_values)),
                )
            self.assertEqual(metrics["position_count"], 2)
            self.assertEqual(metrics["sign_change_count"], 0)


if __name__ == "__main__":
    unittest.main()
