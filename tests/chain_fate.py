import tempfile
import unittest
from pathlib import Path

import numpy as np

from go_model.board import GameState
from go_model.chain_fate import (
    ChainFateModel,
    extract_chain_fate_examples,
    load_chain_fate_model,
    train_chain_fate_model,
)
from go_model.config import (
    CHAIN_FATE_FEATURE_COUNT,
    CHAIN_FATE_HIDDEN_COUNT,
)
from go_model.features import encode_moka_features


class ChainFateTest(unittest.TestCase):
    def test_extracts_symmetry_invariant_chain_features(self) -> None:
        board = np.zeros((9, 9), dtype=np.int8)
        board[0, 0] = 1
        board[0, 1] = 1
        board[4, 4] = -1
        game_state = GameState(
            board=board,
            move_count=3,
            move_history=[0, 40, 1],
            next_color=-1,
        )
        encoded_features = encode_moka_features(game_state)
        ownership = np.linspace(-1, 1, 81, dtype=np.float32).reshape(9, 9)
        examples = extract_chain_fate_examples(
            encoded_features[None, ...],
            ownership[None, ...],
        )
        rotated_examples = extract_chain_fate_examples(
            np.rot90(encoded_features, axes=(0, 1))[None, ...],
            np.rot90(ownership)[None, ...],
        )

        feature_order = np.lexsort(examples.features.T[::-1])
        rotated_feature_order = np.lexsort(
            rotated_examples.features.T[::-1]
        )
        np.testing.assert_allclose(
            examples.features[feature_order],
            rotated_examples.features[rotated_feature_order],
        )
        np.testing.assert_allclose(
            np.sort(examples.targets),
            np.sort(rotated_examples.targets),
        )

    def test_training_and_artifact_round_trip(self) -> None:
        random_generator = np.random.default_rng(3)
        features = random_generator.normal(
            size=(32, CHAIN_FATE_FEATURE_COUNT)
        ).astype(np.float32)
        targets = (features[:, 0] > 0).astype(np.float32)
        model = train_chain_fate_model(
            features,
            targets,
            random_seed=5,
            epoch_count=2,
            batch_size=8,
        )

        self.assertEqual(
            model.hidden_weights.shape,
            (CHAIN_FATE_FEATURE_COUNT, CHAIN_FATE_HIDDEN_COUNT),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_path = Path(temporary_directory) / "chain-fate.npz"
            model.save(artifact_path)
            loaded_model = load_chain_fate_model(artifact_path)

        np.testing.assert_allclose(
            loaded_model.predict_chain_survival(features),
            model.predict_chain_survival(features),
        )

    def test_value_correction_preserves_predicted_winner(self) -> None:
        model = ChainFateModel(
            feature_means=np.zeros(CHAIN_FATE_FEATURE_COUNT),
            feature_standard_deviations=np.ones(CHAIN_FATE_FEATURE_COUNT),
            hidden_weights=np.zeros(
                (CHAIN_FATE_FEATURE_COUNT, CHAIN_FATE_HIDDEN_COUNT)
            ),
            hidden_biases=np.zeros(CHAIN_FATE_HIDDEN_COUNT),
            output_weights=np.zeros(CHAIN_FATE_HIDDEN_COUNT),
            output_bias=np.zeros(1),
            value_coefficient=10,
        )
        values = np.asarray([0.1, -0.1], dtype=np.float32)
        corrected_values = model.correct_values(
            values,
            np.asarray([-1, 1], dtype=np.float32),
        )

        np.testing.assert_array_equal(
            np.sign(corrected_values.astype(np.float32)),
            np.sign(values),
        )

    def test_zero_value_coefficient_is_bit_exact(self) -> None:
        model = ChainFateModel(
            feature_means=np.zeros(CHAIN_FATE_FEATURE_COUNT),
            feature_standard_deviations=np.ones(CHAIN_FATE_FEATURE_COUNT),
            hidden_weights=np.zeros(
                (CHAIN_FATE_FEATURE_COUNT, CHAIN_FATE_HIDDEN_COUNT)
            ),
            hidden_biases=np.zeros(CHAIN_FATE_HIDDEN_COUNT),
            output_weights=np.zeros(CHAIN_FATE_HIDDEN_COUNT),
            output_bias=np.zeros(1),
        )
        values = np.asarray([-1, -0.25, 0, 0.25, 1], dtype=np.float32)

        corrected_values = model.correct_values(
            values,
            np.ones(len(values), dtype=np.float32),
        )

        np.testing.assert_array_equal(corrected_values, values)


if __name__ == "__main__":
    unittest.main()
