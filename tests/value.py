import unittest

import mlx.core as mx
import numpy as np

from go_model.config import POLICY_MOVE_COUNT
from go_model.model import MokaGlobalResidualNetwork
from go_model.value import (
    MokaShortValueTrainingNetwork,
    apply_random_value_symmetry,
    create_child_value_pairs,
    create_short_value_targets,
    create_value_targets,
)


class ChildValuePairTest(unittest.TestCase):
    def test_random_value_symmetry_preserves_feature_values(self) -> None:
        random_generator = np.random.default_rng(5)
        features = np.arange(
            2 * 9 * 9 * 12,
            dtype=np.float32,
        ).reshape(2, 9, 9, 12)

        transformed = apply_random_value_symmetry(
            features,
            random_generator,
        )

        for source, result in zip(features, transformed, strict=True):
            np.testing.assert_array_equal(
                np.sort(source.reshape(-1)),
                np.sort(result.reshape(-1)),
            )

    def test_outcome_targets_mix_teacher_and_terminal_values(self) -> None:
        dataset = {
            "values": np.asarray([1.1, -1.1], dtype=np.float32),
            "teacher_values": np.asarray([0.2, -0.4], dtype=np.float32),
        }

        targets = create_value_targets(dataset, "", 0.25, 0, 0)

        np.testing.assert_allclose(targets, [0.4, -0.55])

    def test_search_targets_ignore_outcome_blending(self) -> None:
        dataset = {
            "child_values": np.asarray([0.3, -0.2], dtype=np.float32),
            "teacher_values": np.asarray([1, 1], dtype=np.float32),
        }

        targets = create_value_targets(dataset, "child_", 0.5, 0, 0)

        np.testing.assert_allclose(targets, [0.3, -0.2])

    def test_short_value_targets_share_weight_with_outcomes(self) -> None:
        dataset = {
            "values": np.asarray([1], dtype=np.float32),
            "teacher_values": np.asarray([0.2], dtype=np.float32),
            "teacher_short_values": np.asarray([0.6], dtype=np.float32),
        }

        targets = create_value_targets(dataset, "", 0.25, 0.25, 0)

        np.testing.assert_allclose(targets, [0.5])

    def test_short_value_auxiliary_targets_require_teacher_values(self) -> None:
        dataset = {
            "teacher_short_values": np.asarray(
                [0.6, -0.4],
                dtype=np.float16,
            ),
        }

        targets = create_short_value_targets(dataset, "")

        np.testing.assert_allclose(targets, [0.6, -0.4], atol=0.001)
        with self.assertRaisesRegex(ValueError, "teacher short values"):
            create_short_value_targets({}, "")

    def test_short_value_head_initially_matches_long_value(self) -> None:
        model = MokaGlobalResidualNetwork()
        training_model = MokaShortValueTrainingNetwork(model)
        features = mx.zeros((2, 9, 9, 12), dtype=mx.float32)

        expected_policy, expected_value = model(features)
        policy, long_value, short_value = training_model(features)
        mx.eval(
            expected_policy,
            expected_value,
            policy,
            long_value,
            short_value,
        )

        np.testing.assert_array_equal(policy, expected_policy)
        np.testing.assert_array_equal(long_value, expected_value)
        np.testing.assert_array_equal(short_value, expected_value)

    def test_root_search_targets_use_visit_weighted_child_values(self) -> None:
        dataset = {
            "values": np.asarray([0.2, -0.4], dtype=np.float32),
            "search_q_values": np.asarray(
                [[0.75, -0.25, 1], [0.5, -0.5, 0]],
                dtype=np.float32,
            ),
            "search_q_weights": np.asarray(
                [[3, 1, 0], [0, 0, 0]],
                dtype=np.float32,
            ),
        }

        targets = create_value_targets(dataset, "", 0, 0, 1)

        np.testing.assert_allclose(targets, [0.5, -0.4])

    def test_pairs_accept_search_collector_q_values(self) -> None:
        dataset = {
            "child_values": np.asarray([-0.5, 0.25], dtype=np.float32),
            "child_weights": np.asarray([4, 2], dtype=np.float32),
            "child_game_ids": np.asarray([7, 7]),
            "child_root_indexes": np.asarray([0, 0]),
            "search_q_values": np.zeros(
                (1, POLICY_MOVE_COUNT),
                dtype=np.float32,
            ),
        }

        preferred_indexes, alternative_indexes, gaps, _, _ = (
            create_child_value_pairs(dataset)
        )

        np.testing.assert_array_equal(preferred_indexes, [0])
        np.testing.assert_array_equal(alternative_indexes, [1])
        np.testing.assert_allclose(gaps, [0.75])

    def test_pairs_rank_lowest_child_value_as_root_preference(self) -> None:
        q_values = np.zeros((2, POLICY_MOVE_COUNT), dtype=np.float32)
        q_values[0, :3] = [0.2, 0.5, -0.4]
        q_values[1, :2] = [0.3, -0.7]
        dataset = {
            "child_values": np.asarray(
                [0.4, -0.6, 0.1, -0.2, -0.25],
                dtype=np.float32,
            ),
            "child_weights": np.asarray(
                [4, 8, 2, 3, 6],
                dtype=np.float32,
            ),
            "child_game_ids": np.asarray([2, 2, 2, 3, 3]),
            "q_values": q_values,
        }

        (
            preferred_indexes,
            alternative_indexes,
            target_value_gaps,
            pair_weights,
            pair_game_ids,
        ) = create_child_value_pairs(dataset)

        np.testing.assert_array_equal(preferred_indexes, [1, 1])
        np.testing.assert_array_equal(alternative_indexes, [0, 2])
        np.testing.assert_allclose(target_value_gaps, [1.0, 0.7])
        np.testing.assert_allclose(
            pair_weights,
            [np.sqrt(32), np.sqrt(16)],
        )
        np.testing.assert_array_equal(pair_game_ids, [2, 2])


if __name__ == "__main__":
    unittest.main()
