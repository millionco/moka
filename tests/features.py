import unittest

import numpy as np

from go_model.board import GameState, play_move
from go_model.config import (
    BOARD_AREA,
    BOARD_SIZE,
    INPUT_PLANE_COUNT,
    TEACHER_GLOBAL_FEATURE_COUNT,
    TEACHER_SPATIAL_FEATURE_COUNT,
)
from go_model.features import encode_student_features, encode_teacher_features


class FeaturesTest(unittest.TestCase):
    def test_feature_shapes_and_perspective(self) -> None:
        game_state = play_move(GameState(), 0)
        self.assertIsNotNone(game_state)
        student_features = encode_student_features(game_state)
        teacher_spatial, teacher_global = encode_teacher_features(game_state)
        self.assertEqual(
            student_features.shape,
            (BOARD_SIZE, BOARD_SIZE, INPUT_PLANE_COUNT),
        )
        self.assertEqual(
            teacher_spatial.shape,
            (TEACHER_SPATIAL_FEATURE_COUNT, BOARD_SIZE, BOARD_SIZE),
        )
        self.assertEqual(teacher_global.shape, (TEACHER_GLOBAL_FEATURE_COUNT,))
        self.assertEqual(student_features[0, 0, 1], 1)
        self.assertGreater(student_features[0, 0, 11], 0)

    def test_pass_history_uses_dedicated_planes(self) -> None:
        game_state = play_move(GameState(), BOARD_AREA)
        self.assertIsNotNone(game_state)
        features = encode_student_features(game_state)
        self.assertEqual(np.count_nonzero(features[:, :, 7]), 0)
        self.assertEqual(np.count_nonzero(features[:, :, 8]), 0)
        self.assertTrue(np.all(features[:, :, 9] == 1))
        self.assertTrue(np.all(features[:, :, 10] == 0))

        game_state = play_move(game_state, BOARD_AREA)
        self.assertIsNotNone(game_state)
        features = encode_student_features(game_state)
        self.assertTrue(np.all(features[:, :, 9] == 1))
        self.assertTrue(np.all(features[:, :, 10] == 1))


if __name__ == "__main__":
    unittest.main()
