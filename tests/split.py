import unittest

import numpy as np

from go_model.split import create_game_split_buckets


class GameSplitTest(unittest.TestCase):
    def test_paired_games_share_buckets(self) -> None:
        game_ids = np.arange(20, dtype=np.int32)

        buckets = create_game_split_buckets(game_ids, 2)

        np.testing.assert_array_equal(buckets[::2], np.arange(10))
        np.testing.assert_array_equal(buckets[1::2], np.arange(10))

    def test_single_games_preserve_original_split(self) -> None:
        game_ids = np.arange(25, dtype=np.int32)

        buckets = create_game_split_buckets(game_ids, 1)

        np.testing.assert_array_equal(buckets, game_ids % 10)

    def test_rejects_non_positive_pair_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            create_game_split_buckets(np.asarray([0]), 0)


if __name__ == "__main__":
    unittest.main()
