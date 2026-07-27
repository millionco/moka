import unittest

from go_model.collect import calculate_goldilocks_weight


class GoldilocksWeightTests(unittest.TestCase):
    def test_mid_difficulty_position_gets_more_weight(self) -> None:
        medium_difficulty_weight = calculate_goldilocks_weight(0.45)
        easy_position_weight = calculate_goldilocks_weight(0.95)
        overwhelming_position_weight = calculate_goldilocks_weight(0.01)

        self.assertGreater(medium_difficulty_weight, easy_position_weight)
        self.assertGreater(medium_difficulty_weight, overwhelming_position_weight)


if __name__ == "__main__":
    unittest.main()
