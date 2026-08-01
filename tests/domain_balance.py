import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from go_model.domain_balance import merge_equal_domain_datasets
from go_model.split import create_game_split_buckets


class DomainBalanceTest(unittest.TestCase):
    def test_merge_preserves_splits_and_equalizes_domain_weight(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory_path = Path(temporary_directory)
            dataset_paths: list[Path] = []
            for domain_index, game_count in enumerate((20, 40)):
                dataset_path = directory_path / f"domain-{domain_index}.npz"
                position_count = game_count * 2
                game_ids = np.repeat(np.arange(game_count), 2)
                np.savez_compressed(
                    dataset_path,
                    features=np.full(
                        (position_count, 1),
                        domain_index,
                        dtype=np.float32,
                    ),
                    policies=np.full(
                        (position_count, 2),
                        0.5,
                        dtype=np.float32,
                    ),
                    values=np.zeros(position_count, dtype=np.float32),
                    game_ids=game_ids,
                    child_features=np.zeros(
                        (position_count + 1, 1),
                        dtype=np.float32,
                    ),
                )
                dataset_paths.append(dataset_path)
            output_path = directory_path / "balanced.npz"

            merge_equal_domain_datasets(dataset_paths, output_path, 2)

            with np.load(output_path) as dataset:
                self.assertNotIn("child_features", dataset.files)
                split_buckets = create_game_split_buckets(
                    dataset["game_ids"],
                    2,
                )
                domain_ids = dataset["domain_ids"]
                weights = dataset["sample_weights"].astype(np.float32)
                for selected_split in (
                    split_buckets == 0,
                    split_buckets == 1,
                    (split_buckets != 0) & (split_buckets != 1),
                ):
                    domain_weights = [
                        float(
                            np.sum(
                                weights[
                                    selected_split
                                    & (domain_ids == domain_index)
                                ]
                            )
                        )
                        for domain_index in range(2)
                    ]
                    self.assertAlmostEqual(
                        domain_weights[0],
                        domain_weights[1],
                        places=2,
                    )
                first_domain_maximum = int(
                    np.max(dataset["game_ids"][domain_ids == 0])
                )
                second_domain_minimum = int(
                    np.min(dataset["game_ids"][domain_ids == 1])
                )
                self.assertLess(
                    first_domain_maximum,
                    second_domain_minimum,
                )


if __name__ == "__main__":
    unittest.main()
