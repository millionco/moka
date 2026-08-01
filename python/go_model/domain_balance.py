import argparse
from pathlib import Path

import numpy as np

from go_model.config import (
    SPLIT_BUCKET_COUNT,
    TEST_BUCKET_INDEX,
    VALIDATION_BUCKET_INDEX,
)
from go_model.split import create_game_split_buckets


def get_root_array_keys(
    archives: list[np.lib.npyio.NpzFile],
) -> list[str]:
    common_keys = set.intersection(
        *(set(archive.files) for archive in archives)
    )
    root_counts = [len(archive["features"]) for archive in archives]
    return sorted(
        key
        for key in common_keys
        if key not in {"game_ids", "sample_weights"}
        and all(
            archive[key].ndim > 0
            and len(archive[key]) == root_count
            for archive, root_count in zip(
                archives,
                root_counts,
                strict=True,
            )
        )
    )


def remap_game_ids(
    game_ids_by_domain: list[np.ndarray],
    game_pair_size: int,
) -> list[np.ndarray]:
    game_period = SPLIT_BUCKET_COUNT * game_pair_size
    next_game_offset = 0
    remapped_game_ids: list[np.ndarray] = []
    for game_ids in game_ids_by_domain:
        remapped_game_ids.append(game_ids + next_game_offset)
        required_game_count = int(np.max(game_ids)) + 1
        offset_period_count = (
            required_game_count + game_period - 1
        ) // game_period
        next_game_offset += offset_period_count * game_period
    return remapped_game_ids


def create_equal_domain_weights(
    game_ids_by_domain: list[np.ndarray],
    game_pair_size: int,
) -> list[np.ndarray]:
    split_buckets_by_domain = [
        create_game_split_buckets(game_ids, game_pair_size)
        for game_ids in game_ids_by_domain
    ]
    weights_by_domain = [
        np.zeros(len(game_ids), dtype=np.float32)
        for game_ids in game_ids_by_domain
    ]
    split_masks_by_domain = [
        [
            split_buckets == VALIDATION_BUCKET_INDEX,
            split_buckets == TEST_BUCKET_INDEX,
            (split_buckets != VALIDATION_BUCKET_INDEX)
            & (split_buckets != TEST_BUCKET_INDEX),
        ]
        for split_buckets in split_buckets_by_domain
    ]
    for split_index in range(3):
        split_counts = [
            int(np.sum(domain_masks[split_index]))
            for domain_masks in split_masks_by_domain
        ]
        if any(split_count == 0 for split_count in split_counts):
            raise ValueError("Every domain must cover every data split.")
        total_split_count = sum(split_counts)
        target_domain_weight = total_split_count / len(game_ids_by_domain)
        for domain_index, split_count in enumerate(split_counts):
            weights_by_domain[domain_index][
                split_masks_by_domain[domain_index][split_index]
            ] = target_domain_weight / split_count
    return weights_by_domain


def merge_equal_domain_datasets(
    dataset_paths: list[Path],
    output_path: Path,
    game_pair_size: int,
) -> None:
    if len(dataset_paths) < 2:
        raise ValueError("Domain balancing requires at least two datasets.")
    archives = [np.load(path) for path in dataset_paths]
    try:
        game_ids_by_domain = [
            archive["game_ids"].astype(np.int64) for archive in archives
        ]
        remapped_game_ids = remap_game_ids(
            game_ids_by_domain,
            game_pair_size,
        )
        weights_by_domain = create_equal_domain_weights(
            remapped_game_ids,
            game_pair_size,
        )
        root_array_keys = get_root_array_keys(archives)
        required_keys = {"features", "policies", "values"}
        missing_keys = required_keys - set(root_array_keys)
        if missing_keys:
            raise ValueError(
                "Datasets are missing shared root arrays: "
                + ", ".join(sorted(missing_keys))
            )
        output_values = {
            key: np.concatenate([archive[key] for archive in archives])
            for key in root_array_keys
        }
        output_values["game_ids"] = np.concatenate(remapped_game_ids)
        output_values["sample_weights"] = np.concatenate(
            weights_by_domain
        ).astype(np.float16)
        output_values["domain_ids"] = np.concatenate(
            [
                np.full(len(game_ids), domain_index, dtype=np.int16)
                for domain_index, game_ids in enumerate(game_ids_by_domain)
            ]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **output_values)
        print(
            f"saved {output_path} positions={len(output_values['features']):,} "
            f"domains={len(dataset_paths)} bytes={output_path.stat().st_size:,}"
        )
    finally:
        for archive in archives:
            archive.close()


def create_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--data",
        required=True,
        action="append",
        type=Path,
    )
    argument_parser.add_argument("--output", required=True, type=Path)
    argument_parser.add_argument("--game-pair-size", type=int, default=2)
    return argument_parser


def main() -> None:
    arguments = create_argument_parser().parse_args()
    merge_equal_domain_datasets(
        arguments.data,
        arguments.output,
        arguments.game_pair_size,
    )


if __name__ == "__main__":
    main()
