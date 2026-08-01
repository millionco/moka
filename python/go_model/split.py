import numpy as np

from go_model.config import SPLIT_BUCKET_COUNT


def create_game_split_buckets(
    game_ids: np.ndarray,
    game_pair_size: int,
) -> np.ndarray:
    if game_pair_size <= 0:
        raise ValueError("Game pair size must be positive.")
    return game_ids // game_pair_size % SPLIT_BUCKET_COUNT
