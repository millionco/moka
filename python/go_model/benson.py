from dataclasses import dataclass

import numpy as np

from go_model.board import get_adjacent_moves, get_group
from go_model.config import (
    BENSON_REQUIRED_VITAL_REGION_COUNT,
    BOARD_AREA,
    KOMI_POINTS,
)


@dataclass
class BensonRegion:
    bordering_group_keys: set[int]
    vital_group_keys: set[int]


def get_color_groups(
    board: np.ndarray,
    color: int,
) -> tuple[dict[int, int], dict[int, list[int]]]:
    group_key_by_move: dict[int, int] = {}
    groups: dict[int, list[int]] = {}
    visited_moves: set[int] = set()

    for move in range(BOARD_AREA):
        row, column = divmod(move, board.shape[1])
        if int(board[row, column]) != color or move in visited_moves:
            continue

        stones, _ = get_group(board, move)
        group_key = stones[0]
        groups[group_key] = stones

        for stone_move in stones:
            visited_moves.add(stone_move)
            group_key_by_move[stone_move] = group_key

    return group_key_by_move, groups


def get_non_color_regions(
    board: np.ndarray,
    color: int,
    group_key_by_move: dict[int, int],
) -> list[BensonRegion]:
    regions: list[BensonRegion] = []
    visited_moves: set[int] = set()

    for move in range(BOARD_AREA):
        row, column = divmod(move, board.shape[1])
        if int(board[row, column]) == color or move in visited_moves:
            continue

        pending_moves = [move]
        bordering_group_keys: set[int] = set()
        vital_group_keys: set[int] | None = None

        while pending_moves:
            region_move = pending_moves.pop()
            if region_move in visited_moves:
                continue

            visited_moves.add(region_move)
            region_row, region_column = divmod(
                region_move,
                board.shape[1],
            )
            adjacent_group_keys: set[int] = set()

            for adjacent_move in get_adjacent_moves(region_move):
                adjacent_row, adjacent_column = divmod(
                    adjacent_move,
                    board.shape[1],
                )
                if int(board[adjacent_row, adjacent_column]) == color:
                    group_key = group_key_by_move[adjacent_move]
                    adjacent_group_keys.add(group_key)
                    bordering_group_keys.add(group_key)
                elif adjacent_move not in visited_moves:
                    pending_moves.append(adjacent_move)

            if int(board[region_row, region_column]) == 0:
                if vital_group_keys is None:
                    vital_group_keys = adjacent_group_keys
                else:
                    vital_group_keys.intersection_update(
                        adjacent_group_keys,
                    )

        regions.append(
            BensonRegion(
                bordering_group_keys,
                vital_group_keys or set(),
            )
        )

    return regions


def get_benson_pass_alive_stones(
    board: np.ndarray,
    color: int,
) -> set[int]:
    group_key_by_move, groups = get_color_groups(board, color)
    regions = get_non_color_regions(board, color, group_key_by_move)
    pass_alive_group_keys = get_benson_pass_alive_group_keys(
        groups,
        regions,
    )
    return {
        stone_move
        for group_key in pass_alive_group_keys
        for stone_move in groups[group_key]
    }


def get_benson_pass_alive_group_keys(
    groups: dict[int, list[int]],
    regions: list[BensonRegion],
) -> set[int]:
    pass_alive_group_keys = set(groups)
    pass_alive_region_indexes = set(range(len(regions)))

    while True:
        removed_group_keys = {
            group_key
            for group_key in pass_alive_group_keys
            if sum(
                group_key in regions[region_index].vital_group_keys
                for region_index in pass_alive_region_indexes
            )
            < BENSON_REQUIRED_VITAL_REGION_COUNT
        }
        if not removed_group_keys:
            break

        pass_alive_group_keys.difference_update(removed_group_keys)
        pass_alive_region_indexes = {
            region_index
            for region_index in pass_alive_region_indexes
            if regions[region_index].bordering_group_keys.isdisjoint(
                removed_group_keys
            )
        }

    return pass_alive_group_keys


def calculate_benson_score_bounds(
    black_safe_point_count: int,
    white_safe_point_count: int,
) -> tuple[float, float]:
    return (
        2 * black_safe_point_count - BOARD_AREA - KOMI_POINTS,
        BOARD_AREA - 2 * white_safe_point_count - KOMI_POINTS,
    )


def get_benson_score_bounds(board: np.ndarray) -> tuple[float, float]:
    black_safe_stones = get_benson_pass_alive_stones(board, 1)
    white_safe_stones = get_benson_pass_alive_stones(board, -1)
    return calculate_benson_score_bounds(
        len(black_safe_stones),
        len(white_safe_stones),
    )


def get_benson_proven_winner(board: np.ndarray) -> int:
    black_score_lower_bound, black_score_upper_bound = (
        get_benson_score_bounds(board)
    )
    if black_score_lower_bound > 0:
        return 1
    if black_score_upper_bound <= 0:
        return -1
    return 0
