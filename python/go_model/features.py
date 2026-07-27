import numpy as np

from go_model.board import GameState, get_group
from go_model.config import (
    BOARD_AREA,
    BOARD_SIZE,
    INPUT_PLANE_COUNT,
    KOMI_NORMALIZATION_POINTS,
    KOMI_POINTS,
    TEACHER_GLOBAL_FEATURE_COUNT,
    TEACHER_SPATIAL_FEATURE_COUNT,
)


def encode_student_features(game_state: GameState) -> np.ndarray:
    features = np.zeros(
        (BOARD_SIZE, BOARD_SIZE, INPUT_PLANE_COUNT),
        dtype=np.float32,
    )
    visited_moves: set[int] = set()

    for move in range(BOARD_AREA):
        row, column = divmod(move, BOARD_SIZE)
        color = int(game_state.board[row, column])

        if color == 0:
            continue

        is_current_player = color == game_state.next_color
        features[row, column, 0 if is_current_player else 1] = 1

        if move in visited_moves:
            continue

        stones, liberties = get_group(game_state.board, move)
        visited_moves.update(stones)
        liberty_count = len(liberties)

        if liberty_count not in (1, 2):
            continue

        feature_index = (
            2
            if is_current_player and liberty_count == 1
            else 3
            if liberty_count == 1
            else 4
            if is_current_player
            else 5
        )

        for stone_move in stones:
            stone_row, stone_column = divmod(stone_move, BOARD_SIZE)
            features[stone_row, stone_column, feature_index] = 1

    if game_state.ko_move >= 0:
        ko_row, ko_column = divmod(game_state.ko_move, BOARD_SIZE)
        features[ko_row, ko_column, 6] = 1

    for history_offset, feature_index in ((1, 7), (2, 8)):
        if len(game_state.move_history) < history_offset:
            continue

        history_move = game_state.move_history[-history_offset]

        if history_move < BOARD_AREA:
            history_row, history_column = divmod(history_move, BOARD_SIZE)
            features[history_row, history_column, feature_index] = 1
        else:
            features[:, :, 8 + history_offset] = 1

    perspective_komi = -KOMI_POINTS * game_state.next_color
    features[:, :, 11] = perspective_komi / KOMI_NORMALIZATION_POINTS
    return features


def encode_teacher_features(game_state: GameState) -> tuple[np.ndarray, np.ndarray]:
    spatial_features = np.zeros(
        (TEACHER_SPATIAL_FEATURE_COUNT, BOARD_SIZE, BOARD_SIZE),
        dtype=np.float32,
    )
    global_features = np.zeros(TEACHER_GLOBAL_FEATURE_COUNT, dtype=np.float32)
    spatial_features[0, :, :] = 1
    visited_moves: set[int] = set()

    for move in range(BOARD_AREA):
        row, column = divmod(move, BOARD_SIZE)
        color = int(game_state.board[row, column])

        if color == 0:
            continue

        spatial_features[1 if color == game_state.next_color else 2, row, column] = 1

        if move in visited_moves:
            continue

        stones, liberties = get_group(game_state.board, move)
        visited_moves.update(stones)
        liberty_count = len(liberties)

        if liberty_count not in (1, 2, 3):
            continue

        for stone_move in stones:
            stone_row, stone_column = divmod(stone_move, BOARD_SIZE)
            spatial_features[2 + liberty_count, stone_row, stone_column] = 1

    if game_state.ko_move >= 0:
        ko_row, ko_column = divmod(game_state.ko_move, BOARD_SIZE)
        spatial_features[6, ko_row, ko_column] = 1

    for recent_move_index, history_move in enumerate(
        reversed(game_state.move_history[-5:])
    ):
        if history_move == BOARD_AREA:
            global_features[recent_move_index] = 1
        else:
            history_row, history_column = divmod(history_move, BOARD_SIZE)
            spatial_features[9 + recent_move_index, history_row, history_column] = 1

    perspective_komi = -KOMI_POINTS * game_state.next_color
    global_features[5] = perspective_komi / KOMI_NORMALIZATION_POINTS
    global_features[6] = 1
    global_features[7] = 0.5
    return spatial_features, global_features
