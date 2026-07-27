from dataclasses import dataclass, field

import numpy as np

from go_model.config import BOARD_AREA, BOARD_SIZE, KOMI_POINTS, MAXIMUM_GAME_MOVE_COUNT


@dataclass
class GameState:
    board: np.ndarray = field(
        default_factory=lambda: np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    )
    consecutive_pass_count: int = 0
    ko_move: int = -1
    move_count: int = 0
    move_history: list[int] = field(default_factory=list)
    next_color: int = 1

    def copy(self) -> "GameState":
        return GameState(
            board=self.board.copy(),
            consecutive_pass_count=self.consecutive_pass_count,
            ko_move=self.ko_move,
            move_count=self.move_count,
            move_history=self.move_history.copy(),
            next_color=self.next_color,
        )


def get_adjacent_moves(move: int) -> list[int]:
    row, column = divmod(move, BOARD_SIZE)
    adjacent_moves: list[int] = []

    if row > 0:
        adjacent_moves.append(move - BOARD_SIZE)
    if row < BOARD_SIZE - 1:
        adjacent_moves.append(move + BOARD_SIZE)
    if column > 0:
        adjacent_moves.append(move - 1)
    if column < BOARD_SIZE - 1:
        adjacent_moves.append(move + 1)

    return adjacent_moves


def get_group(board: np.ndarray, starting_move: int) -> tuple[list[int], set[int]]:
    starting_row, starting_column = divmod(starting_move, BOARD_SIZE)
    color = int(board[starting_row, starting_column])

    if color == 0:
        return [], set()

    pending_moves = [starting_move]
    visited_moves: set[int] = set()
    liberties: set[int] = set()
    stones: list[int] = []

    while pending_moves:
        move = pending_moves.pop()

        if move in visited_moves:
            continue

        visited_moves.add(move)
        stones.append(move)

        for adjacent_move in get_adjacent_moves(move):
            adjacent_row, adjacent_column = divmod(adjacent_move, BOARD_SIZE)
            adjacent_color = int(board[adjacent_row, adjacent_column])

            if adjacent_color == 0:
                liberties.add(adjacent_move)
            elif adjacent_color == color:
                pending_moves.append(adjacent_move)

    return stones, liberties


def play_move(game_state: GameState, move: int) -> GameState | None:
    if move == BOARD_AREA:
        next_state = game_state.copy()
        next_state.consecutive_pass_count += 1
        next_state.ko_move = -1
        next_state.move_count += 1
        next_state.move_history.append(move)
        next_state.next_color *= -1
        return next_state

    row, column = divmod(move, BOARD_SIZE)

    if game_state.board[row, column] != 0 or game_state.ko_move == move:
        return None

    next_state = game_state.copy()
    next_state.board[row, column] = game_state.next_color
    captured_moves: list[int] = []

    for adjacent_move in get_adjacent_moves(move):
        adjacent_row, adjacent_column = divmod(adjacent_move, BOARD_SIZE)

        if next_state.board[adjacent_row, adjacent_column] != -game_state.next_color:
            continue

        opponent_stones, opponent_liberties = get_group(next_state.board, adjacent_move)

        if opponent_liberties:
            continue

        for captured_move in opponent_stones:
            captured_row, captured_column = divmod(captured_move, BOARD_SIZE)
            next_state.board[captured_row, captured_column] = 0
            captured_moves.append(captured_move)

    played_stones, played_liberties = get_group(next_state.board, move)

    if not played_liberties:
        return None

    next_state.consecutive_pass_count = 0
    next_state.ko_move = (
        captured_moves[0]
        if len(captured_moves) == 1 and len(played_stones) == 1 and len(played_liberties) == 1
        else -1
    )
    next_state.move_count += 1
    next_state.move_history.append(move)
    next_state.next_color *= -1
    return next_state


def get_legal_moves(game_state: GameState) -> list[int]:
    legal_moves = [
        move for move in range(BOARD_AREA) if play_move(game_state, move) is not None
    ]
    legal_moves.append(BOARD_AREA)
    return legal_moves


def is_game_over(game_state: GameState) -> bool:
    return (
        game_state.consecutive_pass_count >= 2
        or game_state.move_count >= MAXIMUM_GAME_MOVE_COUNT
    )


def get_area_score(game_state: GameState) -> float:
    black_score = float(np.count_nonzero(game_state.board == 1))
    white_score = float(np.count_nonzero(game_state.board == -1)) + KOMI_POINTS
    visited_moves: set[int] = set()

    for move in range(BOARD_AREA):
        row, column = divmod(move, BOARD_SIZE)

        if game_state.board[row, column] != 0 or move in visited_moves:
            continue

        pending_moves = [move]
        territory_moves: list[int] = []
        bordering_colors: set[int] = set()

        while pending_moves:
            territory_move = pending_moves.pop()

            if territory_move in visited_moves:
                continue

            visited_moves.add(territory_move)
            territory_moves.append(territory_move)

            for adjacent_move in get_adjacent_moves(territory_move):
                adjacent_row, adjacent_column = divmod(adjacent_move, BOARD_SIZE)
                adjacent_color = int(game_state.board[adjacent_row, adjacent_column])

                if adjacent_color == 0 and adjacent_move not in visited_moves:
                    pending_moves.append(adjacent_move)
                elif adjacent_color != 0:
                    bordering_colors.add(adjacent_color)

        if bordering_colors == {1}:
            black_score += len(territory_moves)
        elif bordering_colors == {-1}:
            white_score += len(territory_moves)

    return black_score - white_score
