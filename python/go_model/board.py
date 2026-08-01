from dataclasses import dataclass, field
from functools import cache

import numpy as np

from go_model.config import BOARD_AREA, BOARD_SIZE, KOMI_POINTS, MAXIMUM_GAME_MOVE_COUNT


@dataclass(frozen=True, slots=True)
class PositionHistory:
    position: bytes
    previous: "PositionHistory | None" = None
    position_hash: int = field(init=False)
    position_filter: int = field(init=False)

    def __post_init__(self) -> None:
        position_hash = hash(self.position)
        position_bit = 1 << (position_hash & 255)
        object.__setattr__(self, "position_hash", position_hash)
        object.__setattr__(
            self,
            "position_filter",
            position_bit
            | (
                self.previous.position_filter
                if self.previous is not None
                else 0
            ),
        )

    def contains(self, position: bytes) -> bool:
        position_hash = hash(position)

        if self.position_filter & (1 << (position_hash & 255)) == 0:
            return False

        history: PositionHistory | None = self

        while history is not None:
            if (
                history.position_hash == position_hash
                and history.position == position
            ):
                return True

            history = history.previous

        return False


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
    position_history: PositionHistory | None = None

    def __post_init__(self) -> None:
        if self.position_history is None:
            self.position_history = PositionHistory(self.board.tobytes())

    def copy(self) -> "GameState":
        return GameState(
            board=self.board.copy(),
            consecutive_pass_count=self.consecutive_pass_count,
            ko_move=self.ko_move,
            move_count=self.move_count,
            move_history=self.move_history.copy(),
            next_color=self.next_color,
            position_history=self.position_history,
        )


@cache
def get_adjacent_moves(move: int) -> tuple[int, ...]:
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

    return tuple(adjacent_moves)


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

    position = next_state.board.tobytes()

    if (
        game_state.position_history is not None
        and game_state.position_history.contains(position)
    ):
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
    next_state.position_history = PositionHistory(
        position,
        game_state.position_history,
    )
    return next_state


def get_board_group_data(
    board: np.ndarray,
) -> tuple[dict[int, int], dict[int, list[int]], dict[int, set[int]]]:
    flat_board = board.reshape(-1)
    group_key_by_move: dict[int, int] = {}
    groups_by_key: dict[int, list[int]] = {}
    liberties_by_key: dict[int, set[int]] = {}
    visited_moves: set[int] = set()

    for move in range(BOARD_AREA):
        color = int(flat_board[move])

        if color == 0 or move in visited_moves:
            continue

        pending_moves = [move]
        stones: list[int] = []
        liberties: set[int] = set()

        while pending_moves:
            group_move = pending_moves.pop()

            if group_move in visited_moves:
                continue

            visited_moves.add(group_move)
            stones.append(group_move)

            for adjacent_move in get_adjacent_moves(group_move):
                adjacent_color = int(flat_board[adjacent_move])

                if adjacent_color == 0:
                    liberties.add(adjacent_move)
                elif (
                    adjacent_color == color
                    and adjacent_move not in visited_moves
                ):
                    pending_moves.append(adjacent_move)

        group_key = stones[0]
        groups_by_key[group_key] = stones
        liberties_by_key[group_key] = liberties

        for stone in stones:
            group_key_by_move[stone] = group_key

    return group_key_by_move, groups_by_key, liberties_by_key


def get_legal_move_states(
    game_state: GameState,
) -> list[tuple[int, GameState]]:
    flat_board = game_state.board.reshape(-1)
    (
        group_key_by_move,
        groups_by_key,
        liberties_by_key,
    ) = get_board_group_data(game_state.board)
    legal_move_states: list[tuple[int, GameState]] = []

    for move in range(BOARD_AREA):
        if flat_board[move] != 0 or game_state.ko_move == move:
            continue

        adjacent_moves = get_adjacent_moves(move)
        adjacent_empty_moves = [
            adjacent_move
            for adjacent_move in adjacent_moves
            if flat_board[adjacent_move] == 0
        ]
        friendly_group_keys = {
            group_key_by_move[adjacent_move]
            for adjacent_move in adjacent_moves
            if flat_board[adjacent_move] == game_state.next_color
        }
        opponent_group_keys = {
            group_key_by_move[adjacent_move]
            for adjacent_move in adjacent_moves
            if flat_board[adjacent_move] == -game_state.next_color
        }
        captured_group_keys = {
            group_key
            for group_key in opponent_group_keys
            if liberties_by_key[group_key] == {move}
        }
        has_friendly_liberty = any(
            liberties_by_key[group_key] - {move}
            for group_key in friendly_group_keys
        )

        if (
            not captured_group_keys
            and not adjacent_empty_moves
            and not has_friendly_liberty
        ):
            continue

        next_state = game_state.copy()
        move_row, move_column = divmod(move, BOARD_SIZE)
        next_state.board[move_row, move_column] = game_state.next_color
        captured_moves = [
            captured_move
            for group_key in captured_group_keys
            for captured_move in groups_by_key[group_key]
        ]

        for captured_move in captured_moves:
            captured_row, captured_column = divmod(
                captured_move,
                BOARD_SIZE,
            )
            next_state.board[captured_row, captured_column] = 0

        position = next_state.board.tobytes()

        if (
            game_state.position_history is not None
            and game_state.position_history.contains(position)
        ):
            continue

        next_state.consecutive_pass_count = 0
        next_state.ko_move = -1

        if len(captured_moves) == 1 and not friendly_group_keys:
            played_liberties = sum(
                next_state.board[adjacent_row, adjacent_column] == 0
                for adjacent_row, adjacent_column in (
                    divmod(adjacent_move, BOARD_SIZE)
                    for adjacent_move in adjacent_moves
                )
            )

            if played_liberties == 1:
                next_state.ko_move = captured_moves[0]

        next_state.move_count += 1
        next_state.move_history.append(move)
        next_state.next_color *= -1
        next_state.position_history = PositionHistory(
            position,
            game_state.position_history,
        )
        legal_move_states.append((move, next_state))

    pass_state = play_move(game_state, BOARD_AREA)

    if pass_state is not None:
        legal_move_states.append((BOARD_AREA, pass_state))

    return legal_move_states


def get_legal_moves(game_state: GameState) -> list[int]:
    return [move for move, _ in get_legal_move_states(game_state)]


def is_game_over(game_state: GameState) -> bool:
    return (
        game_state.consecutive_pass_count >= 2
        or game_state.move_count >= MAXIMUM_GAME_MOVE_COUNT
    )


def remove_dead_stones(
    game_state: GameState,
    dead_moves: list[int] | set[int] | tuple[int, ...],
) -> GameState:
    next_state = game_state.copy()

    for dead_move in dead_moves:
        if dead_move < 0 or dead_move >= BOARD_AREA:
            raise ValueError(f"Dead move must be between 0 and {BOARD_AREA - 1}.")

        dead_row, dead_column = divmod(dead_move, BOARD_SIZE)
        next_state.board[dead_row, dead_column] = 0

    return next_state


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
