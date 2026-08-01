from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from go_model.board import GameState, get_adjacent_moves
from go_model.config import (
    BOARD_AREA,
    BOARD_SIZE,
    CHAIN_FATE_ADAM_BETA_ONE,
    CHAIN_FATE_ADAM_BETA_TWO,
    CHAIN_FATE_ADAM_EPSILON,
    CHAIN_FATE_BATCH_SIZE,
    CHAIN_FATE_CORRECTION_LIMIT,
    CHAIN_FATE_EPOCH_COUNT,
    CHAIN_FATE_FEATURE_COUNT,
    CHAIN_FATE_HIDDEN_COUNT,
    CHAIN_FATE_INITIAL_WEIGHT_SCALE,
    CHAIN_FATE_LEARNING_RATE,
    CHAIN_FATE_LOGIT_LIMIT,
    CHAIN_FATE_MAXIMUM_ADJACENT_ENEMY_GROUP_COUNT,
    CHAIN_FATE_MAXIMUM_EYE_COUNT,
    CHAIN_FATE_MAXIMUM_LIBERTY_COUNT,
    CHAIN_FATE_MAXIMUM_LIBERTY_RATIO,
    CHAIN_FATE_MAXIMUM_SHARED_ENEMY_LIBERTY_COUNT,
    CHAIN_FATE_RIDGE_WEIGHT,
    CHAIN_FATE_VALUE_LIMIT,
    DEFAULT_RANDOM_SEED,
)
from go_model.features import encode_moka_features


@dataclass
class ChainFateExamples:
    features: np.ndarray
    targets: np.ndarray
    position_indexes: np.ndarray
    colors: np.ndarray
    stone_counts: np.ndarray
    position_count: int


@dataclass
class ChainFateModel:
    feature_means: np.ndarray
    feature_standard_deviations: np.ndarray
    hidden_weights: np.ndarray
    hidden_biases: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray
    value_coefficient: float = 0

    def predict_chain_survival(self, features: np.ndarray) -> np.ndarray:
        normalized_features = (
            features - self.feature_means
        ) / self.feature_standard_deviations
        hidden_values = np.tanh(
            normalized_features @ self.hidden_weights
            + self.hidden_biases
        )
        logits = hidden_values @ self.output_weights + self.output_bias[0]
        return 1 / (
            1 + np.exp(-np.clip(logits, -CHAIN_FATE_LOGIT_LIMIT, CHAIN_FATE_LOGIT_LIMIT))
        )

    def get_position_signals(
        self,
        examples: ChainFateExamples,
    ) -> np.ndarray:
        chain_survival = self.predict_chain_survival(examples.features)
        signals = np.zeros(examples.position_count, dtype=np.float64)
        np.add.at(
            signals,
            examples.position_indexes,
            examples.colors
            * examples.stone_counts
            * (2 * chain_survival - 1)
            / BOARD_AREA,
        )
        return signals

    def correct_values(
        self,
        values: np.ndarray,
        position_signals: np.ndarray,
    ) -> np.ndarray:
        if self.value_coefficient == 0:
            return values.copy()
        value_logits = np.arctanh(
            np.clip(values, -CHAIN_FATE_VALUE_LIMIT, CHAIN_FATE_VALUE_LIMIT)
        )
        corrections = np.clip(
            self.value_coefficient * position_signals,
            -CHAIN_FATE_CORRECTION_LIMIT,
            CHAIN_FATE_CORRECTION_LIMIT,
        )
        corrected_logits = value_logits + corrections
        minimum_signed_logit = np.finfo(values.dtype).tiny
        corrected_logits = np.where(
            value_logits > 0,
            np.maximum(corrected_logits, minimum_signed_logit),
            np.where(
                value_logits < 0,
                np.minimum(corrected_logits, -minimum_signed_logit),
                0,
            ),
        )
        return np.tanh(corrected_logits)

    def evaluate_game_state(self, game_state: GameState, value: float) -> float:
        encoded_features = encode_moka_features(game_state)[None, ...]
        return float(
            self.correct_encoded_values(
                encoded_features,
                np.asarray([value], dtype=np.float32),
            )[0]
        )

    def correct_encoded_values(
        self,
        encoded_positions: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        examples = extract_chain_fate_examples(encoded_positions)
        return self.correct_values(
            values,
            self.get_position_signals(examples),
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            feature_means=self.feature_means,
            feature_standard_deviations=self.feature_standard_deviations,
            hidden_weights=self.hidden_weights,
            hidden_biases=self.hidden_biases,
            output_weights=self.output_weights,
            output_bias=self.output_bias,
            value_coefficient=np.asarray(
                [self.value_coefficient],
                dtype=np.float32,
            ),
        )


def load_chain_fate_model(path: Path) -> ChainFateModel:
    with np.load(path) as artifact:
        return ChainFateModel(
            feature_means=artifact["feature_means"].astype(np.float32),
            feature_standard_deviations=artifact[
                "feature_standard_deviations"
            ].astype(np.float32),
            hidden_weights=artifact["hidden_weights"].astype(np.float32),
            hidden_biases=artifact["hidden_biases"].astype(np.float32),
            output_weights=artifact["output_weights"].astype(np.float32),
            output_bias=artifact["output_bias"].astype(np.float32),
            value_coefficient=float(artifact["value_coefficient"][0]),
        )


def get_chain_feature_board(encoded_features: np.ndarray) -> np.ndarray:
    return (
        (encoded_features[:, :, 0] > 0.5).astype(np.int8)
        - (encoded_features[:, :, 1] > 0.5).astype(np.int8)
    )


def get_chain_groups(
    board: np.ndarray,
) -> tuple[
    dict[int, dict[int, int]],
    dict[int, dict[int, list[int]]],
    dict[int, dict[int, set[int]]],
]:
    flat_board = board.reshape(-1)
    group_keys_by_color: dict[int, dict[int, int]] = {1: {}, -1: {}}
    groups_by_color: dict[int, dict[int, list[int]]] = {1: {}, -1: {}}
    liberties_by_color: dict[int, dict[int, set[int]]] = {1: {}, -1: {}}
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
        groups_by_color[color][group_key] = stones
        liberties_by_color[color][group_key] = liberties
        for stone in stones:
            group_keys_by_color[color][stone] = group_key

    return group_keys_by_color, groups_by_color, liberties_by_color


def get_chain_features(
    encoded_features: np.ndarray,
) -> tuple[list[list[float]], list[int], list[int], list[list[int]]]:
    board = get_chain_feature_board(encoded_features)
    occupied_fraction = float(np.count_nonzero(board)) / BOARD_AREA
    last_moves = set(
        np.flatnonzero(encoded_features[:, :, 7].reshape(-1) > 0.5)
    )
    (
        group_keys_by_color,
        groups_by_color,
        liberties_by_color,
    ) = get_chain_groups(board)
    chain_features: list[list[float]] = []
    colors: list[int] = []
    stone_counts: list[int] = []
    chain_stones: list[list[int]] = []
    visited_moves: set[int] = set()

    for move in range(BOARD_AREA):
        row, column = divmod(move, BOARD_SIZE)
        color = int(board[row, column])
        if color == 0 or move in visited_moves:
            continue

        group_key = group_keys_by_color[color][move]
        stones = groups_by_color[color][group_key]
        liberties = liberties_by_color[color][group_key]
        visited_moves.update(stones)
        enemy_group_keys: set[int] = set()
        did_last_move_touch_chain = False

        for stone in stones:
            did_last_move_touch_chain = (
                did_last_move_touch_chain or stone in last_moves
            )
            for adjacent_move in get_adjacent_moves(stone):
                did_last_move_touch_chain = (
                    did_last_move_touch_chain
                    or adjacent_move in last_moves
                )
                adjacent_row, adjacent_column = divmod(
                    adjacent_move,
                    BOARD_SIZE,
                )
                if int(board[adjacent_row, adjacent_column]) == -color:
                    enemy_group_keys.add(
                        group_keys_by_color[-color][adjacent_move]
                    )

        enemy_liberty_counts: list[int] = []
        capturable_enemy_stone_count = 0
        shared_enemy_liberties: set[int] = set()
        for enemy_group_key in enemy_group_keys:
            enemy_stones = groups_by_color[-color][enemy_group_key]
            enemy_liberties = liberties_by_color[-color][enemy_group_key]
            enemy_liberty_counts.append(len(enemy_liberties))
            shared_enemy_liberties.update(liberties & enemy_liberties)
            if len(enemy_liberties) == 1:
                capturable_enemy_stone_count += len(enemy_stones)

        second_order_liberties = {
            adjacent_move
            for liberty in liberties
            for adjacent_move in get_adjacent_moves(liberty)
            if int(board[divmod(adjacent_move, BOARD_SIZE)]) == 0
            and adjacent_move not in liberties
        }
        eye_liberties: set[int] = set()
        connectable_friendly_group_keys: set[int] = set()
        for liberty in liberties:
            adjacent_colors: set[int] = set()
            for adjacent_move in get_adjacent_moves(liberty):
                adjacent_row, adjacent_column = divmod(
                    adjacent_move,
                    BOARD_SIZE,
                )
                adjacent_color = int(board[adjacent_row, adjacent_column])
                adjacent_colors.add(adjacent_color)
                if adjacent_color == color:
                    adjacent_group_key = group_keys_by_color[color][
                        adjacent_move
                    ]
                    if adjacent_group_key != group_key:
                        connectable_friendly_group_keys.add(
                            adjacent_group_key
                        )
            if adjacent_colors == {color}:
                eye_liberties.add(liberty)
        edge_stone_count = sum(
            (
                stone_row in (0, BOARD_SIZE - 1)
                or stone_column in (0, BOARD_SIZE - 1)
            )
            for stone_row, stone_column in (
                divmod(stone, BOARD_SIZE) for stone in stones
            )
        )
        liberty_count = len(liberties)
        chain_features.append(
            [
                float(color),
                len(stones) / BOARD_AREA,
                min(
                    liberty_count,
                    CHAIN_FATE_MAXIMUM_LIBERTY_COUNT,
                )
                / CHAIN_FATE_MAXIMUM_LIBERTY_COUNT,
                min(
                    liberty_count / len(stones),
                    CHAIN_FATE_MAXIMUM_LIBERTY_RATIO,
                )
                / CHAIN_FATE_MAXIMUM_LIBERTY_RATIO,
                float(liberty_count == 1),
                float(liberty_count == 2),
                min(
                    len(enemy_group_keys),
                    CHAIN_FATE_MAXIMUM_ADJACENT_ENEMY_GROUP_COUNT,
                )
                / CHAIN_FATE_MAXIMUM_ADJACENT_ENEMY_GROUP_COUNT,
                (
                    min(enemy_liberty_counts)
                    if enemy_liberty_counts
                    else CHAIN_FATE_MAXIMUM_LIBERTY_COUNT
                )
                / CHAIN_FATE_MAXIMUM_LIBERTY_COUNT,
                capturable_enemy_stone_count / BOARD_AREA,
                min(
                    len(shared_enemy_liberties),
                    CHAIN_FATE_MAXIMUM_SHARED_ENEMY_LIBERTY_COUNT,
                )
                / CHAIN_FATE_MAXIMUM_SHARED_ENEMY_LIBERTY_COUNT,
                len(second_order_liberties) / BOARD_AREA,
                edge_stone_count / len(stones),
                float(did_last_move_touch_chain),
                occupied_fraction,
                min(len(eye_liberties), CHAIN_FATE_MAXIMUM_EYE_COUNT)
                / CHAIN_FATE_MAXIMUM_EYE_COUNT,
                min(
                    len(connectable_friendly_group_keys),
                    CHAIN_FATE_MAXIMUM_ADJACENT_ENEMY_GROUP_COUNT,
                )
                / CHAIN_FATE_MAXIMUM_ADJACENT_ENEMY_GROUP_COUNT,
            ]
        )
        colors.append(color)
        stone_counts.append(len(stones))
        chain_stones.append(stones)

    return chain_features, colors, stone_counts, chain_stones


def extract_chain_fate_examples(
    encoded_positions: np.ndarray,
    ownerships: np.ndarray | None = None,
) -> ChainFateExamples:
    all_features: list[list[float]] = []
    all_targets: list[float] = []
    position_indexes: list[int] = []
    colors: list[int] = []
    stone_counts: list[int] = []

    for position_index, encoded_features in enumerate(encoded_positions):
        (
            position_features,
            position_colors,
            position_stone_counts,
            position_chain_stones,
        ) = get_chain_features(encoded_features)
        all_features.extend(position_features)
        colors.extend(position_colors)
        stone_counts.extend(position_stone_counts)
        position_indexes.extend(
            [position_index] * len(position_features)
        )

        if ownerships is None:
            continue

        ownership = ownerships[position_index].reshape(-1)
        all_targets.extend(
            float(
                np.mean(
                    (
                        1
                        + color
                        * ownership[np.asarray(stones, dtype=np.int64)]
                    )
                    / 2
                )
            )
            for color, stones in zip(
                position_colors,
                position_chain_stones,
                strict=True,
            )
        )

    features = np.asarray(all_features, dtype=np.float32).reshape(
        -1,
        CHAIN_FATE_FEATURE_COUNT,
    )
    targets = np.asarray(all_targets, dtype=np.float32)
    return ChainFateExamples(
        features=features,
        targets=targets,
        position_indexes=np.asarray(position_indexes, dtype=np.int64),
        colors=np.asarray(colors, dtype=np.float32),
        stone_counts=np.asarray(stone_counts, dtype=np.float32),
        position_count=len(encoded_positions),
    )


def train_chain_fate_model(
    features: np.ndarray,
    targets: np.ndarray,
    random_seed: int = DEFAULT_RANDOM_SEED,
    epoch_count: int = CHAIN_FATE_EPOCH_COUNT,
    batch_size: int = CHAIN_FATE_BATCH_SIZE,
    learning_rate: float = CHAIN_FATE_LEARNING_RATE,
) -> ChainFateModel:
    random_generator = np.random.default_rng(random_seed)
    feature_means = np.mean(features, axis=0)
    feature_standard_deviations = np.std(features, axis=0)
    feature_standard_deviations[feature_standard_deviations < 1e-6] = 1
    normalized_features = (
        features - feature_means
    ) / feature_standard_deviations
    hidden_weights = random_generator.normal(
        0,
        CHAIN_FATE_INITIAL_WEIGHT_SCALE,
        (CHAIN_FATE_FEATURE_COUNT, CHAIN_FATE_HIDDEN_COUNT),
    )
    hidden_biases = np.zeros(CHAIN_FATE_HIDDEN_COUNT)
    output_weights = random_generator.normal(
        0,
        CHAIN_FATE_INITIAL_WEIGHT_SCALE,
        CHAIN_FATE_HIDDEN_COUNT,
    )
    output_bias = np.zeros(1)
    parameters = [
        hidden_weights,
        hidden_biases,
        output_weights,
        output_bias,
    ]
    first_moments = [np.zeros_like(parameter) for parameter in parameters]
    second_moments = [np.zeros_like(parameter) for parameter in parameters]
    step_count = 0

    for _ in range(epoch_count):
        order = random_generator.permutation(len(normalized_features))
        for start_index in range(0, len(order), batch_size):
            batch_indexes = order[start_index : start_index + batch_size]
            feature_batch = normalized_features[batch_indexes]
            target_batch = targets[batch_indexes]
            hidden_values = np.tanh(
                feature_batch @ hidden_weights + hidden_biases
            )
            logits = hidden_values @ output_weights + output_bias[0]
            predictions = 1 / (
                1
                + np.exp(
                    -np.clip(
                        logits,
                        -CHAIN_FATE_LOGIT_LIMIT,
                        CHAIN_FATE_LOGIT_LIMIT,
                    )
                )
            )
            logit_gradients = (
                predictions - target_batch
            ) / len(batch_indexes)
            hidden_gradients = (
                logit_gradients[:, None]
                * output_weights[None, :]
                * (1 - hidden_values * hidden_values)
            )
            gradients = [
                feature_batch.T @ hidden_gradients,
                np.sum(hidden_gradients, axis=0),
                hidden_values.T @ logit_gradients,
                np.asarray([np.sum(logit_gradients)]),
            ]
            step_count += 1

            for parameter_index, (parameter, gradient) in enumerate(
                zip(parameters, gradients, strict=True)
            ):
                first_moments[parameter_index] = (
                    CHAIN_FATE_ADAM_BETA_ONE
                    * first_moments[parameter_index]
                    + (1 - CHAIN_FATE_ADAM_BETA_ONE) * gradient
                )
                second_moments[parameter_index] = (
                    CHAIN_FATE_ADAM_BETA_TWO
                    * second_moments[parameter_index]
                    + (1 - CHAIN_FATE_ADAM_BETA_TWO)
                    * gradient
                    * gradient
                )
                corrected_first_moment = first_moments[parameter_index] / (
                    1 - CHAIN_FATE_ADAM_BETA_ONE**step_count
                )
                corrected_second_moment = second_moments[parameter_index] / (
                    1 - CHAIN_FATE_ADAM_BETA_TWO**step_count
                )
                parameter -= (
                    learning_rate
                    * corrected_first_moment
                    / (
                        np.sqrt(corrected_second_moment)
                        + CHAIN_FATE_ADAM_EPSILON
                    )
                )

    return ChainFateModel(
        feature_means=feature_means.astype(np.float32),
        feature_standard_deviations=(
            feature_standard_deviations.astype(np.float32)
        ),
        hidden_weights=hidden_weights.astype(np.float32),
        hidden_biases=hidden_biases.astype(np.float32),
        output_weights=output_weights.astype(np.float32),
        output_bias=output_bias.astype(np.float32),
    )


def fit_chain_fate_value_coefficient(
    model: ChainFateModel,
    examples: ChainFateExamples,
    moka_values: np.ndarray,
    teacher_values: np.ndarray,
) -> ChainFateModel:
    position_signals = model.get_position_signals(examples)
    return fit_chain_fate_value_coefficient_from_signals(
        model,
        position_signals,
        moka_values,
        teacher_values,
    )


def fit_chain_fate_value_coefficient_from_signals(
    model: ChainFateModel,
    position_signals: np.ndarray,
    moka_values: np.ndarray,
    teacher_values: np.ndarray,
) -> ChainFateModel:
    target_corrections = np.arctanh(
        np.clip(
            teacher_values,
            -CHAIN_FATE_VALUE_LIMIT,
            CHAIN_FATE_VALUE_LIMIT,
        )
    ) - np.arctanh(
        np.clip(
            moka_values,
            -CHAIN_FATE_VALUE_LIMIT,
            CHAIN_FATE_VALUE_LIMIT,
        )
    )
    value_coefficient = float(
        position_signals @ target_corrections
        / (
            position_signals @ position_signals
            + CHAIN_FATE_RIDGE_WEIGHT
        )
    )
    return replace(model, value_coefficient=value_coefficient)
