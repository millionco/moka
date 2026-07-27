from pathlib import Path

import numpy as np
import onnxruntime as ort

from go_model.board import GameState, get_legal_moves
from go_model.config import BOARD_AREA, BOARD_SIZE, POLICY_MOVE_COUNT
from go_model.features import encode_teacher_features


class KataGoTeacher:
    def __init__(self, model_path: Path) -> None:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path,
            session_options,
            providers=["CPUExecutionProvider"],
        )

    def evaluate(self, game_state: GameState) -> tuple[np.ndarray, float]:
        return self.evaluate_batch([game_state])[0]

    def evaluate_batch(
        self,
        game_states: list[GameState],
    ) -> list[tuple[np.ndarray, float]]:
        feature_pairs = [
            encode_teacher_features(game_state) for game_state in game_states
        ]
        spatial_features = np.stack(
            [feature_pair[0] for feature_pair in feature_pairs]
        )
        global_features = np.stack(
            [feature_pair[1] for feature_pair in feature_pairs]
        )
        policy, pass_policy, value = self.session.run(
            ["policy", "policy_pass", "value"],
            {
                "input_spatial": spatial_features,
                "input_global": global_features,
            },
        )
        evaluations: list[tuple[np.ndarray, float]] = []

        for game_index, game_state in enumerate(game_states):
            policy_logits = np.concatenate(
                [
                    policy[game_index].reshape(BOARD_AREA),
                    pass_policy[game_index].reshape(1),
                ]
            ).astype(np.float32)
            legal_moves = get_legal_moves(game_state)
            maximum_logit = float(np.max(policy_logits[legal_moves]))
            legal_probabilities = np.exp(
                policy_logits[legal_moves] - maximum_logit
            )
            legal_probabilities /= np.sum(legal_probabilities)
            policy_probabilities = np.zeros(POLICY_MOVE_COUNT, dtype=np.float32)
            policy_probabilities[legal_moves] = legal_probabilities
            value_logits = value[game_index].reshape(-1)[:2]
            value_probabilities = np.exp(value_logits - np.max(value_logits))
            value_probabilities /= np.sum(value_probabilities)
            perspective_value = float(
                value_probabilities[0] - value_probabilities[1]
            )
            evaluations.append((policy_probabilities, perspective_value))

        return evaluations
