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
        spatial_features, global_features = encode_teacher_features(game_state)
        policy, pass_policy, value = self.session.run(
            ["policy", "policy_pass", "value"],
            {
                "input_spatial": spatial_features.reshape(
                    1,
                    spatial_features.shape[0],
                    BOARD_SIZE,
                    BOARD_SIZE,
                ),
                "input_global": global_features.reshape(1, global_features.shape[0]),
            },
        )
        policy_logits = np.concatenate(
            [policy.reshape(BOARD_AREA), pass_policy.reshape(1)]
        ).astype(np.float32)
        legal_moves = get_legal_moves(game_state)
        masked_logits = np.full(POLICY_MOVE_COUNT, -1e9, dtype=np.float32)
        masked_logits[legal_moves] = policy_logits[legal_moves]
        maximum_logit = float(np.max(masked_logits[legal_moves]))
        policy_probabilities = np.zeros(POLICY_MOVE_COUNT, dtype=np.float32)
        legal_probabilities = np.exp(masked_logits[legal_moves] - maximum_logit)
        legal_probabilities /= np.sum(legal_probabilities)
        policy_probabilities[legal_moves] = legal_probabilities
        value_logits = value.reshape(-1)[:2]
        value_probabilities = np.exp(value_logits - np.max(value_logits))
        value_probabilities /= np.sum(value_probabilities)
        perspective_value = float(value_probabilities[0] - value_probabilities[1])
        return policy_probabilities, perspective_value

