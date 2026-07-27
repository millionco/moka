import sys
from pathlib import Path

import numpy as np
import torch

from go_model.board import GameState, get_legal_moves
from go_model.config import BOARD_AREA, BOARD_SIZE, KOMI_POINTS, POLICY_MOVE_COUNT


class StrongKataGoTeacher:
    def __init__(self, checkpoint_path: Path, source_path: Path) -> None:
        sys.path.insert(0, str((source_path / "python").resolve()))

        from katago.game.board import Board
        from katago.game.features import Features
        from katago.game.gamestate import GameState as KataGoGameState
        from katago.train.load_model import load_model

        self.board_class = Board
        self.features_class = Features
        self.game_state_class = KataGoGameState
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model, averaged_model, _ = load_model(
            str(checkpoint_path),
            True,
            device=self.device,
            pos_len=BOARD_SIZE,
        )
        self.model = averaged_model or model
        self.model.eval()
        self.features = Features(self.model.config, BOARD_SIZE)

    def create_teacher_game_state(self, game_state: GameState):
        rules = self.game_state_class.RULES_TT.copy()
        rules["whiteKomi"] = KOMI_POINTS
        teacher_game_state = self.game_state_class(BOARD_SIZE, rules)
        color = self.board_class.BLACK

        for move in game_state.move_history:
            location = (
                self.board_class.PASS_LOC
                if move == BOARD_AREA
                else teacher_game_state.board.loc(move % BOARD_SIZE, move // BOARD_SIZE)
            )
            teacher_game_state.play(color, location)
            color = self.board_class.get_opp(color)

        return teacher_game_state

    def evaluate(self, game_state: GameState) -> tuple[np.ndarray, float]:
        return self.evaluate_batch([game_state])[0]

    def evaluate_batch(
        self,
        game_states: list[GameState],
    ) -> list[tuple[np.ndarray, float]]:
        feature_pairs = [
            self.create_teacher_game_state(game_state).get_input_features(self.features)
            for game_state in game_states
        ]
        spatial_features = np.concatenate(
            [feature_pair[0] for feature_pair in feature_pairs],
            axis=0,
        )
        global_features = np.concatenate(
            [feature_pair[1] for feature_pair in feature_pairs],
            axis=0,
        )

        with torch.inference_mode():
            raw_outputs = self.model(
                torch.tensor(
                    spatial_features,
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.tensor(
                    global_features,
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
            outputs = self.model.postprocess_output(raw_outputs)[0]
            policy_logits = outputs[0][:, 0].cpu().numpy()
            value_probabilities = torch.softmax(outputs[1], dim=1).cpu().numpy()

        evaluations: list[tuple[np.ndarray, float]] = []

        for game_index, game_state in enumerate(game_states):
            legal_moves = get_legal_moves(game_state)
            legal_logits = policy_logits[game_index, legal_moves]
            maximum_logit = float(np.max(legal_logits))
            legal_probabilities = np.exp(legal_logits - maximum_logit)
            legal_probabilities /= np.sum(legal_probabilities)
            policy_probabilities = np.zeros(POLICY_MOVE_COUNT, dtype=np.float32)
            policy_probabilities[legal_moves] = legal_probabilities
            perspective_value = float(
                value_probabilities[game_index, 0]
                - value_probabilities[game_index, 1]
            )
            evaluations.append((policy_probabilities, perspective_value))

        return evaluations
