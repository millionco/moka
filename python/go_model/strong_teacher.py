import sys
from pathlib import Path

import numpy as np
import torch

from go_model.board import GameState, get_legal_moves
from go_model.config import BOARD_AREA, BOARD_SIZE, KOMI_POINTS, POLICY_MOVE_COUNT


class StrongKataGoTeacher:
    def __init__(
        self,
        checkpoint_path: Path,
        source_path: Path,
        policy_output_index: int = 0,
    ) -> None:
        sys.path.insert(0, str((source_path / "python").resolve()))

        from katago.game.board import Board
        from katago.game.features import Features
        from katago.game.gamestate import GameState as KataGoGameState
        from katago.train.load_model import load_model
        from katago.train.model_pytorch import ExtraOutputs

        self.board_class = Board
        self.features_class = Features
        self.extra_outputs_class = ExtraOutputs
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
        self.policy_output_index = policy_output_index

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
        return [
            (evaluation[0], evaluation[1])
            for evaluation in self.evaluate_batch_with_auxiliary(game_states)
        ]

    def evaluate_batch_with_auxiliary(
        self,
        game_states: list[GameState],
    ) -> list[tuple[np.ndarray, float, np.ndarray, float, float, np.ndarray]]:
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
            extra_outputs = self.extra_outputs_class(["trunkfinal"])
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
                extra_outputs=extra_outputs,
            )
            outputs = self.model.postprocess_output(raw_outputs)[0]
            policy_logits = outputs[0][
                :,
                self.policy_output_index,
            ].cpu().numpy()
            value_probabilities = torch.softmax(outputs[1], dim=1).cpu().numpy()
            short_value_probabilities = (
                torch.softmax(outputs[2][:, 2], dim=1).cpu().numpy()
            )
            ownerships = torch.tanh(outputs[4][:, 0]).cpu().numpy()
            score_means = outputs[8].cpu().numpy()
            trunk_values = extra_outputs.returned["trunkfinal"]
            attention_values = torch.mean(torch.square(trunk_values), dim=1)
            attention_values /= torch.sqrt(
                torch.sum(torch.square(attention_values), dim=(1, 2), keepdim=True)
                + torch.finfo(attention_values.dtype).eps
            )
            attentions = attention_values.cpu().numpy()

        evaluations: list[
            tuple[np.ndarray, float, np.ndarray, float, float, np.ndarray]
        ] = []

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
            short_value = float(
                short_value_probabilities[game_index, 0]
                - short_value_probabilities[game_index, 1]
            )
            evaluations.append(
                (
                    policy_probabilities,
                    perspective_value,
                    ownerships[game_index].astype(np.float32),
                    short_value,
                    float(score_means[game_index]),
                    attentions[game_index].astype(np.float32),
                )
            )

        return evaluations
