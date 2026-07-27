import mlx.core as mx
import mlx.nn as nn

from go_model.config import (
    INPUT_PLANE_COUNT,
    POLICY_CHANNEL_COUNT,
    POLICY_MOVE_COUNT,
    RESIDUAL_BLOCK_COUNT,
    SCORE_HIDDEN_CHANNEL_COUNT,
    TRUNK_CHANNEL_COUNT,
    VALUE_CHANNEL_COUNT,
)


class ResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first_convolution = nn.Conv2d(
            TRUNK_CHANNEL_COUNT,
            TRUNK_CHANNEL_COUNT,
            kernel_size=3,
            padding=1,
        )
        self.second_convolution = nn.Conv2d(
            TRUNK_CHANNEL_COUNT,
            TRUNK_CHANNEL_COUNT,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, inputs: mx.array) -> mx.array:
        hidden_values = nn.relu(self.first_convolution(inputs))
        return nn.relu(inputs + self.second_convolution(hidden_values))


class StudentNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(
            INPUT_PLANE_COUNT,
            TRUNK_CHANNEL_COUNT,
            kernel_size=3,
            padding=1,
        )
        self.residual_blocks = [
            ResidualBlock() for _ in range(RESIDUAL_BLOCK_COUNT)
        ]
        self.policy_convolution = nn.Conv2d(
            TRUNK_CHANNEL_COUNT,
            POLICY_CHANNEL_COUNT,
            kernel_size=1,
        )
        self.policy_linear = nn.Linear(POLICY_CHANNEL_COUNT * 81, POLICY_MOVE_COUNT)
        self.value_convolution = nn.Conv2d(
            TRUNK_CHANNEL_COUNT,
            VALUE_CHANNEL_COUNT,
            kernel_size=1,
        )
        self.value_hidden = nn.Linear(VALUE_CHANNEL_COUNT * 81, SCORE_HIDDEN_CHANNEL_COUNT)
        self.value_output = nn.Linear(SCORE_HIDDEN_CHANNEL_COUNT, 1)

    def __call__(self, inputs: mx.array) -> tuple[mx.array, mx.array]:
        trunk_values = nn.relu(self.stem(inputs))

        for residual_block in self.residual_blocks:
            trunk_values = residual_block(trunk_values)

        policy_values = nn.relu(self.policy_convolution(trunk_values))
        policy_logits = self.policy_linear(mx.flatten(policy_values, start_axis=1))
        value_values = nn.relu(self.value_convolution(trunk_values))
        value_hidden = nn.relu(self.value_hidden(mx.flatten(value_values, start_axis=1)))
        value = mx.tanh(self.value_output(value_hidden)).squeeze(-1)
        return policy_logits, value

