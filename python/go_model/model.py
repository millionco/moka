import mlx.core as mx
import mlx.nn as nn

from go_model.config import (
    BOARD_AREA,
    CONTEXT_INPUT_PLANE_COUNT,
    GLOBAL_POOL_HIDDEN_CHANNEL_COUNT,
    GLOBAL_POOL_RESIDUAL_BLOCK_COUNT,
    GLOBAL_RESIDUAL_BLOCK_INTERVAL,
    GLOBAL_RESIDUAL_HIDDEN_CHANNEL_COUNT,
    INPUT_PLANE_COUNT,
    NESTED_BOTTLENECK_CHANNEL_COUNT,
    NESTED_RESIDUAL_BLOCK_COUNT,
    POLICY_CHANNEL_COUNT,
    POLICY_MOVE_COUNT,
    RECURRENT_TRUNK_PASS_COUNT,
    RESIDUAL_BLOCK_COUNT,
    SCORE_HIDDEN_CHANNEL_COUNT,
    SPATIAL_POLICY_RESIDUAL_BLOCK_COUNT,
    TRUNK_CHANNEL_COUNT,
    VALUE_CHANNEL_COUNT,
    WIDE_BOTTLENECK_CHANNEL_COUNT,
    WIDE_RESIDUAL_BLOCK_COUNT,
    WIDE_TRUNK_CHANNEL_COUNT,
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


class NestedBottleneckBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reduce_convolution = nn.Conv2d(
            TRUNK_CHANNEL_COUNT,
            NESTED_BOTTLENECK_CHANNEL_COUNT,
            kernel_size=1,
        )
        self.first_spatial_convolution = nn.Conv2d(
            NESTED_BOTTLENECK_CHANNEL_COUNT,
            NESTED_BOTTLENECK_CHANNEL_COUNT,
            kernel_size=3,
            padding=1,
        )
        self.second_spatial_convolution = nn.Conv2d(
            NESTED_BOTTLENECK_CHANNEL_COUNT,
            NESTED_BOTTLENECK_CHANNEL_COUNT,
            kernel_size=3,
            padding=1,
        )
        self.expand_convolution = nn.Conv2d(
            NESTED_BOTTLENECK_CHANNEL_COUNT,
            TRUNK_CHANNEL_COUNT,
            kernel_size=1,
        )

    def __call__(self, inputs: mx.array) -> mx.array:
        hidden_values = nn.relu(self.reduce_convolution(inputs))
        hidden_values = nn.relu(self.first_spatial_convolution(hidden_values))
        hidden_values = nn.relu(self.second_spatial_convolution(hidden_values))
        return nn.relu(inputs + self.expand_convolution(hidden_values))


class GlobalNestedBottleneckBlock(NestedBottleneckBlock):
    def __init__(self) -> None:
        super().__init__()
        self.global_pooling_hidden = nn.Linear(
            NESTED_BOTTLENECK_CHANNEL_COUNT * 2,
            GLOBAL_RESIDUAL_HIDDEN_CHANNEL_COUNT,
        )
        self.global_bias_output = nn.Linear(
            GLOBAL_RESIDUAL_HIDDEN_CHANNEL_COUNT,
            NESTED_BOTTLENECK_CHANNEL_COUNT,
        )
        self.global_bias_output.weight = mx.zeros_like(
            self.global_bias_output.weight
        )
        self.global_bias_output.bias = mx.zeros_like(
            self.global_bias_output.bias
        )

    def __call__(self, inputs: mx.array) -> mx.array:
        hidden_values = nn.relu(self.reduce_convolution(inputs))
        hidden_values = nn.relu(self.first_spatial_convolution(hidden_values))
        global_values = mx.concatenate(
            [
                mx.mean(hidden_values, axis=(1, 2)),
                mx.max(hidden_values, axis=(1, 2)),
            ],
            axis=1,
        )
        global_hidden = nn.relu(self.global_pooling_hidden(global_values))
        global_bias = self.global_bias_output(global_hidden)[:, None, None, :]
        hidden_values = nn.relu(
            self.second_spatial_convolution(hidden_values + global_bias)
        )
        return nn.relu(inputs + self.expand_convolution(hidden_values))


class WideBottleneckBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reduce_convolution = nn.Conv2d(
            WIDE_TRUNK_CHANNEL_COUNT,
            WIDE_BOTTLENECK_CHANNEL_COUNT,
            kernel_size=1,
        )
        self.first_spatial_convolution = nn.Conv2d(
            WIDE_BOTTLENECK_CHANNEL_COUNT,
            WIDE_BOTTLENECK_CHANNEL_COUNT,
            kernel_size=3,
            padding=1,
        )
        self.second_spatial_convolution = nn.Conv2d(
            WIDE_BOTTLENECK_CHANNEL_COUNT,
            WIDE_BOTTLENECK_CHANNEL_COUNT,
            kernel_size=3,
            padding=1,
        )
        self.expand_convolution = nn.Conv2d(
            WIDE_BOTTLENECK_CHANNEL_COUNT,
            WIDE_TRUNK_CHANNEL_COUNT,
            kernel_size=1,
        )

    def __call__(self, inputs: mx.array) -> mx.array:
        hidden_values = nn.relu(self.reduce_convolution(inputs))
        hidden_values = nn.relu(self.first_spatial_convolution(hidden_values))
        hidden_values = nn.relu(self.second_spatial_convolution(hidden_values))
        return nn.relu(inputs + self.expand_convolution(hidden_values))


class MokaNetwork(nn.Module):
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

    def get_training_outputs(
        self,
        inputs: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        policy_logits, value = self(inputs)
        ownership = mx.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2]))
        return policy_logits, value, ownership


class MokaNestedNetwork(MokaNetwork):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.stem = nn.Conv2d(
            INPUT_PLANE_COUNT,
            TRUNK_CHANNEL_COUNT,
            kernel_size=3,
            padding=1,
        )
        self.residual_blocks = [
            NestedBottleneckBlock() for _ in range(NESTED_RESIDUAL_BLOCK_COUNT)
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


class MokaGlobalPoolNetwork(MokaNestedNetwork):
    def __init__(self) -> None:
        super().__init__()
        self.residual_blocks = [
            NestedBottleneckBlock()
            for _ in range(GLOBAL_POOL_RESIDUAL_BLOCK_COUNT)
        ]
        self.global_pooling_hidden = nn.Linear(
            TRUNK_CHANNEL_COUNT * 2,
            GLOBAL_POOL_HIDDEN_CHANNEL_COUNT,
        )
        self.global_policy_output = nn.Linear(
            GLOBAL_POOL_HIDDEN_CHANNEL_COUNT,
            POLICY_MOVE_COUNT,
            bias=False,
        )
        self.global_value_output = nn.Linear(
            GLOBAL_POOL_HIDDEN_CHANNEL_COUNT,
            SCORE_HIDDEN_CHANNEL_COUNT,
            bias=False,
        )
        self.global_policy_output.weight = mx.zeros_like(
            self.global_policy_output.weight
        )
        self.global_value_output.weight = mx.zeros_like(
            self.global_value_output.weight
        )

    def __call__(self, inputs: mx.array) -> tuple[mx.array, mx.array]:
        trunk_values = nn.relu(self.stem(inputs))

        for residual_block in self.residual_blocks:
            trunk_values = residual_block(trunk_values)

        global_values = mx.concatenate(
            [
                mx.mean(trunk_values, axis=(1, 2)),
                mx.max(trunk_values, axis=(1, 2)),
            ],
            axis=1,
        )
        global_hidden = nn.relu(self.global_pooling_hidden(global_values))
        policy_values = nn.relu(self.policy_convolution(trunk_values))
        policy_logits = self.policy_linear(
            mx.flatten(policy_values, start_axis=1)
        ) + self.global_policy_output(global_hidden)
        value_values = nn.relu(self.value_convolution(trunk_values))
        value_hidden = nn.relu(
            self.value_hidden(mx.flatten(value_values, start_axis=1))
            + self.global_value_output(global_hidden)
        )
        value = mx.tanh(self.value_output(value_hidden)).squeeze(-1)
        return policy_logits, value


class MokaGlobalResidualNetwork(MokaNestedNetwork):
    def __init__(
        self,
        global_residual_block_interval: int = GLOBAL_RESIDUAL_BLOCK_INTERVAL,
    ) -> None:
        super().__init__()
        if global_residual_block_interval <= 0:
            raise ValueError(
                "Global-residual block interval must be positive."
            )
        self.global_residual_block_interval = (
            global_residual_block_interval
        )
        self.residual_blocks = [
            (
                GlobalNestedBottleneckBlock()
                if (block_index + 1) % global_residual_block_interval == 0
                else NestedBottleneckBlock()
            )
            for block_index in range(NESTED_RESIDUAL_BLOCK_COUNT)
        ]


class MokaSoftPolicyNetwork(MokaNestedNetwork):
    def __init__(self) -> None:
        super().__init__()
        self.soft_policy_convolution = nn.Conv2d(
            TRUNK_CHANNEL_COUNT,
            POLICY_CHANNEL_COUNT,
            kernel_size=1,
        )
        self.soft_policy_linear = nn.Linear(
            POLICY_CHANNEL_COUNT * BOARD_AREA,
            POLICY_MOVE_COUNT,
        )

    def get_training_outputs(
        self,
        inputs: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        trunk_values = nn.relu(self.stem(inputs))

        for residual_block in self.residual_blocks:
            trunk_values = residual_block(trunk_values)

        policy_values = nn.relu(self.policy_convolution(trunk_values))
        policy_logits = self.policy_linear(mx.flatten(policy_values, start_axis=1))
        soft_policy_values = nn.relu(self.soft_policy_convolution(trunk_values))
        soft_policy_logits = self.soft_policy_linear(
            mx.flatten(soft_policy_values, start_axis=1)
        )
        value_values = nn.relu(self.value_convolution(trunk_values))
        value_hidden = nn.relu(
            self.value_hidden(mx.flatten(value_values, start_axis=1))
        )
        value = mx.tanh(self.value_output(value_hidden)).squeeze(-1)
        ownership = mx.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2]))
        return policy_logits, value, ownership, soft_policy_logits


class MokaAuxiliaryNetwork(MokaNestedNetwork):
    def __init__(self) -> None:
        super().__init__()
        self.auxiliary_ownership_convolution = nn.Conv2d(
            TRUNK_CHANNEL_COUNT,
            1,
            kernel_size=1,
        )
        self.auxiliary_score_convolution = nn.Conv2d(
            TRUNK_CHANNEL_COUNT,
            1,
            kernel_size=1,
        )
        self.auxiliary_score_output = nn.Linear(BOARD_AREA, 1)

    def get_training_outputs(
        self,
        inputs: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        trunk_values = nn.relu(self.stem(inputs))

        for residual_block in self.residual_blocks:
            trunk_values = residual_block(trunk_values)

        policy_values = nn.relu(self.policy_convolution(trunk_values))
        policy_logits = self.policy_linear(mx.flatten(policy_values, start_axis=1))
        value_values = nn.relu(self.value_convolution(trunk_values))
        value_hidden = nn.relu(
            self.value_hidden(mx.flatten(value_values, start_axis=1))
        )
        value = mx.tanh(self.value_output(value_hidden)).squeeze(-1)
        ownership = mx.tanh(
            self.auxiliary_ownership_convolution(trunk_values)
        ).squeeze(-1)
        score_values = self.auxiliary_score_convolution(trunk_values)
        score = mx.tanh(
            self.auxiliary_score_output(
                mx.flatten(score_values, start_axis=1),
            )
        ).squeeze(-1)
        return policy_logits, value, ownership, score


class MokaQAuxiliaryNetwork(MokaNestedNetwork):
    def __init__(self) -> None:
        super().__init__()
        self.auxiliary_q_convolution = nn.Conv2d(
            TRUNK_CHANNEL_COUNT,
            POLICY_CHANNEL_COUNT,
            kernel_size=1,
        )
        self.auxiliary_q_linear = nn.Linear(
            POLICY_CHANNEL_COUNT * BOARD_AREA,
            POLICY_MOVE_COUNT,
        )

    def get_training_outputs(
        self,
        inputs: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        trunk_values = nn.relu(self.stem(inputs))

        for residual_block in self.residual_blocks:
            trunk_values = residual_block(trunk_values)

        policy_values = nn.relu(self.policy_convolution(trunk_values))
        policy_logits = self.policy_linear(mx.flatten(policy_values, start_axis=1))
        value_values = nn.relu(self.value_convolution(trunk_values))
        value_hidden = nn.relu(
            self.value_hidden(mx.flatten(value_values, start_axis=1))
        )
        value = mx.tanh(self.value_output(value_hidden)).squeeze(-1)
        ownership = mx.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2]))
        q_values = mx.tanh(
            self.auxiliary_q_linear(
                mx.flatten(
                    nn.relu(self.auxiliary_q_convolution(trunk_values)),
                    start_axis=1,
                )
            )
        )
        return policy_logits, value, ownership, q_values


class MokaAttentionAuxiliaryNetwork(MokaNestedNetwork):
    def get_training_outputs(
        self,
        inputs: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        trunk_values = nn.relu(self.stem(inputs))

        for residual_block in self.residual_blocks:
            trunk_values = residual_block(trunk_values)

        policy_values = nn.relu(self.policy_convolution(trunk_values))
        policy_logits = self.policy_linear(mx.flatten(policy_values, start_axis=1))
        value_values = nn.relu(self.value_convolution(trunk_values))
        value_hidden = nn.relu(
            self.value_hidden(mx.flatten(value_values, start_axis=1))
        )
        value = mx.tanh(self.value_output(value_hidden)).squeeze(-1)
        ownership = mx.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2]))
        attention = mx.mean(mx.square(trunk_values), axis=3)
        attention /= mx.sqrt(
            mx.sum(mx.square(attention), axis=(1, 2), keepdims=True)
            + mx.finfo(attention.dtype).eps
        )
        return policy_logits, value, ownership, attention



class MokaContextNetwork(MokaNestedNetwork):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(
            CONTEXT_INPUT_PLANE_COUNT,
            TRUNK_CHANNEL_COUNT,
            kernel_size=3,
            padding=1,
        )


class MokaWideNetwork(MokaNetwork):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.stem = nn.Conv2d(
            INPUT_PLANE_COUNT,
            WIDE_TRUNK_CHANNEL_COUNT,
            kernel_size=3,
            padding=1,
        )
        self.residual_blocks = [
            WideBottleneckBlock() for _ in range(WIDE_RESIDUAL_BLOCK_COUNT)
        ]
        self.policy_convolution = nn.Conv2d(
            WIDE_TRUNK_CHANNEL_COUNT,
            POLICY_CHANNEL_COUNT,
            kernel_size=1,
        )
        self.policy_linear = nn.Linear(POLICY_CHANNEL_COUNT * 81, POLICY_MOVE_COUNT)
        self.value_convolution = nn.Conv2d(
            WIDE_TRUNK_CHANNEL_COUNT,
            VALUE_CHANNEL_COUNT,
            kernel_size=1,
        )
        self.value_hidden = nn.Linear(VALUE_CHANNEL_COUNT * 81, SCORE_HIDDEN_CHANNEL_COUNT)
        self.value_output = nn.Linear(SCORE_HIDDEN_CHANNEL_COUNT, 1)


class MokaRecurrentNetwork(MokaNestedNetwork):
    def __call__(self, inputs: mx.array) -> tuple[mx.array, mx.array]:
        trunk_values = nn.relu(self.stem(inputs))

        for _ in range(RECURRENT_TRUNK_PASS_COUNT):
            for residual_block in self.residual_blocks:
                trunk_values = residual_block(trunk_values)

        policy_values = nn.relu(self.policy_convolution(trunk_values))
        policy_logits = self.policy_linear(mx.flatten(policy_values, start_axis=1))
        value_values = nn.relu(self.value_convolution(trunk_values))
        value_hidden = nn.relu(self.value_hidden(mx.flatten(value_values, start_axis=1)))
        value = mx.tanh(self.value_output(value_hidden)).squeeze(-1)
        return policy_logits, value


class MokaSpatialNetwork(MokaNetwork):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.stem = nn.Conv2d(
            INPUT_PLANE_COUNT,
            TRUNK_CHANNEL_COUNT,
            kernel_size=3,
            padding=1,
        )
        self.residual_blocks = [
            NestedBottleneckBlock() for _ in range(SPATIAL_POLICY_RESIDUAL_BLOCK_COUNT)
        ]
        self.policy_convolution = nn.Conv2d(
            TRUNK_CHANNEL_COUNT,
            1,
            kernel_size=1,
        )
        self.policy_pass = nn.Linear(TRUNK_CHANNEL_COUNT, 1)
        self.ownership_convolution = nn.Conv2d(
            TRUNK_CHANNEL_COUNT,
            1,
            kernel_size=1,
        )
        self.value_convolution = nn.Conv2d(
            TRUNK_CHANNEL_COUNT,
            VALUE_CHANNEL_COUNT,
            kernel_size=1,
        )
        self.value_hidden = nn.Linear(VALUE_CHANNEL_COUNT * 81, SCORE_HIDDEN_CHANNEL_COUNT)
        self.value_output = nn.Linear(SCORE_HIDDEN_CHANNEL_COUNT, 1)

    def get_training_outputs(
        self,
        inputs: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        trunk_values = nn.relu(self.stem(inputs))

        for residual_block in self.residual_blocks:
            trunk_values = residual_block(trunk_values)

        board_policy_logits = mx.flatten(
            self.policy_convolution(trunk_values),
            start_axis=1,
        )
        pooled_trunk_values = mx.mean(trunk_values, axis=(1, 2))
        pass_policy_logit = self.policy_pass(pooled_trunk_values)
        policy_logits = mx.concatenate(
            [board_policy_logits, pass_policy_logit],
            axis=1,
        )
        ownership = mx.tanh(self.ownership_convolution(trunk_values)).squeeze(-1)
        value_values = nn.relu(self.value_convolution(trunk_values))
        value_hidden = nn.relu(self.value_hidden(mx.flatten(value_values, start_axis=1)))
        value = mx.tanh(self.value_output(value_hidden)).squeeze(-1)
        return policy_logits, value, ownership

    def __call__(self, inputs: mx.array) -> tuple[mx.array, mx.array]:
        policy_logits, value, _ = self.get_training_outputs(inputs)
        return policy_logits, value


def create_moka_network(
    use_nested_network: bool,
    use_spatial_network: bool,
    use_recurrent_network: bool,
    use_context_network: bool,
    use_wide_network: bool,
    use_global_pool_network: bool = False,
    use_global_residual_network: bool = False,
    global_residual_block_interval: int = GLOBAL_RESIDUAL_BLOCK_INTERVAL,
) -> MokaNetwork:
    if use_global_residual_network:
        return MokaGlobalResidualNetwork(global_residual_block_interval)

    if use_global_pool_network:
        return MokaGlobalPoolNetwork()

    if use_wide_network:
        return MokaWideNetwork()

    if use_context_network:
        return MokaContextNetwork()

    if use_recurrent_network:
        return MokaRecurrentNetwork()

    if use_spatial_network:
        return MokaSpatialNetwork()

    if use_nested_network:
        return MokaNestedNetwork()

    return MokaNetwork()


def checkpoint_uses_global_residual_network(
    checkpoint_path: str,
) -> bool:
    return get_checkpoint_global_residual_block_interval(checkpoint_path) > 0


def get_checkpoint_global_residual_block_interval(
    checkpoint_path: str,
) -> int:
    parameters = mx.load(checkpoint_path)
    block_numbers = sorted(
        {
            int(parameter_name.split(".")[1]) + 1
            for parameter_name in parameters
            if parameter_name.startswith("residual_blocks.")
            and ".global_pooling_hidden." in parameter_name
        }
    )
    if not block_numbers:
        return 0

    block_interval = block_numbers[0]
    if block_numbers != list(
        range(
            block_interval,
            NESTED_RESIDUAL_BLOCK_COUNT + 1,
            block_interval,
        )
    ):
        raise ValueError("Unsupported irregular global-residual checkpoint.")
    return block_interval


def checkpoint_uses_nested_network(checkpoint_path: str) -> bool:
    parameters = mx.load(checkpoint_path)
    return any(
        parameter_name.startswith("residual_blocks.")
        and ".reduce_convolution." in parameter_name
        for parameter_name in parameters
    )


def create_moka_network_for_checkpoint(
    checkpoint_path: str,
    use_nested_network: bool = False,
    use_spatial_network: bool = False,
    use_recurrent_network: bool = False,
    use_context_network: bool = False,
    use_wide_network: bool = False,
    use_global_pool_network: bool = False,
    use_global_residual_network: bool = False,
    global_residual_block_interval: int | None = None,
) -> MokaNetwork:
    checkpoint_global_residual_block_interval = (
        get_checkpoint_global_residual_block_interval(checkpoint_path)
    )
    return create_moka_network(
        (
            use_nested_network
            or checkpoint_uses_nested_network(checkpoint_path)
        ),
        use_spatial_network,
        use_recurrent_network,
        use_context_network,
        use_wide_network,
        use_global_pool_network,
        (
            use_global_residual_network
            or checkpoint_global_residual_block_interval > 0
        ),
        (
            global_residual_block_interval
            or checkpoint_global_residual_block_interval
            or GLOBAL_RESIDUAL_BLOCK_INTERVAL
        ),
    )
