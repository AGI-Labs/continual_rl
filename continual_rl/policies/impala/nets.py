"""
This file contains networks that are capable of handling (batch, time, [applicable features])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from continual_rl.utils.common_nets import get_network_for_size, ModelUtils, CommonConv, ConvNet7x7, ConvNet28x28


def get_custom_network_for_size(size, flavor):
    """
    Size is expected to be [channel, dim, dim]
    """
    size = list(size)  # In case the input is a tuple
    if flavor == "100x":
        if size[1:] == [7, 7]:
            #net = ConvNet7x7_100x_tall
            net = ConvNet7x7_100x_wide
        elif size[1:] == [28, 28]:
            net = ConvNet28x28_100x
        elif size[1:] == [84, 84]:
            net = ConvNet84x84_100x
        else:
            raise AttributeError("Unexpected input size")

    elif flavor == "1x_larger_sane":  # SANE was run using the larger size (extra layers), so comparisons with the same
        if size[1:] == [7, 7]:
            net = ConvNet7x7_1x_larger_SANE
        elif size[1:] == [28, 28]:
            net = ConvNet28x28_1x_larger_SANE
        else:
            raise AttributeError("Unexpected input size")

    elif flavor == "100x_larger_sane":
        if size[1:] == [7, 7]:
            net = ConvNet7x7_100x_larger_SANE
        elif size[1:] == [28, 28]:
            net = ConvNet28x28_100x_larger_SANE
        else:
            raise AttributeError("Unexpected input size")
    else:
        raise AttributeError("Unexpected network flavor")

    return net(size)


class ImpalaNet(nn.Module):
    """
    Ensures the models have the parameters IMPALA requires.
    """
    # In part taken from https://github.com/facebookresearch/torchbeast/blob/6ed409587e8eb16d4b2b1d044bf28a502e5e3230/torchbeast/monobeast.py
    # LICENSE available in continual_rl/policies/impala/torchbeast/LICENSE
    def __init__(self, observation_space, num_actions, use_lstm=False, max_actions=None, net_flavor=None):
        super().__init__()
        self.num_actions = max_actions  # The max number of actions - the policy's output size is always this
        self.use_lstm = use_lstm

        if net_flavor == "default":
            self._conv_net = get_network_for_size(observation_space)
        else:
            self._conv_net = get_custom_network_for_size(observation_space, net_flavor)

        self._current_action_size = num_actions  # What subset of the max actions we should be using

        # FC output size + one-hot of last action + last reward.
        core_output_size = self._conv_net.output_size + self.num_actions + 1
        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size):
        assert not self.use_lstm, "LSTM not currently implemented. Ensure this gets initialized correctly when it is" \
                                  "implemented."
        return tuple()

    def forward(self, inputs, core_state=()):
        x = inputs["frame"]  # [T, B, C, H, W].
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0
        x = self._conv_net(x)
        x = F.relu(x)

        one_hot_last_action = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions
        ).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1).float()
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        # Used to select the action appropriate for this task (might be from a reduced set)
        policy_logits_subset = policy_logits[:, :self._current_action_size]

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits_subset, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits_subset, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )


# TODO: not especially tested, but ...it's here for consistency (170M params)
class ConvNet84x84_100x(CommonConv):
    def __init__(self, observation_shape):
        # This is the same as used in AtariNet in Impala (torchbeast implementation)
        output_size = 512
        conv_net = nn.Sequential(
                                  nn.Conv2d(in_channels=observation_shape[0], out_channels=32, kernel_size=8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=32, out_channels=8192, kernel_size=1, stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=8192, out_channels=8192, kernel_size=1, stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=8192, out_channels=8192, kernel_size=1, stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=8192, out_channels=32, kernel_size=1, stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=64, out_channels=4096, kernel_size=1, stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=4096, out_channels=64, kernel_size=1, stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                                  nn.ReLU(),
                                  nn.Flatten())
        intermediate_dim = ModelUtils.compute_output_size(conv_net, observation_shape)
        post_flatten = nn.Linear(intermediate_dim, output_size)
        super().__init__(conv_net, post_flatten, output_size)


class ConvNet28x28_100x(CommonConv):  # ~5.5M
    def __init__(self, observation_shape):
        output_size = 32
        conv_net = nn.Sequential(
            nn.Conv2d(observation_shape[0], 24, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(24, 2048, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(2048, 2048, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(2048, 24, kernel_size=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(48, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1024, 48, kernel_size=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        intermediate_dim = ModelUtils.compute_output_size(conv_net, observation_shape)
        post_flatten = nn.Linear(intermediate_dim, output_size)
        super().__init__(conv_net, post_flatten, output_size)


class ConvNet7x7_100x_tall(CommonConv):  # ~1.4 M (Note: doesn't train on minigrid effectively)
    def __init__(self, observation_shape):
        # Based on: https://github.com/lcswillems/rl-starter-files/blob/master/model.py
        output_size = 32
        conv_net = nn.Sequential(
            nn.Conv2d(observation_shape[0], 16, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1024, 16, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
            nn.Flatten()
        )
        intermediate_dim = ModelUtils.compute_output_size(conv_net, observation_shape)
        post_flatten = nn.Linear(intermediate_dim, output_size)
        super().__init__(conv_net, post_flatten, output_size)


class ConvNet7x7_100x_wide(CommonConv):  # ~1.2 M
    def __init__(self, observation_shape):
        output_size = 32
        conv_net = nn.Sequential(
            nn.Conv2d(observation_shape[0], 1024, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(1024, 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
            nn.Flatten()
        )
        intermediate_dim = ModelUtils.compute_output_size(conv_net, observation_shape)
        post_flatten = nn.Linear(intermediate_dim, output_size)
        super().__init__(conv_net, post_flatten, output_size)


class ConvNet28x28_100x_larger_SANE(CommonConv):  # ~7.5M
    def __init__(self, observation_shape):
        output_size = 32
        conv_net = nn.Sequential(
            nn.Conv2d(observation_shape[0], 2700, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(2700, 2700, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(2700, 24, kernel_size=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten()
        )
        intermediate_dim = ModelUtils.compute_output_size(conv_net, observation_shape)
        post_flatten = nn.Linear(intermediate_dim, output_size)
        super().__init__(conv_net, post_flatten, output_size)


class ConvNet7x7_100x_larger_SANE(CommonConv):  # ~3.5 M
    def __init__(self, observation_shape):
        output_size = 32
        conv_net = nn.Sequential(
            nn.Conv2d(observation_shape[0], 1800, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(1800, 1800, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(1800, 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
            nn.Flatten()
        )
        intermediate_dim = ModelUtils.compute_output_size(conv_net, observation_shape)
        post_flatten = nn.Linear(intermediate_dim, output_size)
        super().__init__(conv_net, post_flatten, output_size)


class ConvNet28x28_1x_larger_SANE(CommonConv):  # ~80k
    def __init__(self, observation_shape):
        output_size = 32
        conv_net = ConvNet28x28(observation_shape)
        intermediate_dim = ModelUtils.compute_output_size(conv_net, observation_shape)
        post_flatten = nn.Sequential(nn.Linear(intermediate_dim, output_size),
                                     nn.ReLU(),
                                     nn.Linear(output_size, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, output_size))
        super().__init__(conv_net, post_flatten, output_size)


class ConvNet7x7_1x_larger_SANE(CommonConv):  # ~39k
    def __init__(self, observation_shape):
        output_size = 32
        conv_net = ConvNet7x7(observation_shape)
        intermediate_dim = ModelUtils.compute_output_size(conv_net, observation_shape)
        post_flatten = nn.Sequential(nn.Linear(intermediate_dim, output_size),
                                     nn.ReLU(),
                                     nn.Linear(output_size, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, output_size))
        super().__init__(conv_net, post_flatten, output_size)

