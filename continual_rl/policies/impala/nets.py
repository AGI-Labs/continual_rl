"""
This file contains networks that are capable of handling (batch, time, [applicable features])
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from continual_rl.utils.common_nets import get_network_for_size
from continual_rl.utils.utils import Utils


class ImpalaNet(nn.Module):
    """
    Based on Impala's AtariNet, taken from:
    https://github.com/facebookresearch/torchbeast/blob/6ed409587e8eb16d4b2b1d044bf28a502e5e3230/torchbeast/monobeast.py
    """
    def __init__(self, observation_space, action_spaces, model_flags, conv_net=None):
        super().__init__()
        self.use_lstm = model_flags.use_lstm
        conv_net_arch = model_flags.conv_net_arch
        self.num_actions = Utils.get_max_discrete_action_space(action_spaces).n
        self._model_flags = model_flags
        self._action_spaces = action_spaces  # The max number of actions - the policy's output size is always this
        self._current_action_size = None  # Set by the environment_runner
        self._observation_space = observation_space

        if conv_net is None:
            # The conv net gets channels and time merged together (mimicking the original FrameStacking)
            combined_observation_size = [observation_space.shape[0] * observation_space.shape[1],
                                         observation_space.shape[2],
                                         observation_space.shape[3]]
            self._conv_net = get_network_for_size(combined_observation_size, arch=conv_net_arch)
        else:
            self._conv_net = conv_net

        # FC output size + one-hot of last action + last reward.
        core_output_size = self._conv_net.output_size + self.num_actions + 1
        self.policy = nn.Linear(core_output_size, self.num_actions)

        self._baseline_output_dim = 2 if model_flags.baseline_includes_uncertainty else 1

        # The first output value is the standard critic value. The second is an optional value the policies may use
        # which we call "uncertainty".
        if model_flags.baseline_extended_arch:
            self.baseline = nn.Sequential(
                nn.Linear(core_output_size, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, self._baseline_output_dim)
            )
        else:
            self.baseline = nn.Linear(core_output_size, self._baseline_output_dim)

        # used by update_running_moments()
        # second moment is variance
        self.register_buffer("reward_sum", torch.zeros(()))
        self.register_buffer("reward_m2", torch.zeros(()))
        self.register_buffer("reward_count", torch.zeros(()).fill_(1e-8))

    def initial_state(self, batch_size):
        assert not self.use_lstm, "LSTM not currently implemented. Ensure this gets initialized correctly when it is" \
                                  "implemented."
        return tuple()

    def forward(self, inputs, action_space_id, core_state=()):
        x = inputs["frame"]  # [T, B, S, C, H, W]. T=timesteps in collection, S=stacked frames
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = torch.flatten(x, 1, 2)  # Merge stacked frames and channels.
        x = x.float() / self._observation_space.high.max()
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
        current_action_size = self._action_spaces[action_space_id].n
        policy_logits_subset = policy_logits[:, :current_action_size]

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits_subset, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits_subset, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B, self._baseline_output_dim)
        action = action.view(T, B)

        output_dict = dict(policy_logits=policy_logits, baseline=baseline[:, :, 0], action=action)

        if self._model_flags.baseline_includes_uncertainty:
            output_dict["uncertainty"] = baseline[:, :, 1]

        return (
            output_dict,
            core_state,
        )

    # from https://github.com/MiniHackPlanet/MiniHack/blob/e124ae4c98936d0c0b3135bf5f202039d9074508/minihack/agent/polybeast/models/base.py#L67
    @torch.no_grad()
    def update_running_moments(self, reward_batch):
        """Maintains a running mean of reward."""
        new_count = len(reward_batch)
        new_sum = torch.sum(reward_batch)
        new_mean = new_sum / new_count

        curr_mean = self.reward_sum / self.reward_count
        new_m2 = torch.sum((reward_batch - new_mean) ** 2) + (
            (self.reward_count * new_count)
            / (self.reward_count + new_count)
            * (new_mean - curr_mean) ** 2
        )

        self.reward_count += new_count
        self.reward_sum += new_sum
        self.reward_m2 += new_m2

    @torch.no_grad()
    def get_running_std(self):
        """Returns standard deviation of the running mean of the reward."""
        return torch.sqrt(self.reward_m2 / self.reward_count)
