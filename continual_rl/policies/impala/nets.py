"""
This file contains networks that are capable of handling (batch, time, [applicable features])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from continual_rl.utils.common_nets import get_network_for_size
from continual_rl.utils.utils import Utils
from continual_rl.policies.impala.random_process import OrnsteinUhlenbeckProcess


class ImpalaNet(nn.Module):
    """
    Based on Impala's AtariNet, taken from:
    https://github.com/facebookresearch/torchbeast/blob/6ed409587e8eb16d4b2b1d044bf28a502e5e3230/torchbeast/monobeast.py
    """
    def __init__(self, observation_space, action_spaces, model_flags, conv_net=None, skip_net_init=False):
        super().__init__()
        self.use_lstm = model_flags.use_lstm

        if not skip_net_init:  # TODO: this is hackier than I'd like. Mostly doing this to keep the running moments code easy. Do this better though
            self.num_actions = Utils.get_max_discrete_action_space(action_spaces).n
            self._action_spaces = action_spaces  # The max number of actions - the policy's output size is always this
            self._current_action_size = None  # Set by the environment_runner
            self._observation_space = observation_space

            if conv_net is None:
                # The conv net gets channels and time merged together (mimicking the original FrameStacking)
                combined_observation_size = [observation_space.shape[0] * observation_space.shape[1],
                                             observation_space.shape[2],
                                             observation_space.shape[3]]
                self._conv_net = get_network_for_size(combined_observation_size)
            else:
                self._conv_net = conv_net

            # FC output size + one-hot of last action + last reward.
            core_output_size = self._conv_net.output_size + self.num_actions + 1
            self.policy = nn.Linear(core_output_size, self.num_actions)
            self.baseline = nn.Linear(core_output_size, 1)

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
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
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


"""
Here down is from https://github.com/ghliu/pytorch-ddpg/blob/master/ddpg.py
This is a DDPG model that works, so I'm integrating as much of it as I can, and generalizing from there.
"""
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out


class ContinuousImpalaNet(ImpalaNet):
    def __init__(self, observation_space, action_spaces, model_flags, conv_net=None):
        super().__init__(observation_space, action_spaces, model_flags, conv_net, skip_net_init=True)
        self._action_spaces = action_spaces
        first_action_space = list(action_spaces.values())[0]
        self.num_actions = first_action_space.shape[0]
        combined_observation_size = observation_space.shape[0] * observation_space.shape[1]

        self._model_flags = model_flags
        self._actor = Actor(nb_states=combined_observation_size, nb_actions=self.num_actions)
        self._critic = Critic(nb_states=combined_observation_size, nb_actions=self.num_actions)
        self._random_process = OrnsteinUhlenbeckProcess(size=self.num_actions, theta=model_flags.ou_theta, mu=model_flags.ou_mu, sigma=model_flags.ou_sigma)
        self._epsilon = 1.0

    def actor_parameters(self):
        return self._actor.parameters()

    def critic_parameters(self):
        return self._critic.parameters()

    def forward(self, inputs, action_space_id, core_state=(), use_exploration=True):
        observation = inputs['frame']
        T, B, *_ = observation.shape
        observation = torch.flatten(observation, 0, 1)  # Merge time and batch.
        observation = torch.flatten(observation, 1, 2)  # Merge stacked frames and channels.

        action = self._actor(observation)

        exploration = torch.tensor(use_exploration * max(self._epsilon, 0) * self._random_process.sample())
        exploration = exploration.to(action.device)

        action = action + exploration

        # Scale the action to the range expected by the environment (Pytorch-DDPG does this in an environment wrapper)...TODO
        action = torch.clip(action, -1., 1.)
        action_scale = (self._action_spaces[action_space_id].high - self._action_spaces[action_space_id].low) / 2.
        action_scale = torch.tensor(action_scale).to(action.device)
        action_bias = (self._action_spaces[action_space_id].high + self._action_spaces[action_space_id].low) / 2.
        action_bias = torch.tensor(action_bias).to(action.device)
        action = action_scale * action + action_bias

        if self._model_flags.decay_epsilon:
            self._epsilon -= self._model_flags.epsilon_decay_rate

        q_batch = self._critic([observation, action.float()])

        q_batch = q_batch.view(T, B)
        policy_logits = action.view(T, B, self.num_actions).float()  # TODO:...currently putting it here for CLEAR, but the naming is clearly misleading, at the very least

        return (
            dict(baseline=q_batch, action=action, policy_logits=policy_logits),
            core_state,
        )
