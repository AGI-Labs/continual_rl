"""
This file contains networks that are capable of handling (batch, time, [applicable features])
"""
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from continual_rl.utils.common_nets import get_network_for_size
from continual_rl.utils.utils import Utils
from continual_rl.policies.impala.random_process import OrnsteinUhlenbeckProcess
#from ravens_torch.agents.transporter import OriginalTransporterAgent, GoalTransporterAgent
from cliport.agents.transporter_image_goal import ImageGoalTransporterAgent
#from cliport.agents.transporter import TwoStreamClipUNetLatTransporterAgent, FullAttentionTransporterAgent, ClipUNetTransporterAgent
from cliport.agents.transporter import TwoStreamClipUNetLatTransporterAgent, ClipUNetTransporterAgent
from cliport.agents.transporter_lang_goal import TwoStreamClipLingUNetLatTransporterAgent
from continual_rl.envs.ravens_demonstration_env import RavensDemonstrationEnv

# TODO: ...
from home_robot.ros.camera import Camera
from home_robot.utils.image import opengl_depth_to_xyz
from home_robot.utils.image import depth_to_xyz
from data_tools.point_cloud import (depth_to_xyz, show_point_cloud, get_pcd)
from home_robot.policy.pt_query import QueryPointnet
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, MLP
import trimesh
import torchvision
from data_tools.point_cloud import add_additive_noise_to_xyz, add_multiplicative_noise


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
        x = inputs["image"]  # [T, B, S, C, H, W]. T=timesteps in collection, S=stacked frames
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


def get_net_for_observation_space(observation_space):
    if isinstance(observation_space, gym.spaces.Dict):
        combined_observation_size = {}
        for key in observation_space:
            shape = observation_space[key].shape
            combined_observation_size[key] = [shape[0] * shape[1], *shape[2:]]
    elif len(observation_space.shape) == 2:
        combined_observation_size = [observation_space.shape[0] * observation_space.shape[1]]
    elif len(observation_space.shape) == 4:
        combined_observation_size = [observation_space.shape[0] * observation_space.shape[1],
                                     observation_space.shape[2],
                                     observation_space.shape[3]]
    else:
        raise Exception(f"Unexpected observation shape: {observation_space.shape}")
    return get_network_for_size(combined_observation_size)


class Actor(nn.Module):
    def __init__(self, observation_space, nb_actions, hidden1=400, hidden2=300, init_w=3e-3, use_clip=False, preprocess=None):
        super(Actor, self).__init__()

        # TODO: de-dupe with Critic
        if use_clip:
            import clip
            self.model, clip_preprocess = clip.load("ViT-B/32", "cpu")  # Model is a class parameter so it gets moved to the right device...TODO: in theory
            self.preprocess = preprocess #lambda x: clip_preprocess(x['image'].reshape(-1, *x['image'].shape[-3:]))  # TODO... preprocess seems to require PIL images which is inconvenient for batching...so just experimenting with *not*
            self.encoder = lambda x: self.model.encode_image(x['image']).detach()  # TODO
            output_dim = self.model.visual.output_dim
        else:
            self.encoder = get_net_for_observation_space(observation_space)
            self.preprocess = preprocess
            output_dim = self.encoder.output_size

        self.fc1 = nn.Linear(output_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        #self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        if self.preprocess is not None:
            x = self.preprocess(x)

        out = self.encoder(x)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        #out = self.tanh(out)
        return out


class Critic(nn.Module):
    def __init__(self, observation_space, nb_actions, hidden1=400, hidden2=300, init_w=3e-3, use_clip=False, preprocess=None):
        super(Critic, self).__init__()

        # TODO: de-dupe with Actor
        if use_clip:
            import clip
            self.model, clip_preprocess = clip.load("ViT-B/32", "cpu")  # Model is a class parameter so it gets moved to the right device...TODO: in theory. RN50
            self.preprocess = preprocess #lambda x: clip_preprocess(x['image'].reshape(-1, *x['image'].shape[-3:]))  # TODO... preprocess seems to require PIL images which is inconvenient for batching...so just experimenting with *not*
            self.encoder = lambda x: self.model.encode_image(x['image']).detach()  # TODO
            output_dim = self.model.visual.output_dim
        else:
            self.encoder = get_net_for_observation_space(observation_space)
            self.preprocess = preprocess
            output_dim = self.encoder.output_size

        self.fc1 = nn.Linear(output_dim, hidden1)
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        #self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs

        if self.preprocess is not None:
            x = self.preprocess(x)

        out = self.encoder(x)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out


class ContinuousImpalaNet(ImpalaNet):
    def __init__(self, observation_space, action_spaces, model_flags, conv_net=None):
        super().__init__(observation_space, action_spaces, model_flags, conv_net, skip_net_init=True)
        self._observation_space = observation_space
        self._action_spaces = action_spaces
        first_action_space = list(action_spaces.values())[0]
        self.num_actions = first_action_space.shape[0]

        self._model_flags = model_flags
        preprocess = (lambda x: self._normalize_all_observations(x, self._observation_space)) #if not model_flags.use_clip else None
        self._actor = Actor(observation_space=observation_space, nb_actions=self.num_actions, use_clip=model_flags.use_clip, preprocess=preprocess) #, init_w=0.5)
        self._critic = Critic(observation_space=observation_space, nb_actions=self.num_actions, use_clip=model_flags.use_clip, preprocess=preprocess) #, init_w=0.5)
        self._random_process = OrnsteinUhlenbeckProcess(size=self.num_actions, theta=model_flags.ou_theta, mu=model_flags.ou_mu, sigma=model_flags.ou_sigma)
        self._epsilon = 1.0  # TODO: this is weird for parallelism

        self.register_buffer("out_mean_sum", torch.zeros((self.num_actions,)))
        self.register_buffer("out_std_sum", torch.zeros((self.num_actions,)))
        self.register_buffer("out_count", torch.zeros(()).fill_(1e-8))

    def update_running_stats(self, dataset):
        reshaped_actions = dataset['action'].reshape((-1, self.num_actions))
        self.out_mean_sum += reshaped_actions.mean(0)
        self.out_std_sum += reshaped_actions.std(0)
        self.out_count += 1  # TODO: alternatively batch size...HACKY/temp

    def get_out_mean(self):
        return self.out_mean_sum / self.out_count

    def get_out_std(self):
        return self.out_std_sum / self.out_count

    def actor_parameters(self):
        return self._actor.parameters()

    def critic_parameters(self):
        return self._critic.parameters()

    def _normalize_observation(self, observation, obs_low, obs_high):
        observation = torch.flatten(observation, 0, 1)  # Merge time and batch.
        observation = torch.flatten(observation, 1, 2)  # Merge stacked frames and channels.
        observation = observation.float()
        
        obs_high = torch.tensor(obs_high).to(device=observation.device)
        obs_low = torch.tensor(obs_low).to(device=observation.device)

        observation = (observation - obs_low) / (obs_high - obs_low)
        return observation

    def _normalize_all_observations(self, inputs, observation_space):
        if isinstance(observation_space, gym.spaces.Dict):
            observation = {}

            for key in observation_space.spaces.keys():
                if key != "state_vector":  # TODO: temp for testing, state_vector stuff
                    observation[key] = self._normalize_observation(inputs[key], observation_space[key].low, observation_space[key].high)
                else:
                    # TODO: temp! De-dupe with normalize if I keep
                    print("Reminder: state vector not normalized")
                    key_obs = inputs[key]
                    key_obs = torch.flatten(key_obs, 0, 1)  # Merge time and batch.
                    key_obs = torch.flatten(key_obs, 1, 2)  # Merge stacked frames and channels.
                    key_obs = key_obs.float()
                    observation[key] = key_obs

                # TODO for testing, 0 out the image so we're only using the state vector
                #if key == "image":
                #    observation[key] *= 0
        else:
            observation = self._normalize_observation(inputs['frame'], observation_space.low, observation_space.high)

        return observation

    def _get_time_and_batch(self, inputs, observation_space):
        if isinstance(observation_space, gym.spaces.Dict):
            T, B = None, None
            for key in observation_space.spaces.keys():
                if T is None:
                    T, B, *_ = inputs[key].shape
                else:
                    assert T == inputs[key].shape[0] and B == inputs[key].shape[1], f"Mismatched T and B: {T, B} vs {inputs[key].shape[:2]}"
        else:
            T, B, *_ = inputs['frame'].shape

        return T, B

    def forward(self, inputs, action_space_id, core_state=(), action=None):
        T, B = self._get_time_and_batch(inputs, self._observation_space)
        observation = inputs

        if action is None:
            action_raw = self._actor(observation)

            # TODO: turn off in eval-mode, I guess
            if self._model_flags.use_exploration:
                exploration = torch.tensor(max(self._epsilon, 0) * self._random_process.sample())
                exploration = exploration.to(action_raw.device)
                action = action_raw + exploration
            else:
                action = action_raw

            # Scale the action to the range expected by the environment (Pytorch-DDPG does this in an environment wrapper)...TODO
            # TODO: handle (-inf, inf) action spaces
            if self._model_flags.use_running_stats:
                action = self.get_out_mean() + self.get_out_std() * action
            else:
                action = torch.clip(action, -1., 1.)
                action_scale = (self._action_spaces[action_space_id].high - self._action_spaces[action_space_id].low) / 2.
                action_scale = torch.tensor(action_scale).to(action.device)
                action_bias = (self._action_spaces[action_space_id].high + self._action_spaces[action_space_id].low) / 2.
                action_bias = torch.tensor(action_bias).to(action.device)
                action = action_scale * action + action_bias
        else:
            action_raw = action.flatten(0, 1)  # TODO double check

        if self._model_flags.decay_epsilon:  # TODO: this ...doesn't make sense, for parallelism, and because this method is called during training
            self._epsilon -= self._model_flags.epsilon_decay_rate

        q_batch = self._critic([observation, action_raw.float()])

        q_batch = q_batch.view(T, B)
        policy_logits = action.view(T, B, self.num_actions).float()  # TODO:...currently putting it here for CLEAR, but the naming is clearly misleading, at the very least
        action = policy_logits  # TODO...why do I have this separation? I think it's outdated
        action = action.float()  # TODO: temp for multigoal robot?

        return (
            dict(baseline=q_batch, action=action, policy_logits=policy_logits),
            core_state,
        )


# The following are from: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        original_device = pos.device
        row, col = radius(pos.cpu(), pos[idx].cpu(), self.r, batch.cpu(), batch[idx].cpu(),
                          max_num_neighbors=64)  # WTF this gives *very* different results on cpu vs cuda
        row = row.to(original_device)
        col = col.to(original_device)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class SimplePointNet(torch.nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()

        # Input channels account for both pos and node features.
        #self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))

        #self.sa1_module = SAModule(0.5, 0.2, MLP([input_size, 64, 64, 128]))  # TODO: first defaults to 3, but seems to need to be 6? TODO
        #self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa1_module = SAModule(0.5, 0.2, MLP([input_size, 64, 64, 128])) #, norm=None, act="leakyrelu"))  # TODO: first defaults to 3, but seems to need to be 6? TODO
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256])) #, norm=None, act="leakyrelu"))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))  #, norm=None, act="leakyrelu"))

        self.mlp = MLP([1024, 512, 256, num_actions], norm=None) #dropout=0.5, norm=None)

    def forward(self, x, pos, batch):
        #sa0_out = (data.x, data.pos, data.batch)
        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x) #.log_softmax(dim=-1)


class PointQueryImpalaNet(nn.Module):

    def __init__(self, observation_space, action_spaces, model_flags, conv_net=None):
        super().__init__()

        self.use_lstm = model_flags.use_lstm
        self._observation_space = observation_space
        self._action_spaces = action_spaces
        first_action_space = list(action_spaces.values())[0]
        self.num_actions = first_action_space.shape[0]

        self._model_flags = model_flags

        # PointQuery takes in rgb, depth
        #self._point_cloud_encoder = QueryPointnet(proprio_dim_size=8)  # TODO: don't hardcode
        self._num_points = 5000
        self._point_cloud_encoder = SimplePointNet(6+9, 32)  # TODO: not hard-coded
        self._policy_encoder = nn.Sequential(nn.Linear(32, 32),  # +8
                                             nn.ReLU(),
                                             nn.Linear(32, self.num_actions))
        #self._end_effector_rot_encoder = nn.Sequential(nn.Linear(3, 32),
        #                                                 nn.ReLU(),
        #                                                 nn.Linear(32, 4))  # TODO:

        #self._actor = Actor(observation_space=observation_space, nb_actions=self.num_actions, use_clip=model_flags.use_clip)
        #self._critic = Critic(observation_space=observation_space, nb_actions=self.num_actions, use_clip=model_flags.use_clip)  # TODO: not really used

        self._camera = None  # TODO: assuming we're holding the camera fixed
        self._voxel_size = 0.01
        self._num_downsampled_voxel_points = 10000

        self.register_buffer("out_mean_sum", torch.zeros((self.num_actions,)))
        self.register_buffer("out_square_sum", torch.zeros((self.num_actions,)))
        self.register_buffer("out_count", torch.zeros(()).fill_(1e-8))

    def update_running_stats(self, dataset):
        reshaped_actions = dataset['action'].reshape((-1, self.num_actions))
        self.out_mean_sum += reshaped_actions.mean(0)
        self.out_square_sum += (reshaped_actions ** 2).mean(0)  # TODO: this is probably wrong
        self.out_count += 1  # TODO: alternatively batch size...HACKY/temp

    def get_out_mean(self):
        return self.out_mean_sum / self.out_count

    def get_out_std(self):
        #return self.out_std_sum / self.out_count
        return torch.sqrt(torch.clip(self.out_square_sum/self.out_count - self.out_mean_sum**2/(self.out_count**2), 0))  # TODO: sometimes floating point errors make this negative (are there other cases?)

    def initial_state(self, batch_size):
        assert not self.use_lstm, "LSTM not currently implemented. Ensure this gets initialized correctly when it is" \
                                  "implemented."
        return tuple()

    def actor_parameters(self):
        parameters = list(self._point_cloud_encoder.parameters())
        parameters.extend(list(self._policy_encoder.parameters()))
        return parameters

    def critic_parameters(self):
        return [] #self._critic.parameters()

    def _normalize_observation(self, observation, obs_low, obs_high):  # TODO: de-dupe
        observation = torch.flatten(observation, 0, 1)  # Merge time and batch.
        observation = torch.flatten(observation, 1, 2)  # Merge stacked frames and channels.
        observation = observation.float()

        obs_high = torch.tensor(obs_high).to(device=observation.device)
        obs_low = torch.tensor(obs_low).to(device=observation.device)

        observation = (observation - obs_low) / (obs_high - obs_low)
        return observation

    def _normalize_all_observations(self, inputs, observation_space):
        if isinstance(observation_space, gym.spaces.Dict):
            observation = {}

            for key in observation_space.spaces.keys():
                if key != "state_vector":  # TODO: temp for testing, state_vector stuff
                    observation[key] = self._normalize_observation(inputs[key], observation_space[key].low, observation_space[key].high)
                else:
                    # TODO: temp! De-dupe with normalize if I keep
                    print("Reminder: state vector not normalized")
                    key_obs = inputs[key]
                    key_obs = torch.flatten(key_obs, 0, 1)  # Merge time and batch.
                    key_obs = torch.flatten(key_obs, 1, 2)  # Merge stacked frames and channels.
                    key_obs = key_obs.float()
                    observation[key] = key_obs

                # TODO for testing, 0 out the image so we're only using the state vector
                #if key == "image":
                #    observation[key] *= 0
        else:
            observation = self._normalize_observation(inputs['frame'], observation_space.low, observation_space.high)

        return observation

    def _create_camera(self, k):
        # TODO: non-hardcoded height and width
        height = 720
        width = 1280
        #height = 1280  # TODO...?
        #width = 720

        # From RosCamera definition
        near_val = 0.0
        far_val = 1.0
        pos = None
        orn = None
        pose_matrix = None
        fov = None

        fx = k[0, 0]
        fy = k[1, 1]
        px = k[0, 2]
        py = k[1, 2]

        # The Nones are not being used by Camera, so not bothering for the moment (TODO)
        camera = Camera(pos, orn, height, width, fx, fy, px, py, near_val, far_val=far_val, pose_matrix=pose_matrix,
                 proj_matrix=None, view_matrix=None, fov=fov)
        return camera

    def forward(self, inputs, action_space_id, core_state=(), action=None):
        T, B, *_ = inputs['image'].shape
        inputs = self._normalize_all_observations(inputs, self._observation_space)

        # TODO: assuming the breakdown of states as given in stretch_demo_env: construct_observation
        state_index = 0
        state = inputs['state_vector'][:, state_index:state_index+9]

        if self._camera is None:
            # TODO: like nothing is used...?
            state_index += 9
            camera_d = inputs['state_vector'][:, state_index:state_index+5]
            state_index += 5
            camera_k = inputs['state_vector'][:, state_index:state_index+9].reshape((-1, 3, 3))
            state_index += 9
            camera_r = inputs['state_vector'][:, state_index:state_index+9].reshape((-1, 3, 3))
            state_index += 9
            camera_p = inputs['state_vector'][:, state_index:state_index+12].reshape((-1, 3, 4))
            state_index += 12
            camera_pose = inputs['state_vector'][:, state_index:state_index+16].reshape((-1, 1, 4, 4))
            self._camera = self._create_camera(camera_k[0].cpu().numpy())  # TODO: assumes all cameras in the batch are the same...
            self._camera.camera_r = camera_r[0]  # TODO: temp hacky
            self._camera.camera_pose = camera_pose[0].squeeze(0)  # TODO: temp hacky

        if action is None:
            all_batch_ids = []
            all_xyzs = []
            all_colors = []
            #all_goal_xyz = []
            for batch_id in range(state.shape[0]): # TODO spowers TEMP FOR TESTING state.shape[0]):
                color = inputs['image'][batch_id, :3, :, :]
                depth = inputs['image'][batch_id, 3:4, :, :]
                batch_state = state[batch_id]

                # Rotate the input, because the camera itself is rotated. This is necessary to work with the camera parameters correctly (TODO: check rotation direction)
                color = torchvision.transforms.functional.rotate(color, 90, expand=True).permute(1, 2, 0)
                color = torch.cat((color, torch.tile(batch_state.unsqueeze(0).unsqueeze(0), (*color.shape[:2], 1))), axis=-1)   # Early fusion
                depth = torchvision.transforms.functional.rotate(depth, 90, expand=True).squeeze(0)
                depth = depth.cpu().numpy() * 2 ** 16 / 10000
                depth = self._camera.fix_depth(depth)

                #xyz = depth_to_xyz(depth.cpu().numpy() * 2**16/10000, self._camera)  # TODO: don't hard-code this conversion here
                xyz = depth_to_xyz(depth, self._camera)
                #xyz = add_additive_noise_to_xyz(xyz)  # TODO spowers TEMP FOR TESTING
                xyz_flat = xyz.reshape((-1, 3))
                color_flat = color.reshape((-1, color.shape[-1]))  # TODO: check if mirrored!  The permutes are due to an inconsistency between the output image shape and what the camera thinks it's outputting (TODO) It's because the camera is rotated 90...

                indices = np.arange(len(xyz_flat))
                np.random.shuffle(indices)  # TODO spowers TEMP FOR TESTING
                subsample_indices = indices[:self._num_points]

                #pcd = get_pcd(xyz_flat, color_flat)
                #pcd_downsampled = pcd.voxel_down_sample(self._voxel_size)
                #color_downsampled = np.asarray(pcd_downsampled.colors)
                #xyz_downsampled = np.asarray(pcd_downsampled.points)
                xyz_downsampled = xyz_flat[subsample_indices]
                color_downsampled = color_flat[subsample_indices]
                transformed_xyz = trimesh.transform_points(xyz_downsampled @ self._camera.camera_r.T.cpu().numpy(), self._camera.camera_pose.cpu().numpy())

                batch_map_indices = [batch_id for _ in range(len(transformed_xyz))]

                all_batch_ids.extend(batch_map_indices)
                all_xyzs.extend(transformed_xyz)
                all_colors.extend(color_downsampled)

            all_colors = torch.stack(all_colors)
            all_xyzs = torch.tensor(np.array(all_xyzs)).to(all_colors.device)
            batch_ids = torch.tensor(all_batch_ids).to(all_colors.device)
            encoding = self._point_cloud_encoder(all_colors, all_xyzs.float(), batch_ids)

            #encoding_with_state = torch.cat((encoding, state), axis=-1)
            encoding_with_state = encoding
            #encoding_with_state = state
            raw_action = self._policy_encoder(encoding_with_state)

            # Scale the action to the range expected by the environment (Pytorch-DDPG does this in an environment wrapper)...TODO
            # TODO: handle (-inf, inf) action spaces
            if self._model_flags.use_running_stats:
                action = self.get_out_mean() + self.get_out_std() * raw_action  # TODO...inherit from Impala?
            else:
                action = torch.clip(raw_action, -1., 1.)
                action_scale = (self._action_spaces[action_space_id].high - self._action_spaces[action_space_id].low) / 2.
                action_scale = torch.tensor(action_scale).to(action.device)
                action_bias = (self._action_spaces[action_space_id].high + self._action_spaces[action_space_id].low) / 2.
                action_bias = torch.tensor(action_bias).to(action.device)
                action = action_scale * action + action_bias
        else:
            action = action.flatten(0, 1)  # TODO double check

        q_batch = torch.zeros((T, B))  # TODO: temp and hacky. Unused
        action = action.view(T, B, self.num_actions).float()
        policy_logits = action  # TODO... not accurate, but also not necessary (as it currently is...)

        return (
            dict(baseline=q_batch, action=action, policy_logits=policy_logits),
            core_state,
        )


class TransporterImpalaNet(ImpalaNet):
    def __init__(self, observation_space, action_spaces, model_flags, conv_net=None):
        super().__init__(observation_space, action_spaces, model_flags, conv_net, skip_net_init=True)
        self._observation_space = observation_space
        self._action_spaces = action_spaces

        first_action_space = list(action_spaces.values())[0]
        self.num_actions = first_action_space.shape[0]

        #self.agent = GoalTransporterAgent(name="transporter_net", task=None, root_dir=model_flags.output_dir, learning_rate=model_flags.actor_learning_rate)
        cfg = RavensDemonstrationEnv.construct_cfg()
        #self.agent = ImageGoalTransporterAgent(name="transporter_net", cfg=cfg, train_ds=None, test_ds=None)
        #self.agent = FullAttentionTransporterAgent(name="transporter_net", cfg=cfg, train_ds=None, test_ds=None)
        #self.agent = TwoStreamClipUNetLatTransporterAgent(name="transporter_net", cfg=cfg, train_ds=None, test_ds=None)
        self.agent = ClipUNetTransporterAgent(name="transporter_net", cfg=cfg, train_ds=None, test_ds=None)
        #self.agent = TwoStreamClipLingUNetLatTransporterAgent(name="transporter_net", cfg=cfg, train_ds=None, test_ds=None)

    def parameters(self):
        return self.agent.parameters()

    # TODO:  move these conversion methods to a standard place
    def _convert_dict_to_unified_action(self, dict_action):
        unified_action = []
        for pose_id in ("pose0", "pose1"):
            for space in dict_action[pose_id]:
                unified_action.append(space)

        return torch.tensor(np.concatenate(unified_action))  # TODO: this really shouldn't convert to torch here, but it is very convenient

    def _convert_aggregated_images_to_per_camera_data(self, image):
        all_color_data = []
        all_depth_data = []

        for camera_id in range(3):
            start_id = 4 * camera_id
            all_color_data.append(image[start_id:start_id+3, :, :].cpu().numpy().transpose(1, 2, 0))
            all_depth_data.append(image[start_id+3:start_id+4, :, :].cpu().numpy().squeeze(0))

        return all_color_data, all_depth_data

    def forward(self, inputs, action_space_id, core_state=(), action=None):
        # TODO: ravens_torch doesn't currently support batching
        squeezed_inputs = inputs["image"].squeeze(0).squeeze(0).squeeze(0).permute(1, 2, 0)
        #assert squeezed_inputs.shape[0] == 12 #24
        #image_inputs = squeezed_inputs[:6, :, :]
        #goal_inputs = squeezed_inputs[6:, :, :]

        """all_color_data, all_depth_data = self._convert_aggregated_images_to_per_camera_data(image_inputs)
        inputs["color"] = all_color_data
        inputs["depth"] = all_depth_data

        goal_dict = {}
        all_goal_color_data, all_goal_depth_data = self._convert_aggregated_images_to_per_camera_data(goal_inputs)
        goal_dict["color"] = all_goal_color_data
        goal_dict["depth"] = all_goal_depth_data"""

        action = self.agent.act(squeezed_inputs)
        action = self._convert_dict_to_unified_action(action)
        q_batch = torch.zeros((1,))
        policy_logits = torch.zeros(action.shape)

        return (
            dict(baseline=q_batch, action=action, policy_logits=policy_logits),
            core_state,
        )
