import torch
import torch.nn as nn
import yaml
import os
from continual_rl.policies.impala.impala_policy import ImpalaPolicy
from continual_rl.policies.clear.clear_policy import ClearPolicy
from continual_rl.policies.clear.clear_policy_config import ClearPolicyConfig
from continual_rl.policies.impala.impala_policy_config import ImpalaPolicyConfig
from continual_rl.utils.utils import Utils
from hackrl.models.baseline import BaselineNet
from minihack.agent.polybeast.models.base import BaseNet
from hackrl.models.chaotic_dwarf import ChaoticDwarvenGPT5
import minihack
import hackrl

"""
TODO: this is not a permanent location nor design for how this will work. Just rapid prototyping, mostly
"""


class ConfigHolder:
    def __init__(self, **entries):
        self.__dict__.update(entries)

        if "msg" in entries:
            self.msg = ConfigHolder(**entries["msg"])

        self.restrict_action_space = False  # TODO: the restriction is breaking due to action space Discrete disconnect

    def __contains__(self, key):
        return key in self.__dict__

def create_impala_net(config):
    class MinihackImpalaNet(nn.Module):  # TODO: rename, and clarify that this is mimicking ImpalaNet
        def __init__(self, observation_space, action_spaces, use_lstm=False, conv_net=None):
            super().__init__()
            common_action_space = Utils.get_max_discrete_action_space(action_spaces)
            self.num_actions = common_action_space.n
            self.use_lstm = use_lstm

            # TODO: just for simplicity, using the existing config yaml. Move to be consistent with other config files
            if config.net_base == "baseline":
                config_path = os.path.join(os.path.dirname(hackrl.__file__), "models", "baseline.yaml")
            elif config.net_base == "nle_net":
                config_path = os.path.join(os.path.dirname(minihack.__file__), "agent", "polybeast", "config.yaml")
            elif config.net_base == "chaotic_dwarf":
                config_path = os.path.join(os.path.dirname(hackrl.__file__), "config.yaml")

            with open(config_path, "r") as config_file:
                try:
                    model_flags = ConfigHolder(**yaml.safe_load(config_file))
                except yaml.YAMLError as exc:
                    print(exc)
            print(f"device: {config.device}")

            # Minihack nets expect the "action space" to be a list of actions, not an OpenAI gym-like action space
            faked_action_space = list(range(common_action_space.n))
            if config.net_base == "baseline":
                self._model = BaselineNet(observation_space["glyphs"].shape, action_space=faked_action_space, flags=model_flags, device=config.device)
            elif config.net_base == "nle_net":
                self._model = BaseNet(observation_space["glyphs"].shape, num_actions=common_action_space.n, flags=model_flags, device='cpu') #config.device)
            elif config.net_base == "chaotic_dwarf":
                self._model = ChaoticDwarvenGPT5(observation_space["glyphs"].shape, action_space=faked_action_space, flags=model_flags, device='cpu')

            # Based on continual_rl/policies/impala/nets.py, which this network is mimicking
            # used by update_running_moments()
            # second moment is variance
            self.register_buffer("reward_sum", torch.zeros(()))
            self.register_buffer("reward_m2", torch.zeros(()))
            self.register_buffer("reward_count", torch.zeros(()).fill_(1e-8))

        def to(self, *args, **kwargs):
            super().to(*args, **kwargs)
            self._model.to(*args, **kwargs)  # Should happen automatically, but seems not to be, so...double checking, I guess
            return self
            
        def initial_state(self, batch_size=1):
            return self._model.initial_state(batch_size)

        def forward(self, inputs, action_space_id, core_state):
            #print(f"Inputs: {inputs.keys()}")
            #print(f"Inputs type: {inputs['glyphs'].device}")
            return self._model.forward(inputs, core_state)

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

    return MinihackImpalaNet


class MinihackImpalaPolicyConfig(ImpalaPolicyConfig):
    def __init__(self):
        super().__init__()
        self.net_base = "chaotic_dwarf"


class MinihackImpalaPolicy(ImpalaPolicy):
    def __init__(self, config: MinihackImpalaPolicyConfig, observation_space, action_spaces):
        policy_net_class = create_impala_net(config)
        super().__init__(config, observation_space, action_spaces, policy_net_class=policy_net_class)


class MinihackClearPolicyConfig(ClearPolicyConfig):
    def __init__(self):
        super().__init__()
        self.net_base = "chaotic_dwarf"


class MinihackClearPolicy(ClearPolicy):
    def __init__(self, config: MinihackClearPolicyConfig, observation_space, action_spaces):
        policy_net_class = create_impala_net(config)
        super().__init__(config, observation_space, action_spaces, policy_net_class=policy_net_class)
