from torch import nn
from continual_rl.utils.common_nets import get_network_for_size
from continual_rl.policies.impala.nets import ImpalaNet
from continual_rl.policies.ewc.ewc_policy import EWCPolicy
from continual_rl.policies.progress_and_compress.progress_and_compress_policy_config import ProgressAndCompressPolicyConfig


class ActiveColumnNet(nn.Module):
    """
    This network not only has a column of its own, it also incorporates the layer-wise results from the KB column
    into itself.
    """
    def __init__(self, observation_space, knowledge_base_column):
        super().__init__()
        # The conv net gets channels and time merged together (mimicking the original FrameStacking)
        combined_observation_size = [observation_space.shape[0] * observation_space.shape[1],
                                     observation_space.shape[2],
                                     observation_space.shape[3]]
        self._conv_net = get_network_for_size(combined_observation_size)
        self._knowledge_base = knowledge_base_column
        self.output_size = self._conv_net.output_size

        for module_name, module in self._conv_net.named_modules():
            module.register_forward_hook(self.incorporate_knowledge_base_hook)

    def incorporate_knowledge_base_hook(self, module, input, output):
        outputs = self._knowledge_base.latest_layerwise_outputs
        pass

    def forward(self, input):
        knowledge_base_output = self._knowledge_base(input)
        active_column_output = self._conv_net(input)


class KnowledgeBaseNet(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        combined_observation_size = [observation_space.shape[0] * observation_space.shape[1],
                                     observation_space.shape[2],
                                     observation_space.shape[3]]
        self._conv_net = get_network_for_size(combined_observation_size)
        self.latest_layerwise_outputs = {}

        for module_name, module in self._conv_net.named_modules():
            module.register_forward_hook(self.save_output_hook)

    def save_output_hook(self, module, input, output):
        self.latest_layerwise_outputs[module] = output
        pass

    def forward(self, input):
        return self._conv_net(input)


class ProgressAndCompressNet(ImpalaNet):
    def __init__(self, observation_space, action_space, use_lstm):
        knowledge_base_column = KnowledgeBaseNet(observation_space)
        active_column = ActiveColumnNet(observation_space, knowledge_base_column)
        super().__init__(observation_space, action_space, use_lstm, conv_net=active_column)


class ProgressAndCompressPolicy(EWCPolicy):
    """
    Based on Progress & Compress, as described here: https://arxiv.org/pdf/1805.06370.pdf

    Things necessary:
    1. Two networks, where the second one (active layer) gets the layerwise results of the first (KB) merged in according to eqn (1)
    2. Train the active normally (w/o updating KB) via IMPALA
    3. Use Online EWC to update the KB parameters + KL div to active
    """
    def __init__(self, config: ProgressAndCompressPolicyConfig, observation_space, action_spaces):  # Switch to your config type
        super().__init__(config, observation_space, action_spaces, policy_net_class=ProgressAndCompressNet)
