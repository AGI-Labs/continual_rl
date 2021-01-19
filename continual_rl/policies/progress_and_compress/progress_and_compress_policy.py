import torch
from torch import nn
from continual_rl.utils.common_nets import get_network_for_size, CommonConv
from continual_rl.policies.impala.nets import ImpalaNet
from continual_rl.policies.ewc.ewc_policy import EWCPolicy
from continual_rl.policies.progress_and_compress.progress_and_compress_monobeast import ProgressAndCompressMonobeast
from continual_rl.policies.progress_and_compress.progress_and_compress_policy_config import ProgressAndCompressPolicyConfig


class ModuleNotAdaptedException(Exception):
    pass


class ElementwiseScaleModule(nn.Module):
    def __init__(self, num_elements):
        super().__init__()
        self._scale = nn.Parameter(torch.zeros(size=(num_elements,)).uniform_(0, 0.1), requires_grad=True)

    def forward(self, input):
        if len(input.shape) == 4:
            output = self._scale.unsqueeze(1).unsqueeze(1) * input
        else:
            output = self._scale * input

        return output


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
        self._adaptors = {}
        self._adaptor_params = nn.ParameterList()
        self.output_size = self._conv_net.output_size

        # When doing a forward pass, we'll grab the per-layer result from the KB, and incorporate it in (via the hook),
        # using the adaptors created here.
        for module_name, module in self._conv_net.named_modules():
            module.register_forward_hook(self._create_incorporate_knowledge_base_hook(module_name))

            # Create the adaptor and ensure its parameters get properly registered (just stored in a normal dict won't work)
            adaptor = self._create_adaptor(module)
            self._adaptors[module_name] = adaptor  # If it's None, save it anyway so we know to no-op
            if adaptor is not None:
                self._adaptor_params.extend(adaptor.parameters())

    def _reset_layer(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.reset_parameters()

    def reset(self):
        # Note: reset is only applied to Linear and Conv2D layers
        # TODO: reset adaptors?
        self._conv_net.apply(self._reset_layer)

    def _create_incorporate_knowledge_base_hook(self, module_name):
        """
        The module doesn't conveniently provide its own name, as far as I can tell, so pass it in.
        """
        def hook(module, input, output):
            result = output

            # Apply adaptor to knowledge base outputs
            adaptor = self._adaptors[module_name]
            if adaptor is not None:
                # Assuming KB outputs have already had V_i and c_i applied (via KB layer weights) - TODO, unclear if accurate
                knowledge_base_outputs = self._knowledge_base.latest_layerwise_outputs[module_name]
                adapted_knowledge = adaptor(knowledge_base_outputs)

                # Then add the active column outputs (which have already had weight + bias applied)
                result = output + adapted_knowledge

            return result
        return hook

    def _create_adaptor(self, module):
        nonlinearity = nn.ReLU()  # TODO: check

        if isinstance(module, nn.Conv2d):
            # U_i in the paper. It's ambiguous to me whether this should be two layer (V_i, c_i) or if those
            # are the KB weights themselves. Currently treating it as the second (TODO)
            adaptor = nn.Conv2d(kernel_size=(1, 1),
                                      in_channels=module.out_channels,
                                      out_channels=module.out_channels, bias=False)

            # TODO: this will broadcast its out_channel-length vector to the requisite shape. Should it actually
            # have the *full* shape? Not broadcasted?
            alpha = ElementwiseScaleModule(module.out_channels)
            full_adaptor = nn.Sequential(nonlinearity,
                                         adaptor,
                                         alpha)
        elif isinstance(module, nn.Linear):
            adaptor = nn.Linear(in_features=module.out_features, out_features=module.out_features)
            alpha = ElementwiseScaleModule(module.out_features)
            full_adaptor = nn.Sequential(nonlinearity,
                                         adaptor,
                                         alpha)

        elif isinstance(module, nn.ReLU) or isinstance(module, nn.Flatten) or isinstance(module, CommonConv) or \
                isinstance(module, nn.Sequential):
            # Capture everything we know should no-op. This is so if we do add another new layer, we know to adapt it too
            # CommonConv and Sequential are both wrappers; the actual adaptors will be created for their inner modules
            full_adaptor = None  # Don't add in the KB at this point

        else:
            raise ModuleNotAdaptedException(f"Module of type {type(module)} not adapted. Add the intended adaption method to _create_adaptor")

        return full_adaptor

    def forward(self, input):
        with torch.no_grad():
            # This will cause the knowledge base to update its layerwise computations, in latest_layerwise_outputs,
            # which gets incorporated in the active column's forward hook
            self._knowledge_base(input)

        active_column_output = self._conv_net(input)
        return active_column_output


class KnowledgeBaseColumnNet(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        combined_observation_size = [observation_space.shape[0] * observation_space.shape[1],
                                     observation_space.shape[2],
                                     observation_space.shape[3]]
        self._conv_net = get_network_for_size(combined_observation_size)
        self.output_size = self._conv_net.output_size
        self.latest_layerwise_outputs = {}

        for module_name, module in self._conv_net.named_modules():
            module.register_forward_hook(self.create_save_output_hook(module_name))

    def create_save_output_hook(self, module_name):
        def hook(module, input, output):
            self.latest_layerwise_outputs[module_name] = output
        return hook

    def forward(self, input):
        return self._conv_net(input)


class KnowledgeBaseImpalaNet(ImpalaNet):
    def __init__(self, knowledge_base_column, observation_space, action_space, use_lstm):
        super().__init__(observation_space, action_space, use_lstm, conv_net=knowledge_base_column)


class ProgressAndCompressNet(ImpalaNet):
    def __init__(self, observation_space, action_space, use_lstm):
        knowledge_base_column = KnowledgeBaseColumnNet(observation_space)
        active_column = ActiveColumnNet(observation_space, knowledge_base_column)
        super().__init__(observation_space, action_space, use_lstm, conv_net=active_column)

        # Have to set these after the super, otherwise it gets mad that we're creating parameters too soon
        # We additionally wrap the KB so we can get a policy from it and train it directly (EWC + KL)
        self.knowledge_base = KnowledgeBaseImpalaNet(knowledge_base_column, observation_space, action_space, use_lstm)
        self._active_column = active_column

    def reset_active_column(self):
        self._active_column.reset()


class ProgressAndCompressPolicy(EWCPolicy):
    """
    Based on Progress & Compress, as described here: https://arxiv.org/pdf/1805.06370.pdf

    Things necessary:
    1. Two networks, where the second one (active layer) gets the layerwise results of the first (KB) merged in according to eqn (1)
    2. Train the active normally (w/o updating KB) via IMPALA
    3. Use Online EWC to update the KB parameters + KL div to active
    """
    def __init__(self, config: ProgressAndCompressPolicyConfig, observation_space, action_spaces):
        super().__init__(config, observation_space, action_spaces, policy_net_class=ProgressAndCompressNet,
                         impala_class=ProgressAndCompressMonobeast)
        self._current_task_id = None

    def set_task_id(self, task_id):
        super().set_task_id(task_id)

        if self._current_task_id != task_id:
            self.impala_trainer.model.reset_active_column()
            self.impala_trainer.learner_model.reset_active_column()  # TODO: I think technically only this one is necessary?
            self._current_task_id = task_id