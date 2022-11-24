import torch
import copy
from torch import nn
from continual_rl.utils.common_nets import get_network_for_size, CommonConv
from continual_rl.policies.impala.nets import ImpalaNet
from continual_rl.policies.ewc.ewc_policy import EWCPolicy
from continual_rl.policies.progress_and_compress.progress_and_compress_monobeast import ProgressAndCompressMonobeast
from continual_rl.policies.progress_and_compress.progress_and_compress_policy_config import ProgressAndCompressPolicyConfig
from continual_rl.utils.utils import Utils


class ModuleNotAdaptedException(Exception):
    pass


class ActiveColumnNet(ImpalaNet):
    """
    This network not only has a column of its own, it also incorporates the layer-wise results from the KB column
    into itself.
    Here's how it does it:
    1. Knowledge base is its own network, run separately (KnowledgeBaseColumnNet)
    2. When it's run, the KB column saves off its intermediate values (via a forward hook)
    3. When this module (ActiveColumnNet) is run, at each layer (via a forward hook) we look up the corresponding
    layer in the KB, run an adaptor on it, and add it in to this column's result
    All variable references are to eqn (1) in the P&C paper: https://arxiv.org/pdf/1805.06370.pdf
    """

    def __init__(self, observation_space, action_spaces, model_flags, knowledge_base_column):
        super().__init__(observation_space, action_spaces, model_flags)
        self._adaptors = {}
        self._adaptor_params = []
        self.eval_on_kb = None  # Gets set externally
        self.eval_is_stochastic = None  # Gets set externally

        # When doing a forward pass, we'll grab the per-layer result from the KB, and incorporate it in (via the hook),
        # using the adaptors created here. We skip the first (top) module because its input is the observation, not
        # a latent from the KB, so contains no consolidated information.
        first_skipped = False
        for module_name, module in self.named_modules():
            # Only register on leaf nodes (e.g. Conv2d, Linear, etc, where computation is actually taking place)
            if len(list(module.children())) == 0:
                if first_skipped:
                    # Create the adaptor and ensure its parameters get properly registered
                    adaptor = self._create_adaptor(module)
                    self._adaptors[module_name] = adaptor  # If it's None, save it anyway so we know to no-op
                    if adaptor is not None:
                        self._adaptor_params.extend(adaptor.parameters())

                    # Register afterwards because otherwise the copied module will have the hook too
                    module.register_forward_hook(self._create_incorporate_knowledge_base_hook(module_name))
                else:
                    first_skipped = True

        # Convert to ParameterList after we've adapted everything, so these aren't caught too.
        self._adaptor_params = nn.ParameterList(self._adaptor_params)

        # Also, assign knowledge base after creating hooks, so we're not adding hooks to it
        self._knowledge_base = knowledge_base_column

    def _reset_layer(self, module):
        # Don't reset knowledge base modules
        if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and not module in self._knowledge_base.modules():
            module.reset_parameters()

    def reset(self):
        # Note: reset is only applied to Linear and Conv2D layers, including adaptors
        self.apply(self._reset_layer)

    def _create_incorporate_knowledge_base_hook(self, module_name):
        """
        The module doesn't conveniently provide its own name, as far as I can tell, so pass it in.
        This is where we use the adaptor to incorporate knowledge base knowledge into the active column.
        """

        def hook(module, input, output):
            result = output

            # Apply adaptor to knowledge base outputs
            adaptor = self._adaptors[module_name]
            if adaptor is not None:
                # We adapt using the KB's previous layer. We do this by using the input to the layer that matches
                # the current one. We skip the first layer. See comment in __init__
                knowledge_base_inputs = self._knowledge_base.latest_layerwise_inputs[module_name][0]
                adapted_knowledge = adaptor(knowledge_base_inputs)

                # Then add the active column inputs to the next layer
                result = output + adapted_knowledge

            return result

        return hook

    def _create_adaptor(self, module):
        """
        This is adapting the previous layer of the KB to be merge-able into the next layer of the active column.
        The description of eqn (1) in the paper is a little vague. In particular, it is not very clear where, say,
        a convolution with stride > 1 would be applied. The way I'm interpreting it is that W_i and V_i are both
        convolutions that do the work of changing size (e.g. from 20x20x32 to 9x9x64), and U_i is just 1x1 on top.
        """

        # Section 2.1 of P&C, this is σ in equation (1) but not specified. Assuming ReLU as that is used in IMPALA.
        nonlinearity = nn.ReLU()

        if isinstance(module, nn.Conv2d):
            # Copy the module, so we have an adaptor that does the right input->output conversion
            cloned_module = copy.deepcopy(module)
            cloned_module.reset_parameters()

            full_adaptor = nn.Sequential(
                cloned_module,  # V_i, c_i. V_i can't simply be a 1x1, since someone needs to actually do the "real" conv
                nonlinearity,
                nn.Conv2d(kernel_size=(1, 1),  # Conv2d here represents U_i, a_i
                          in_channels=module.out_channels,
                          out_channels=module.out_channels, bias=True)
            )

        elif isinstance(module, nn.Linear):
            full_adaptor = nn.Sequential(
                nn.Linear(in_features=module.in_features, out_features=module.out_features, bias=True),  # V_i, c_i
                nonlinearity,
                nn.Linear(in_features=module.out_features, out_features=module.out_features, bias=True),  # U_i, a_i
            )

        elif isinstance(module, nn.ReLU) or isinstance(module, nn.Flatten) or isinstance(module, CommonConv) or \
                isinstance(module, nn.Sequential) or isinstance(module, ImpalaNet) or isinstance(module, nn.MaxPool2d):
            # Capture everything we know should no-op. This is so if we do add another new layer, we know to adapt it too
            # CommonConv and Sequential are both wrappers; the actual adaptors will be created for their inner modules
            full_adaptor = None  # Don't add in the KB at this point

        else:
            raise ModuleNotAdaptedException(f"Module of type {type(module)} not adapted. Add the intended adaption method to _create_adaptor")

        return full_adaptor

    def forward(self, input, action_space_id, core_state=()):
        with torch.no_grad():
            # PnC uses the training flag to say whether we should be using AC. However, in some cases we want the eval
            # to be non-deterministic (sometimes policies perform significantly better with a bit of randomness)
            # So in that case, set the train flag during the forward and no where else
            false_train_mode_on = False
            if self.eval_is_stochastic and not self.training:
                false_train_mode_on = True
                self.train()

            # This will cause the knowledge base to update its layerwise computations, in latest_layerwise_outputs,
            # which gets incorporated in the active column's forward hook
            column_output = self._knowledge_base(input, action_space_id)

            if false_train_mode_on:
                self.eval()

        # Note that during eval we only look at the output of the KB
        if self.training or not self.eval_on_kb:
            column_output = super().forward(input, action_space_id, core_state)

        return column_output


class KnowledgeBaseColumnNet(ImpalaNet):
    def __init__(self, observation_space, action_spaces, model_flags):
        super().__init__(observation_space, action_spaces, model_flags)
        self.latest_layerwise_inputs = {}

        for module_name, module in self.named_modules():
            module.register_forward_hook(self.create_save_output_hook(module_name))

    def create_save_output_hook(self, module_name):
        def hook(module, input, output):
            self.latest_layerwise_inputs[module_name] = input

        return hook


class ProgressAndCompressNet(nn.Module):
    """
    This class is a shadow of ImpalaNet (same API, so it can be used by Monobeast), but instead manages the
    two columns used by Progress and Compress.
    """
    def __init__(self, observation_space, action_spaces, model_flags):
        super().__init__()
        self.use_lstm = model_flags.use_lstm
        self.num_actions = Utils.get_max_discrete_action_space(action_spaces).n
        self.knowledge_base = KnowledgeBaseColumnNet(observation_space, action_spaces, model_flags)
        self._active_column = ActiveColumnNet(observation_space, action_spaces, model_flags, self.knowledge_base)

    def parameters(self):  # TODO: necessary at all? Probably not tbh
        parameters = []
        parameters.extend(self._active_column.parameters())
        parameters.extend(self.knowledge_base.parameters())
        return parameters

    def reset_active_column(self):
        self._active_column.reset()

    def forward(self, inputs, action_space_id, core_state=()):
        return self._active_column(inputs, action_space_id, core_state)

    def initial_state(self, batch_size):
        assert not self.use_lstm, "LSTM not currently implemented. Ensure this gets initialized correctly when it is" \
                                  "implemented."
        return tuple()


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
        # Rather than piping it all the way through, set it here
        self.impala_trainer.actor_model._active_column.eval_on_kb = config.eval_on_kb
        self.impala_trainer.actor_model._active_column.eval_is_stochastic = config.eval_is_stochastic
