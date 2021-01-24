import torch
import torch.nn as nn
import numpy as np
import uuid
import torch.utils.checkpoint
from torch.distributions.categorical import Categorical
from continual_rl.policies.sane.hypothesis.core_accessor import CoreAccessor
from continual_rl.utils.common_nets import get_network_for_size
from continual_rl.policies.sane.hypothesis.replay_buffer import ReplayBufferFileBacked


class InputScaler(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        self._observation_space = observation_space

    def forward(self, x):
        return x.float() / self._observation_space.high


class ConvNetTimeWrapper(nn.Module):
    def __init__(self, observation_size):
        super().__init__()
        input_size = list(observation_size)
        input_size = [input_size[0] * input_size[1], input_size[2], input_size[3]]
        self._conv_net = get_network_for_size(input_size)

        self.output_size = self._conv_net.output_size

    def forward(self, x):
        # Combine time and channels
        x = x.view((x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        return self._conv_net(x)


class Hypothesis(nn.Module):
    """
    The hypothesis holding:
    1. The pattern filter (does this hypothesis apply?)
    2. The policy (what do we do if the hypothesis does apply)
    3. The value (the expected value of this state: a bias for policy training)
    4. The replay buffer (containing old examples, along with the impact they had on the filter value)

    The hypothesis requires two forms of training:
    1. Training the pattern filter according to what's stored in the replay buffer (train_pattern_filter)
    2. Training the policy (and value) according to the environment - this happens externally, but provides the
        information that the pattern filter trainer uses to evaluate each input
    """

    def __init__(self, config, device, master_device, layer_id, output_dir, input_space, output_size,
                 replay_buffer_size,
                 filter_learning_rate, parent_hypothesis, pattern_filter=None, policy=None):
        super().__init__()

        # ========== Parameters that are SHARED or STATIC. Available on CORE, TRAIN, and USAGE processes ============
        self._config = config
        self._device = device
        self._master_device = master_device
        self._output_dir = output_dir
        self._input_space = input_space
        self._output_size = output_size
        self._replay_buffer_size = replay_buffer_size
        self._filter_learning_rate = filter_learning_rate
        self._tanh = nn.Tanh()
        self._sigmoid = nn.Sigmoid()

        self.unique_id = uuid.uuid4()
        self.is_long_term = False
        self.is_prototype = False
        self.layer_id = layer_id

        self._eps = 1e-6
        intermediate_dim = 128  # 64

        if pattern_filter is not None:
            self.pattern_filter = pattern_filter
        else:
            preprocessor = ConvNetTimeWrapper(self._input_space.shape)

            # TODO: these added linear layers are NOT necessary, they are here for consistency with what's in the paper
            # (The smaller nets did not have time to run before the ICLR deadline)
            self.pattern_filter = nn.Sequential(
                InputScaler(input_space),
                preprocessor,
                #nn.Linear(preprocessor.output_size, intermediate_dim),
                #nn.ReLU(),
                #nn.Linear(intermediate_dim, intermediate_dim),
                nn.ReLU(),
                nn.Linear(preprocessor.output_size, 2))  # Mean, error
            self.pattern_filter.apply(lambda m: self._weights_init_normal(m, weight_mean=0, weight_std=0.001))

        # This is the consequent of the hypothesis. What happens when it fires.
        if policy is not None:
            self._policy = policy
            self._policy.requires_grad = True  # In case the input policy doesn't
        else:
            self._policy = nn.Parameter(torch.Tensor([1e-1 for _ in range(output_size)]))

        # ========= Parameters that are used on the USAGE process. They are NOT shared with TRAIN process, but do get communicated over to CORE ============
        # All of these are currently serving two purposes: on the USAGE process these are deltas (get reset to 0, count the total usages since the process was created)
        # On the CORE process, they are the cumulative counts: the sums across all the USAGE processes. TODO: two sets of variables
        self.usage_count = 0
        self.usage_count_since_last_update = 0
        self.non_decayed_usage_count = 0  # TODO: not a good name, but the good names are taken. Rename stuff
        self.usage_count_since_creation = 0

        # ========== Parameters that are used on the CORE process. They are NOT shared, but they do get communicated over to USAGE ============
        # The hypotheses associated with this one:
        # Child hypothesis: the hypothesis that gets used when an input falls into this hypothesis's lower half (TODO clearer name?)
        # Parent hypothesis: the hypothesis for which this hypothesis is a child
        self.parent_hypothesis = None
        if parent_hypothesis is not None:
            parent_hypothesis.add_short_term(self)  # Sets parent_hypothesis

        # ========== Parameters that are used on the TRAIN process. These get communicated from USAGE->CORE->TRAIN ============
        # This currently initializes the replay_buffer, since the replay buffer encoder should be consistent with the pattern filter
        CoreAccessor.load_pattern_filter_from_state_dict(self,
                                                         self.pattern_filter.state_dict())  # TODO: this is admittedly hacky, clean it up if I keep it

        #self._replay_buffer = ReplayBuffer(non_permanent_maxlen=self._replay_buffer_size)
        #self._negative_examples = ReplayBuffer(non_permanent_maxlen=self._replay_buffer_size)
        self._replay_buffer = ReplayBufferFileBacked(maxlen=self._replay_buffer_size, observation_space=self._input_space,
                                                     large_file_path=config.large_file_path)
        #self._negative_examples = ReplayBufferFileBacked(maxlen=self._replay_buffer_size, observation_space=self._input_space,
        #                                             large_file_path=config.large_file_path)

        self._pattern_filter_optimizer = None
        self.replay_entries_since_last_train = 0
        self._pattern_filter_learner = None  # Actually does the learning, and then gets copied into the pattern filter

    @property
    def policy(self):
        policy = 3 * self._tanh(self._policy) if self._policy is not None else None  # self._policy #

        if self.is_prototype:
            policy = policy.detach()

        return policy

    @classmethod
    def get_preprocessor(cls, input_size):
        input_size = list(input_size)  # Tuple to the more convenient, here, list
        input_size = [input_size[0] * input_size[1], input_size[2], input_size[3]]
        preprocessor = get_network_for_size(input_size)
        return preprocessor, preprocessor.output_size

    def get_policy_with_entropy(self, x, detach_policy=False):
        if detach_policy:
            policy = self.policy.detach()
        else:
            policy = self.policy

        result = policy.to(self._device)

        return result

    @property
    def friendly_name(self):
        return str(self.unique_id)[:6]

    def get_policy_as_categorical(self, policy=None):  # TODO: find where hypothesis uses this and replace it
        # By default gets the "unaltered" policy, but if one is passed in, uses that one instead (i.e. where it's been
        # modified by an entropy scaling factor
        if policy is None:
            policy = self.policy
        return Categorical(logits=policy)

    def _init(self, module, gain=1):
        # No-op
        return module

    def _recursive_init(self, module, gain=1):
        for sub_module in module._modules.values():
            # If the sub_module has sub_modules, recurse
            if isinstance(sub_module, nn.Linear) or isinstance(sub_module, nn.Conv2d):
                self._init(sub_module, gain)
            elif "_modules" in sub_module.__dict__ and len(sub_module._modules) > 0:
                self._recursive_init(sub_module, gain=gain)

        return module

    def _weights_init_normal(self, m, weight_mean, weight_std):
        """
        Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.
        Courtesy: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
        """
        classname = m.__class__.__name__
        # for every Linear layer in a model
        if isinstance(m, nn.Linear): #classname.find('Linear') != -1:
            y = m.in_features
            m.weight.data.normal_(weight_mean/np.sqrt(y), weight_std/np.sqrt(y))

    def parameters(self):
        policy_parameters = [self._policy] #list(super(Hypothesis, self).parameters())
        filter_parameters = [] #list(super(Hypothesis, self).parameters())

        return policy_parameters, filter_parameters

    def share_parameters_memory(self):
        parameters = list(super(Hypothesis, self).parameters())
        for parameter in parameters:
            parameter.share_memory_()

