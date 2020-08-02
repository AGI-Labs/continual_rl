import torch
import torch.nn as nn
from torch_ac.model import ACModel
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from continual_rl.utils.common_exceptions import ObservationShapeNotRecognized


class AdaptiveConvNet(nn.Module):
    """
    Allows for images larger than their stated minimums, and will auto-compute the output size accordingly
    """
    def __init__(self, conv_net):
        super().__init__()
        self._conv_net = conv_net

    def compute_output_size(self, observation_size):
        dummy_input = torch.zeros(observation_size).unsqueeze(0)  # Observation size doesn't include batch, so add it
        dummy_output = self._conv_net(dummy_input).squeeze(0)  # Remove batch
        return dummy_output.shape[0]


class ConvNet7x7(AdaptiveConvNet):
    def __init__(self):

        # ConvNet structure from: https://github.com/lcswillems/rl-starter-files/blob/master/model.py
        conv_net = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten()
        )
        super().__init__(conv_net)

    def forward(self, observation):
        return self._conv_net(observation)


class ConvNet84x84(AdaptiveConvNet):
    def __init__(self):
        # Expecting time_batch_size to be 4
        conv_net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),  # 64 * 7 * prevents the Adaptive from really enabling larger images
            nn.ReLU()
        )
        super().__init__(conv_net)

    def forward(self, observation):
        return self._conv_net(observation)


class ActorCritic(nn.Module, ACModel):
    """
    Using the architecture described in Human-level control through deep reinforcement learning (Mnih et al, 2015)
    https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
    adapted to actor/critic.
    """
    def __init__(self, observation_space, action_space):
        super().__init__()

        if observation_space == [4, 84, 84]:
            self._embedding = ConvNet84x84()
        elif observation_space == [3, 7, 7]:
            self._embedding = ConvNet7x7()
        else:
            # The ConvNets are intended to be adaptive, to allow for larger sizes. If/when applicable, check
            # accordingly and use the correct one. (84x84 will require some modification.) For now just being strict.
            raise ObservationShapeNotRecognized(f"Observation shape {observation_space[1:]} not found")

        output_size = self._embedding.compute_output_size(observation_space)

        self._actor = nn.Sequential(
            nn.Linear(output_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space)
        )
        self._critic = nn.Sequential(
            nn.Linear(output_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self._task_action_count = None

    def set_task_action_count(self, task_action_count):
        # I really don't like this, but PPOAlgo assumes the inputs to forward are just (observation), so this
        # will do for now, until possibly a re-implementation of PPO, or at least pulling it into the repo and changing it.
        self._task_action_count = task_action_count

    def forward(self, observation, task_action_count=None):
        if task_action_count is None:
            task_action_count = self._task_action_count

        embedding = self._embedding(observation)
        actor_result = self._actor(embedding)
        critic_result = self._critic(embedding)

        distribution = Categorical(logits=F.log_softmax(actor_result[:, :task_action_count], dim=1))

        return distribution, critic_result
