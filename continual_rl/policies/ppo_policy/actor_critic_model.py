import torch.nn as nn
from torch_ac.model import ACModel
from torch.distributions.categorical import Categorical
from continual_rl.utils.common_exceptions import ObservationShapeNotRecognized


class ConvNet7x7(nn.Module):  # TODO: lazily using the existing one as a base, extract out
    def __init__(self):
        super().__init__()
        self.output_size = 32  # Consistent, I think, with the minigrid rl benchmark

        # From: https://github.com/lcswillems/rl-starter-files/blob/master/model.py
        # TODO: temp, or include license?
        self._conv_net = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),  # 32
            nn.ReLU(),
            nn.Conv2d(32, 32, (2, 2)),  # 32, 64
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, observation):
        return self._conv_net(observation)


class ConvNet84x84(nn.Module):  # TODO: lazily using the existing one as a base, extract out
    def __init__(self):
        super().__init__()
        self.output_size = 512

        # Expecting time_batch_size to be 4
        self._conv_net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU()
        )

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
            raise ObservationShapeNotRecognized(f"Observation shape {observation_space[1:]} not found")

        self._actor = nn.Linear(self._embedding.output_size, action_space)
        self._critic = nn.Linear(self._embedding.output_size, 1)

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

        distribution = Categorical(logits=actor_result[:, :task_action_count])  # TODO: F.log_softmax()?

        return distribution, critic_result
