import torch.nn as nn
from torch_ac.model import ACModel
from torch.distributions.categorical import Categorical


class ActorCritic(nn.Module, ACModel):
    """
    Using the architecture described in Human-level control through deep reinforcement learning (Mnih et al, 2015)
    https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
    adapted to actor/critic.
    """
    def __init__(self, action_space):
        super().__init__()

        # Expecting time_batch_size to be 4
        self._embedding = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU()
        )

        self._actor = nn.Linear(512, action_space)
        self._critic = nn.Linear(512, action_space)

    def forward(self, observation, task_action_count):
        embedding = self._embedding(observation)
        actor_result = self._actor(embedding)
        critic_result = self._critic(embedding)

        distribution = Categorical(logits=actor_result[:task_action_count])  # TODO: F.log_softmax()?

        return distribution, critic_result
