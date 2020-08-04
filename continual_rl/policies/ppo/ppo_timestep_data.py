import torch
from continual_rl.policies.timestep_data_base import TimestepDataBase


class PPOTimestepDataSingle(TimestepDataBase):

    def __init__(self, observation, action, value, log_prob, task_action_count):
        super().__init__()

        self.observation = observation
        self.action = action
        self.value = value
        self.log_prob = log_prob
        self.task_action_count = task_action_count

    def to_tensors(self, use_cuda):
        # Given just an integer, torch.Tensor(x) will create a tensor of size x, not with content x.
        # So it has to be in a list, even if we don't want that dimension in the Tensor.
        # To be safe, just listify each then remove that dim.
        # All other inputs should already be Tensors, so just converting the two that were populated outside of PPO
        self.reward = torch.Tensor([self.reward]).squeeze(0)
        self.done = torch.Tensor([self.done]).squeeze(0)

        if use_cuda:
            self.observation = self.observation.cuda()
            self.action = self.action.cuda()
            self.value = self.value.cuda()
            self.log_prob = self.log_prob.cuda()
            self.reward = self.reward.cuda()
            self.done = self.done.cuda()

        return self


class PPOTimestepDataBatch(TimestepDataBase):

    def __init__(self, observations, actions, values, log_probs, task_action_count):
        super().__init__()

        self.observations = observations
        self.actions = actions
        self.values = values
        self.log_probs = log_probs

        # Not a list, just a single value, since it's consistent across the collection
        self.task_action_count = task_action_count

    def regroup_by_env(self):
        """
        Since this TimestepData contains multiple envs' worth of data, regroup by env, storing each one in a
        PPOTimestepDataSingle
        """
        assert len(self.reward) == len(self.done) == len(self.actions) == len(self.values) == len(self.log_probs), \
            "All entries should be storing the same amount of data"

        per_env_data = []

        for env_id in range(len(self.reward)):
            single_env_timestep_data = PPOTimestepDataSingle(self.observations[env_id],
                                                             self.actions[env_id],
                                                             self.values[env_id],
                                                             self.log_probs[env_id],
                                                             self.task_action_count)
            single_env_timestep_data.reward = self.reward[env_id]
            single_env_timestep_data.done = self.done[env_id]

            per_env_data.append(single_env_timestep_data)

        return per_env_data
