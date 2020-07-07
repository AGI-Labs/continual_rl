from continual_rl.policies.info_to_store_base import InfoToStoreBase


class PPOInfoToStoreSingle(InfoToStoreBase):

    def __init__(self, observation, action, value, log_prob, task_action_count):
        super().__init__()

        self.observation = observation
        self.action = action
        self.value = value
        self.log_prob = log_prob
        self.task_action_count = task_action_count


class PPOInfoToStoreBatch(InfoToStoreBase):

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
        Since this InfoToStore contains multiple envs' worth of data, regroup by env, storing each one in a
        PPOInfoToStoreSingle
        """
        assert len(self.reward) == len(self.done) == len(self.actions) == len(self.values) == len(self.log_probs), \
            "All entries should be storing the same amount of data"

        per_env_data = []

        for env_id in range(len(self.reward)):
            single_env_info_to_store = PPOInfoToStoreSingle(self.observations[env_id],
                                                            self.actions[env_id],
                                                            self.values[env_id],
                                                            self.log_probs[env_id],
                                                            self.task_action_count)
            single_env_info_to_store.reward = self.reward[env_id]
            single_env_info_to_store.done = self.done[env_id]

            per_env_data.append(single_env_info_to_store)

        return per_env_data
