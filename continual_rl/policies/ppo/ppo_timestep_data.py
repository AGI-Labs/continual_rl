from continual_rl.policies.timestep_data_base import TimestepDataBase


class PPOTimestepData(TimestepDataBase):
    def __init__(self, observation, recurrent_hidden_states, actions, action_log_probs, values, action_space):
        super().__init__()
        self.observation = observation
        self.recurrent_hidden_states = recurrent_hidden_states
        self.actions = actions
        self.action_log_probs = action_log_probs
        self.values = values
        self.action_space = action_space
