from continual_rl.policies.config_base import ConfigBase


class ImpalaPolicyConfig(ConfigBase):

    def __init__(self):
        super().__init__()
        self.num_actors = 4
        self.batch_size = 8
        self.unroll_length = 80
        self.num_buffers = None
        self.num_learner_threads = 2
        self.use_lstm = False
        self.entropy_cost = 0.0006
        self.baseline_cost = 0.5
        self.discounting = 0.99
        self.reward_clipping = "abs_one"
        self.learning_rate = 0.00048
        self.alpha = 0.99  # RMSProp smoothing constant
        self.momentum = 0
        self.epsilon = 0.01
        self.grad_norm_clipping = 40.0

    def _load_from_dict_internal(self, config_dict):
        # Automatically grab all parameters in this class from the configuration dictionary, if they are there.
        for key, value in self.__dict__.items():
            # Get the class of the default (e.g. int) and cast to it (if not None)
            default_val = self.__dict__[key]
            type_to_cast_to = type(default_val) if default_val is not None else lambda x: x
            self.__dict__[key] = type_to_cast_to(config_dict.pop(key, value))

        # This is the only parameter with a default of None, so cast it to the right type manually
        if self.num_buffers is not None:
            self.num_buffers = int(self.num_buffers)

        return self
