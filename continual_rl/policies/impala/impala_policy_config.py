from continual_rl.policies.config_base import ConfigBase


class ImpalaPolicyConfig(ConfigBase):
    def __init__(self):
        super().__init__()
        self.num_actors = 4
        self.batch_size = 8
        self.unroll_length = 80
        self.num_buffers = None
        self.num_learner_threads = 2
        self.use_lstm = False  # Not presently fully supported
        self.entropy_cost = 0.0006
        self.baseline_cost = 0.5
        self.discounting = 0.99
        self.reward_clipping = "abs_one"
        self.learning_rate = 0.00048
        self.optimizer = "rmsprop"
        self.scheduler = True
        self.alpha = 0.99  # RMSProp smoothing constant
        self.momentum = 0  # RMSProp momentum
        self.epsilon = 0.01  # RMSProp epsilon
        self.grad_norm_clipping = 40.0
        self.disable_cuda = False
        self.disable_checkpoint = False
        self.comment = ""
        self.render_freq = 200000  # Timesteps between outputting a video to the tensorboard log
        self.seconds_between_yields = 5
        self.pause_actors_during_yield = True
        self.no_eval_mode = False

    def _load_from_dict_internal(self, config_dict):
        self._auto_load_class_parameters(config_dict)
        return self
