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
        self.normalize_reward = False
        self.learning_rate = 0.00048
        self.optimizer = "rmsprop"
        self.use_scheduler = True
        self.alpha = 0.99  # RMSProp smoothing constant
        self.momentum = 0  # RMSProp momentum
        self.epsilon = 0.01  # RMSProp epsilon
        self.grad_norm_clipping = 40.0
        self.device = "cuda:0"
        self.disable_checkpoint = False
        self.comment = ""
        self.render_freq = 200000  # Timesteps between outputting a video to the tensorboard log
        self.seconds_between_yields = 5
        self.pause_actors_during_yield = True
        self.eval_episode_num_parallel = 10  # The number to run in parallel at a time
        self.encode_frame_as_uint8 = True  # TODO autodetermine. Saves considerable space for large replay buffers, but will do bad things to observations that need to be floats

        # Continuous action-related parameters
        self.continuous_actions = False
        self.ou_theta = 0.15
        self.ou_mu = 0.0
        self.ou_sigma = 0.2
        self.decay_epsilon = True
        self.epsilon_decay_rate = 1/50000.0
        self.target_learner_tau = 0.001
        self.actor_learning_rate = 1e-4

        # Does not call eval() on the policy before evaluation,
        # use when you want the same policy to run on the environment in eval as it does in test.
        self.no_eval_mode = False

    def _load_from_dict_internal(self, config_dict):
        self._auto_load_class_parameters(config_dict)
        return self
