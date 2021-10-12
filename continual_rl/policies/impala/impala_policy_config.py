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

        # Does not call eval() on the policy before evaluation,
        # use when you want the same policy to run on the environment in eval as it does in test.
        self.no_eval_mode = False

        # If APPO is true, we augment this class's parameters with those in MonobeastAPPOPolicyConfig
        self.use_appo_loss = False

    def _load_from_dict_internal(self, config_dict):
        # Load in our defaults for using APPO loss before we load in what the user has configured
        if config_dict.get("use_appo_loss", self.use_appo_loss):
            self.__dict__.update(MonobeastAPPOPolicyConfig().__dict__)
        
        self._auto_load_class_parameters(config_dict)
        return self


class MonobeastAPPOPolicyConfig(ImpalaPolicyConfig):
    """
    Storage class for APPO parameters. Should not be necessary to use directly - can use by setting use_appo_loss true
    in MonobeastPolicyConfig
    """

    def __init__(self):
        super().__init__()

        # APPO related params
        self.use_appo_loss = True
        self.reward_scale = 0.1
        self.reward_clip = 10
        self.normalize_reward = True
        self.normalize_advantages = True
        self.appo_clip_policy = 0.1
        self.appo_clip_baseline = 1.0
