from continual_rl.policies.impala.impala_policy_config import ImpalaPolicyConfig


class EWCPolicyConfig(ImpalaPolicyConfig):

    def __init__(self):
        super().__init__()
        # following parameters specified by Progress&Compress, see appendix C.2
        self.batch_size = 20
        self.unroll_length = 20
        self.epsilon = 0.1  # RMSProp epsilon
        self.learning_rate = 0.0006
        self.entropy_cost = 0.01
        self.reward_clipping = "abs_one"
        self.baseline_cost = 0.5
        self.discounting = 0.99

        self.replay_buffer_frames = int(1e6)  # save a buffer per task for computing Fisher estimates
        self.large_file_path = None  # No default, since it can be very large and we want no surprises

        self.n_fisher_samples = 100  # num of batches to draw to recompute the diagonal of the Fisher

        self.ewc_lambda = 500  # "tuned choosing from [500, 1000, 1500, 2000, 2500, 3000]? exact value not specified by Progress & Compress"
        self.ewc_per_task_min_frames = int(20e6)  # "EWC penalty is only applied after 20 million frames per game" (from original EWC paper)

        self.online_ewc = False
        self.online_gamma = None

        self.omit_ewc_for_current_task = False  # Feature flag for not including the current task's EWC loss

        self.normalize_fisher_method = None  # if None, then do not normalize

        self.scale_ewc_by_num_tasks = True
        self.use_ewc_mean = False  # Default is sum

        # NOTE:
        # the original EWC paper augments the network with
        # "biases and per element multiplicative gains that were specific to each game."
        # they also implement a task-recognition module. We omit these in this implementation.

    def _load_from_dict_internal(self, config_dict):
        config = super()._load_from_dict_internal(config_dict)
        assert config.large_file_path is not None, "A file path must be specified where large files may be stored."
        return config


class OnlineEWCPolicyConfig(EWCPolicyConfig):

    def __init__(self):
        super().__init__()

        self.online_ewc = True
        self.ewc_lambda = 25  # "As the scale of the losses differ, we selected λ for online EWC as applied in P&C among [25, 75, 125, 175]."
        self.online_gamma = 0.99  # "γ < 1 is a hyperparameter associated with removing the approximation term associated with the previous presen-tation of task i."
        self.normalize_fisher_method = "full"  # "We counteract this issue by normalising the Fisher information matrices Fi for each task.""
