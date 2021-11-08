from continual_rl.policies.config_base import ConfigBase
from continual_rl.policies.clear.clear_policy_config import ClearPolicyConfig
import omegaconf
from hackrl.experiment import uid


class HackRLPolicyConfig(ConfigBase):

    def __init__(self):
        super().__init__()
        self.device = "cuda:0"
        self.render_freq = 100000
        #self.localdir = "${savedir}/peers/${local_name}"  # Set to output dir
        self.savedir_prefix = "/checkpoint/${env:USER}/hackrl/"

        # Logging and saving (note: not used by top-level continual_rl logs)
        self.wandb = True
        self.log_interval = 20
        self.checkpoint_interval = 600
        self.checkpoint_history_interval = 3600

        # Run identification params
        self.project = "project"
        #self.group = "group2"  # Set to output dir. This is because project+group is how the broker associates experiments, so they need to be distinct
        self.entity = None
        self.local_name = "${uid:}"
        self.connect = "127.0.0.1:4431"

        # Actor specifications
        self.actor_batch_size = 256
        self.num_actor_batches = 2
        self.num_actor_cpus = 10
        self.unroll_length = 32

        # Learning specifications
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 0.000001
        self.adam_learning_rate = 0.0001
        self.appo_clip_policy = 0.1  # 'null' to disable clipping
        self.appo_clip_baseline = 1.0  # 'null' to disable clipping
        self.baseline_cost = 0.25
        self.batch_size = 128  # TODO: what exactly is this
        self.discounting = 0.999
        self.entropy_cost = 0.001
        self.grad_norm_clipping = 4
        self.normalize_advantages = True
        self.normalize_reward = False
        self.reward_clip = 10
        self.reward_scale = 0.1
        self.virtual_batch_size = 128

        # Model specifications
        self.model = "ChaoticDwarvenGPT5"

        # Plugin specifications
        self.use_clear_plugin = False


        """

hydra:
  job_logging:
    formatters:
      simple:
        format: ${log_fmt}
  run:
    dir: "${localdir}"

activation_function: relu
actor_batch_size: 256
add_image_observation: True
adam_beta1: 0.9
adam_beta2: 0.999
adam_eps: 0.000001
adam_learning_rate: 0.0001
appo_clip_policy: 0.1  # 'null' to disable clipping
appo_clip_baseline: 1.0  # 'null' to disable clipping
baseline_cost: 0.25
batch_size: 128
character: 'mon-hum-neu-mal'
checkpoint_interval: 600
checkpoint_history_interval: 3600
connect: 127.0.0.1:4431
crop_dim: 9
device: "cuda:0"
discounting: 0.999
entity: null
entropy_cost: 0.001
env:
  name: challenge  # One of challenge, staircase, pet, eat, gold, score, scout, oracle.
  max_episode_steps: 100000
exp_point: point-A       # spare parameter, useful for wandb grouping
exp_set: experiment-set  # spare parameter, useful for wandb grouping
fixup_init: true
fn_penalty_step: constant
grad_norm_clipping: 4
group: group2 
learning_rate: 0.0002
# Savedir is used for storing the checkpoint(s),
# including flags and any global settings/stats for the training
# localdir (which is a subdirectory of savedir) should be used
# for storing logs and anything local to each instance
localdir: "${savedir}/peers/${local_name}"
local_name: "${uid:}"
log_fmt: "[%(levelname)s:${local_name} %(process)d %(module)s:%(lineno)d %(asctime)s] %(message)s"
log_interval: 20
model: ChaoticDwarvenGPT5
normalize_advantages: True
normalize_reward: False
num_actor_batches: 2
num_actor_cpus: 10
penalty_step: -0.001
penalty_time: 0.0
project: project
rms_alpha: 0.99
rms_epsilon: 0.000001
rms_momentum: 0
reward_clip: 10
reward_scale: 0.1
savedir: "/checkpoint/${env:USER}/hackrl/${project}/${group}"
state_counter: none
total_steps: 1_000_000_000
unroll_length: 32
use_bn: false
use_lstm: true
virtual_batch_size: 128
wandb: true

baseline:
  # Parameters for models/baseline.py
  embedding_dim: 64
  hidden_dim: 512
  layers: 5
  msg:
    embedding_dim: 32
    hidden_dim: 64
  restrict_action_space: True  # Use a restricted ACTION SPACE (only nethack.USEFUL_ACTIONS)
  use_index_select: False

"""

    def _load_from_dict_internal(self, config_dict):
      # If we're using CLEAR, load in its parameters. We therefore automatically support users overriding CLEAR params
      # the same way they override any others
      # TODO: probably there are some Monobeast params mixed up in the CLEAR ones - separate them out
      # TODO: we are by-passing CLEAR's custom load, and not really properly handling what to do if both define a key of the same name
      # this is mostly just quick-and-dirty... (also see APPO in IMPALA for another similar situation)
        if config_dict.get("use_clear_plugin", self.use_clear_plugin):
            self.__dict__.update(ClearPolicyConfig().__dict__)

        self._auto_load_class_parameters(config_dict)

        # HackRL uses OmegaConf to resolve parameters (e.g. env:USER will resolve to the user's username)
        omegaconf.OmegaConf.register_new_resolver("uid", uid, use_cache=True)
        self.omega_conf = omegaconf.OmegaConf.create(self.__dict__)
        omegaconf.OmegaConf.resolve(self.omega_conf)

        return self
