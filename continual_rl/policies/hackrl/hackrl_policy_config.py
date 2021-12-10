from continual_rl.policies.config_base import ConfigBase
from continual_rl.policies.clear.clear_policy_config import ClearPolicyConfig
import omegaconf
import json
from hackrl.experiment import uid
import dotmap


class VitalyBaseNetInventoryConfig(ConfigBase):
    def __init__(self):
        super().__init__()
        self.model =  "transformer"
        self.hidden_dim = 256
        self.layers = 1
        self.heads = 2
        self.decoder_conditioning = "none"

    def _load_from_dict_internal(self, config_dict):
        return self._auto_load_class_parameters(config_dict)


class VitalyBaseNetConfig(ConfigBase):
    def __init__(self):
        super().__init__()
        self.use_lstm = True
        self.hidden_dim = 256    # use at least 128
        self.embedding_dim = 64  # use at least 32
        self.glyph_type = "all_cat"     # full, id_group, color_char, all, all_cat*
        self.equalize_input_dim = False  # project inputs to same dim (*false unless doing dynamics)
        self.equalize_factor = 2    # multiplies hdim by this when equalize is enabled
        self.layers = 5          # number of cnn layers for crop/glyph model
        self.crop_model = "cnn"
        self.crop_dim = 9        # size of crop
        self.use_index_select = False    # use index select instead of normal embedding lookup
        self.append_goal = False  # Model-only flag, doesn't control the env

        self.inv = None

    def _load_from_dict_internal(self, config_dict):
        return self._auto_load_class_parameters(config_dict)

"""
      defaults:
- hydra/job_logging: colorlog
- hydra/hydra_logging: colorlog
- hydra/launcher: submitit_slurm

# pip install hydra-core hydra_colorlog
# (skip hydra-fair-plugins if not on devfair)
# can set these on the commandline too, e.g. `hydra.launcher.params.partition=dev`
hydra:
  launcher:
    timeout_min: 4300
    cpus_per_task: 20  # you get 10 CPUs per GPU allocated on the cluster
    gpus_per_node: 2
    tasks_per_node: 1
    mem_gb: 20
    nodes: 1
    partition: learnfair
    comment: null  # replace this when using priority
    # constraint: gpu2  # 2-GPU machines in H2
    # exclude: null
    # signal_delay_s: 120
    max_num_timeout: 5  # will requeue on timeout or preemption


name: null  # can use this to have multiple runs with same params, eg name=1,2,3,4,5
project: nle  # specifies the W&B project to log to
group: default  # defines a group name for the experiment

# Polybeast settings
mock: false
single_ttyrec: true
num_seeds: 0

write_profiler_trace: false
normalize_reward: false
relative_reward: false
fn_penalty_step: constant
penalty_time: 0.0
penalty_step: -0.001
reward_lose: 0
reward_win: 100
character: mon-hum-neu-mal
# mon-hum-neu-mal
# val-dwa-law-fem
# wiz-elf-cha-mal
# tou-hum-neu-fem

wandb: true  # enable wandb logging

# Run settings.
mode: train
env: pickandeat
# pickandeat
# inventory_management
# scorefullkeyboard
# staircase
# pet
# eat
# gold
# score
# scout
# oracle

# Training settings.
num_actors: 256           # should be at least batch_size
total_steps: 1e9          # 1e9 is standard
batch_size: 32            # 32 is standard
unroll_length: 70         # 80 is standard, 70 to reduce memory usage
num_learner_threads: 1    # maybe just keep at 1, uses more GPU RAM
num_inference_threads: 1  # maybe just keep at 1
disable_cuda: false
learner_device: cuda:1
actor_device: cuda:0

# Model settings.
model: baseline
use_lstm: true
hidden_dim: 256    # use at least 128
embedding_dim: 64  # use at least 32
glyph_type: all_cat     # full, id_group, color_char, all, all_cat*
equalize_input_dim: false  # project inputs to same dim (*false unless doing dynamics)
equalize_factor: 2    # multiplies hdim by this when equalize is enabled
layers: 5          # number of cnn layers for crop/glyph model
crop_model: cnn
crop_dim: 9        # size of crop
use_index_select: false    # use index select instead of normal embedding lookup


# Loss settings.
entropy_cost: 0.0001
baseline_cost: 0.5
discounting: 0.999       # with RND, rec to use 0.999 instead of 0.99
reward_clipping: tim     # timclipping is amazing

# Optimizer settings.
learning_rate: 0.0002
grad_norm_clipping: 40
# rmsprop settings
alpha: 0.99        # 0.99 vs 0.9 vs 0.5 seems to make no difference
momentum: 0        # keep at 0
epsilon: 0.000001  # do not use 0.01, 1e-6 seems same as 1e-8

# Experimental settings.
state_counter: none        # none, coordinates
no_extrinsic: false        # ignore extrinsic reward

int:                       # intrinsic reward options
  twoheaded: true          # separate value heads for extrinsic & intrinsic, use True
  input: full              # what to model? full, crop_only, glyph_only
  intrinsic_weight: 1      # this need sto be tuned per-model, each have different scale
  discounting: 0.99
  baseline_cost: 0.5
  episodic: true
  reward_clipping: tim
  normalize_reward: true  # whether to use reward normalization for intrinsic reward

ride:                     # Rewarding Impact-Driven Exploration
  count_norm: true        # normalise reward by the number of visits to a state
  forward_cost: 1
  inverse_cost: 0.1
  hidden_dim: 128

rnd:                      # Random Network Distillation
  forward_cost: 0.01      # weight on modelling loss (ie convergence of predictor)

msg:
  model: lt_cnn           # character model? none, lt_cnn*, cnn, gru, lstm
  hidden_dim: 128         # recommend 256
  embedding_dim: 64       # recommend 64

dynamics:
  intrinsic: simple       # intrinsic reward options: none, simple, ride
  forward_cost: 10        # if > 0, do forward modelling
  inverse_cost: -1        # if > 0, do inverse modelling
  hidden_dim: 512
  crop_size: 3            # crop size for any crop-based targets
  input: rnn              # rnn, full, crop_rep, glyphs_rep, feature_rep, msg_rep
  targets:                # if all targets are disabled, reverts to baseline
    crop: false           # predict the glyphs present in the crop around the agent
    crop_id: false
    crop_group: false
    crop_char: false
    crop_color: false     # ** this is probably the best one
    crop_special: false
    features: false       # predict the features in the next state
    feature_delta: false  # predict the change in features from this state to the next
    msg_exist: false      # whether there is a message in the next state
    reward: false         # predict the reward value
    reward_exist: false   # predict presence of positive or negative reward (two heads)
    # msg_type: false     # TODO: a kind of categorical clustering of the messages

inv:
  model: transformer # | transformer | agi_transformer
  hidden_dim: 256
  layers: 1
  heads: 2
  decoder_conditioning: "none" # none | action_rep
randomise_goal: true
randomise_inventory_order: false
wizkit_list_size: 2
max_num_steps: 1000
inventory_distractors: 10
randomise_num_distractors: true"""


class HackRLPolicyConfig(ConfigBase):

    def __init__(self):
        super().__init__()
        self.comment = ""  # Not used, just for record-keeping
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
        self.use_inventory = False

        # Plugin specifications
        self.use_clear_plugin = False
        
        # Should be populated with a dictionary corresponding to CLEAR's config params, if use_clear_plugin is True
        # Note that all strings may need to be in escaped double quotes (\")
        self.clear_config = None


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
        # the same way they override any others (by putting them in the clear_config param)
        clear_config = config_dict.pop("clear_config", self.clear_config)
        if config_dict.get("use_clear_plugin", self.use_clear_plugin):
            clear_config_container = ClearPolicyConfig()
            clear_config_container._load_from_dict_internal(config_dict=json.loads(clear_config))
            clear_config = clear_config_container
        
        # Load in VitalyBaseNet-specific parameters. (TODO: Very hacky right now)
        inv_config = None
        if config_dict.get("model", self.model) == "BaseNet":
            self.__dict__.update({key: value for key, value in VitalyBaseNetConfig().__dict__.items() 
                if not callable(value) and not isinstance(value, frozenset) and not key.startswith("_abc_") and not key.startswith("__")})  # TODO: hacky method, probably switch to something more like the above

            inv_config = VitalyBaseNetInventoryConfig()
            raw_inv_config = {key: config_dict.get(f"inv_{key}", value) for key, value in VitalyBaseNetInventoryConfig.__dict__.items()}
            inv_config._load_from_dict_internal(config_dict=raw_inv_config)

            # Modify the config in-place to say we've handled these keys
            for key in config_dict.keys():
                if key.startswith("inv_"):
                    config_dict.pop(key)

        self._auto_load_class_parameters(config_dict)
        self.inv = inv_config.__dict__   # TODO: omegaconf not playing nice with nested classes

        # HackRL uses OmegaConf to resolve parameters (e.g. env:USER will resolve to the user's username)
        omegaconf.OmegaConf.register_new_resolver("uid", uid, use_cache=True)
        self.omega_conf = omegaconf.OmegaConf.create(self.__dict__)
        omegaconf.OmegaConf.resolve(self.omega_conf)

        # Add after the rest to avoid confusing the autoloaders.
        if clear_config is not None:
            self.clear_config = clear_config_container

            # Some things need to be forcibly made the same. Prioritize the HackRL configs over the CLEAR ones
            self.clear_config.num_actors = self.batch_size  # One per learner input.... TODO: very not sure this is right...
            self.clear_config.unroll_length = self.unroll_length
            self.clear_config.device = self.device
        
        return self
