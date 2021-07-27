# Continual Reinforcement Learning

This repository provides a simple way to run continual reinforcement learning experiments in PyTorch, including 
evaluating existing baseline algorithms, writing your own agents, and specifying custom experiments.

## Quick Start (trains CLEAR on Atari)
Clone the repo, and cd into it.
```
pip install torch>=1.7.1 torchvision
pip install -e .
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python main.py --config-file configs/clear_atari.json --output-dir tmp
```

## Getting Started

### Setup your environment

There are two flavors of installation: pip and conda.

#### Pip setup
```
pip install torch>=1.7.1 torchvision
pip install -e .
```

Depending on your platform you may need a different torch installation command. See https://pytorch.org/

If you prefer not to install continual_rl as a pip package, you can alternatively do `pip install -r requirements.txt`

#### Conda Setup
1. Run this command to set up a conda environment with the required packages:
    ```
    conda env create -f environment.yml -n <venv_name> 
    ```
    Replace <venv_name> with a virtual environment name of your choosing. If you leave off the -n argument, the default 
    name venv_continual_rl will be used. 

2. Activate your new virtual environment: `conda activate <venv_name>`

### Run an experiment (Command-line Mode)
An experiment is a list of tasks, executed sequentially. Each task manages the training of a policy on a single 
environment. A simple experiment can be run with:

```
python main.py --policy ppo --experiment mini_atari_3_tasks_3_cycles
```

The available policies are in `continual_rl/available_policies.py`. The available experiments are in
`continual_rl/experiment_specs.py`. 

#### Additional command-line arguments
In addition to `--policy` and `--experiment`, the following command-line arguments to `main.py`
are also permitted:
* `--output-dir [tmp/<policy>_<experiment>_<timestamp>]`: Where logs and saved models are stored


Any policy configuration changes can be made simply by appending `--param new_value` to the arguments passed to main. The default policy configs
(e.g. hyperparameters) are in the `config.py` file within the policy's folder, and any of them can be set in this way.

For example:

```
python main.py --policy ppo --experiment mini_atari_3_tasks_3_cycles --learning_rate 1e-3
```
will override the default learning_rate, and instead use 1e-3.


### Run an Experiment (Configuration File)
There is another way experiments can be run: in "config-file" mode instead of "command-line".

Configuration files are an easy way to keep track of large numbers of experiments, and enables resuming an experiment
from where it left off.

A configuration file contains JSON representing a list of dictionaries, where each dictionary is a single experiment's
configuration. The parameters in the dictionary are all exactly the same as those used by the command line (without --).
In other words, they are the config settings found in the policy's `config.py` file.
Example config files can be found in `configs/`.

When you run the code with:
```
python main.py --config-file <path_to_file/some_config_file.json> [--output-dir tmp]
```

A new folder with the name "some_config_file" will be created in output-dir (tmp if otherwise unspecified).

Each experiment in `some_config_file.json` will be executed sequentially, creating subfolders "0", "1", "2", etc. under
`output_dir/some_config_file`. The subfolder number corresponds to
the index of the experiment in the config file's list. Each time the command above is run, it will
find the first experiment not yet started by finding the first missing numbered subfolder in output-dir. Thus
you can safely run the same command on multiple machines (if they share a filesystem) or multiple sessions on the
same machine, and each will be executing different experiments in your queue.

If you wish to resume an experiment from where it left off, you can add the argument:
```
--resume-id n
```
and it will resume the experiment corresponding to subfolder n. (This can also be used to start an experiment by its
run id even if it hasn't been run yet, i.e. skipping forward in the config file's list.)

### Environment Variables
Useful environment variables:

1. OpenMP thread limit (required for IMPALA-based policies)
    ```
    OMP_NUM_THREADS=1
    ```

2. Which CUDA devices are visible to the code. In this example GPUs 0 and 1.
   ```
   CUDA_VISIBLE_DEVICES=0,1
   ```

3. Display Python log messages immediately to the terminal.
    ```
    PYTHONUNBUFFERED=1
    ```



## Custom Code
### High Level Code Structure
#### Experiments
An experiment is a list of tasks, executed sequentially. Each task represents the training of an agent on a single
environment. The default set of experiments can be seen in `experiment_spec.py`.

Conceptually, experiments and tasks contain information that should be consistent between runs of the experiment across 
different algorithms, to maintain a consistent setting for a baseline.

Each task has a type (i.e. subclasses TaskBase) based on what type of preprocessing the observation may require. 
For instance, ImageTasks will resize your image to the specified size, and permute the channels to match PyTorch's 
requirements. Only the most basic pre-processing happens here; everything else should be handled by the policy.

#### Policies
Policies are the core of how an agent operates, and have 3 key functions: 
1. During `compute_action()`, given an observation, a policy produces an action 
to take and an instance of TimestepData containing information necessary for the train step. 
2. In `get_environment_runner()`, Policies specify what type of EnvironmentRunner they should be run with, described further below. 
3. During `policy.train()` the policy updates its parameters according to the data collected and passed in.

Policy configuration files allow convenient specification of hyperparameters and feature flags, 
easily settable either via command line (`--my_arg`) or in a config file.

Conceptually, policies (and policy_configs) contain information that is specific to the algorithm currently being run,
and is not expected to be held consistent for experiments using other policies (e.g. clip_eps for PPO).

#### Environment Runners

EnvironmentRunners specify how the environment should be called; they contain the core loop 
(observation to action to next observation). The available EnvironmentRunners are:
1. EnvironmentRunnerSync: Runs environments individually, synchronously. 
2. EnvironmentRunnerBatch: Passes a batch of observations to policy.compute_action, and runs the environments in parallel.
3. EnvironmentRunnerFullParallel: Spins up a specified number of processes and runs the environment for n steps (or until the end of the episode) separately on each. 

More detail about what each Runner provides to the policy are specified in the collect_data method of
each Runner. 


### Creating a new Policy
1. Duplicate the prototype folder in `policies/`, renaming it to something distinctive to your new policy.
2. Rename all other instances of the word "prototype" in filenames, class names, and imports in your new directory.
3. Your `X_policy_config.py` file contains all configurations that will automatically be accepted as command line 
arguments or config file parameters, provided you follow the pattern provided (add a new instance variable, and an 
entry in `_load_dict_internal` that populates the variable from a provided dictionary, or utilize `_auto_load_class_parameters`).
4. Your `X_policy.py` file contains the meat of your implementation. What each method requires is described fully in 
`policy_base.py`
5. Your `X_timestep_data.py` file contains any data you want stored from your compute_action, to be passed to your 
train step. This object contains one timestep's worth of data, and its .reward and .done will be populated by
the environment_runner you select in your `X_policy.py` file.

    *Note: if not using the FullParallel runner, compute_action can instead save data off into a more efficient structure. See: ppo_policy.py*
    
6. Create unit tests in `tests/policies/X_policy` (highly encouraged as much as possible)
7. Add a new entry to `available_policies.py`


### Create a new Environment Runner
If your policy requires a custom environment collection loop, you may consider subclassing EnvironmentRunnerBase.
An example of doing this can be seen in IMPALA.


### Create a new Experiment
Each entry in the experiments dictionary in `experiment_specs.py` contains a lambda that, when called, returns
an instance of Experiment. The only current requirements for the list of tasks is that the observation space is
the same for all tasks, and that the action space is discrete. How different sizes of action space is handled is
up to the policy.


### Create a new Task
If you want to create an experiment with tasks that cannot be handled with the existing TaskBase subclasses,
 implement a subclass of TaskBase according to the methods defined therein, using the 
existing tasks as a guide.

Create unit tests in `tests/experiments/tasks/X_task.py`


## Contributing to the repository
Yes, please do! Pull requests encouraged. Please run pytest on the repository before submitting.

If there's anything you want to customize that does not seem possible, or seems overly challenging, feel free to file an issue 
in the issue tracker and I'll look into it as soon as possible.

This repository uses pytest for tests. Please write tests wherever possible, in the appropriate folder in tests/.

