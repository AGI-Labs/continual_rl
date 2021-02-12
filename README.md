# Continual Reinforcement Learning

This repository provides a simple way to run continual reinforcement learning experiments in PyTorch, including 
evaluating existing baseline algorithms, writing your own agents, and specifying custom experiments.

## Quick Start

### Setup your environment

1. Git clone this repo and cd into it:
```
git clone git@github.com:SamNPowers/continual-reinforcement-learning.git
cd continual-reinforcement-learning
```

There are two flavors of installation: pip-only, and conda. Pick your poison.

#### Pip setup
```
pip install torch torchvision
pip install -e .
```

Depending on your platform you may need a different torch installation command. See https://pytorch.org/

If you prefer not to install continual_rl as a pip package, you can alternatively do `pip install -r requirements.txt`

#### Conda Setup (Currently quite slow)
1. Run this command to set up a conda environment with the required packages:
```
conda env create -f environment.yml -n <venv_name> 
```
Replace <venv_name> with a virtual environment name of your choosing. If you leave off the -n argument, the default 
name venv_continual_rl will be used. 

2. Activate your new virtual environment: `conda activate <venv_name>`

### Run an experiment
An experiment is a list of tasks, executed sequentially. Each task manages the training of a policy on a single 
environment. A simple experiment can be run with:

```
python main.py --policy impala --experiment mnist_full
```

The available policies are in continual_rl/available_policies.py. The available experiments are in 
continual_rl/experiment_specs.py.

The output directory will be `tmp/<policy>_<experiment>_<timestamp>` This will contain output log files and saved
models.

## Use as a package
If you simply wish to use the policies or experiments in your own code and have no wish to edit, continual_rl can be 
installed as a pip package with:
```
pip install .
```

## More advanced usage

### Additional command line arguments
In addition to `--policy` and `--experiment`, the following command line arguments to `main.py`
are also permitted:
* `--output-dir [tmp/<policy>_<experiment>_<timestamp>]`: Where logs and saved models should be stored
* [Not yet implemented] `--save-frequency [500000]`: How many timesteps between saves (models will always be saved at the end of a task)
* [Not yet implemented] `--load-experiment`: The path to the folder of the experiment you would like to resume (starts from the last
time a model was saved)
* [Not yet implemented] `--load-model`: Begin your new experiment with a pre-trained model, loaded from this path (including 
filename)
* [Not yet implemented] `--output-format [tensorboard]`: The format of the output logs



By default, the experiment will be run in "command-line" mode, where any policy configuration changes can be made
simply by appending `--param new_value` to the arguments passed to main. The default policy configurations 
(e.g. hyperparameters) are in the config.py file within the policy's folder, and any of them can be set in this way.

For example:

```
python main.py --policy impala --experiment mnist_full --learning_rate 1e-3
```
Will override the default learning_rate, and instead use 1e-3.


### Configuration files
There is another way experiments can be run: in "config-file" mode instead of "command-line". 

Configuration files are an easy way to keep track of large numbers of experiments.

A configuration file contains JSON representing a list of dictionaries, where each dictionary is an experiment 
configuration. The parameters in the dictionary are all exactly the same as those used by the command line (without --).
An example config file can be found in `configs/example_configs.json`.

When you run the code with:
```
python main.py --config-file <path_to_file/some_config_file.json> [--output-dir tmp]
```

A new folder with the name "some_config_file" will be created in output-dir (tmp if otherwise unspecified).

Each experiment in some_config_file.json will be executed sequentially, creating subfolders "0", "1", "2", etc. under 
output_dir/some_config_file as experiments finish and new ones are started. The subfolder number corresponds to 
the index of the experiment in the config file's list. If the command above is run a second time, it will 
find the first experiment not yet started by finding the first missing numbered subfolder in output-dir. Thus
you can safely run the same command on multiple machines (if they share a filesystem) or multiple sessions on the 
same machine, and each will be executing different experiments in your queue.

[Not yet implemented] If you wish to re-run an experiment, you can add the argument:
```
--force-id n
```

The previous instance of that experiment will be deleted, so do this with caution.


## Custom Code
### High Level Code Structure
An experiment is a list of tasks, executed sequentially. Each task manages the training of an agent on a single 
environment. The default set of experiments can be seen in experiment_spec.py.

Each task has a type (i.e. subclasses TaskBase) based on what type of preprocessing the observation may require. 
For instance, ImageTasks will resize your image to the specified size, and permute the channels to match PyTorch's 
requirements.

Conceptually, experiments and tasks contain information that should be consistent between runs of the experiment across 
different algorithms, to maintain a consistent setting for a baseline.

Policies are the core of how an agent operates. During compute_action(), given an observation, they produce an action 
to take and an instance of TimestepData containing information necessary for the train step. 
In get_environment_runner(), Policies specify what type of EnvironmentRunner they should be run with, described further below. 
During policy.train() the policy updates its parameters according to the data collected and passed in.

Conceptually, policies (and policy_configs) contain information that is specific to the algorithm currently being run,
and is not expected to be held consistent for experiments using other policies (e.g. clip_eps for PPO).

EnvironmentRunners specify how the environment should be called; they contain the core loop 
(observation to action to next observation). The available EnvironmentRunners are:
1. EnvironmentRunnerSync: Runs environments individually, synchronously. 
2. EnvironmentRunnerBatch: Passes a batch of observations to policy.compute_action, and runs the environments in parallel.
3. EnvironmentRunnerFullParallel: Spins up a specified number of processes and runs the environment for n steps (or until the end of the episode) separately on each. 

More detail about what each Runner provides to the policy are specified in the collect_data method of
each Runner. 


### Creating a new Policy
1. Duplicate the prototype_policy folder in policies/, renaming it to something distinctive to your new policy.
2. Rename all other instances of the word "prototype" in filenames, class names, and imports in your new directory.
3. Your X_policy_config.py file contains all configurations that will automatically be accepted as command line 
arguments or config file parameters, provided you follow the pattern provided (add a new instance variable, and an 
entry in _load_dict_internal that populates the variable from a provided dictionary, or utilize _auto_load_class_parameters).
4. Your X_policy.py file contains the meat of your implementation. What each method requires is described fully in 
policy_base.py
5. Your X_timestep_data.py file contains any data you want stored from your compute_action, to be passed to your 
train step. This object contains one timestep's worth of data, and its .reward and .done will be populated by
the environment_runner you select in your X_policy.py file.
6. Create unit tests in tests/policies/X_policy (highly encouraged as much as possible)
7. Add a new entry to available_policies.py

### Create a new Task
If you want to run an experiment with tasks that cannot be handled with the existing TaskBase subclasses,
simply implement a subclass of TaskBase according to the methods defined therein, using the 
existing tasks as a guide.

Create unit tests in tests/experiments/tasks/X_task.py

Add a new entry to experiment_specs.py with any experiments using your new task.

### Further customization
New experiments can be defined in experiment_specs.py. 

If you need environments run in a particular way, subclass environment_runner_base.py.

If there's anything you want to customize that does not seem possible, or overly challenging, feel free to file an issue 
in the issue tracker and I'll look as soon as possible.

This repository uses pytest for tests. Please write tests wherever possible, in the appropriate folder in tests/.


### Contributing to the repository
Yes, please do! Pull requests encouraged. Please run pytest on the repository before submitting.


