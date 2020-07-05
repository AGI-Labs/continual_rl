# Continual Reinforcement Learning

This repository provides a simple way to run continual reinforcement learning experiments in PyTorch, including 
evaluating existing baseline algorithms, writing your own agents, and specifying custom experiments.

## Quick Start

### Setup your environment

1. Git clone this repo and cd into it

2. Run this command to set up a conda environment with the required packages:
```
conda env create -f environment.yml -n <venv_name> 
```
Replace <venv_name> with a virtual environment name of your choosing. If you leave off the -n argument, the default 
name venv_continual_rl will be used.

3. Activate your new virtual environment: `conda activate <venv_name>`

### Run an experiment
An experiment is a list of tasks, executed sequentially. Each task manages the training of a policy on a single 
environment. A simple experiment can be run with:

```
python main.py --policy PPO --experiment recall_minigrid_empty8x8_unlock
```

The available policies are the folders under the policies directory. The available experiments are in 
experiment_specs.py.

The output directory will be `tmp/<policy>_<experiment>_<timestamp>` This will contain output log files and saved
models.

#### Experiment types
Generally, there are two types of experiment: recall experiments and transfer experiments.
 
Recall experiments will train a sequence of tasks, and then test again on the first task, and see how well it does 
compared to when it was first trained. This tests how much catastrophic forgetting was mitigated.

Transfer tasks train on a sequence of tasks, then train on a brand new task, and see how well the experience 
from the first tasks aid in (or hinder) the learning of the last task. This type of test is sometimes called an
interference test.

## Use as a package
If you simply wish to use the policies or experiments in your own code, continual_rl can be installed as a pip 
package with:
```
pip install .
```

or for editable mode:

```
pip install -e .
```

or simply add the root folder to your PYTHONPATH.

## More advanced usage

### Additional command line arguments
In addition to `--policy` and `--experiment`, the following command line arguments to `main.py`
are also permitted:
* `--output-dir [tmp/<policy>_<experiment>_<timestamp>]`: Where logs and saved models should be stored
* `--save-frequency [500000]`: How many timesteps between saves (models will always be saved at the end of a task)
* `--load-experiment`: The path to the folder of the experiment you would like to resume (starts from the last
time a model was saved)
* `--load-model`: Begin your new experiment with a pre-trained model, loaded from this path (including 
filename)
* `--output-format [tensorboard]`: The format of the output logs



By default, the experiment will be run in "command-line" mode, where any policy configuration changes can be made
simply by appending `--param new_value` to the arguments passed to main. The default policy configurations 
(e.g. hyperparameters) are in the config.py file within the policy's folder, and any of them can be set in this way.

For example:

```
python main.py --policy PPO --experiment recall_mnist_0-4_5 --learning_rate 1e-3
```
Will override the default learning_rate, and instead use 1e-3.


### Configuration files
There is another way experiments can be run: in "config-file" mode instead of "command-line". 

Configuration files are an easy way to keep track of large numbers of experiments.

A configuration file contains JSON representing a list of dictionaries, where each dictionary is an experiment 
configuration. The parameters in the dictionary are all exactly the same as those used by the command line (without --).
An example config file can be found in [TODO].

When you run the code with:
```
python main.py --config-file <path_to_file/some_config_file.json> [--output-dir tmp]
```

A new folder with the name "some_config_file" will be created in output-dir (tmp if otherwise unspecified).

Each experiment in some_config_file.json will be executed sequentially, creating subfolders "0", "1", "2", etc. under 
output_dir as experiments finish and new ones are started. If the command above is run a second time, it will find the 
first experiment not yet started by finding the first missing numbered subfolder in output-dir. Thus
you can safely run the same command on multiple machines (if they share a filesystem) or multiple sessions on the 
same machine, and each will be executing different experiments in your queue.

If you wish to re-run an experiment, you can add the argument:
```
--force-id n
```

The previous instance of that experiment will be deleted, so do this with caution.


## So you want to modify code?
### High Level Code Structure
An experiment is a list of tasks, executed sequentially. Each task manages the training of an agent on a single 
environment. The default set of experiments can be seen in experiment_spec.py.

Each task has a type (i.e. subclasses TaskBase) based on what type of preprocessing the observation may require. 
For instance, ImageTasks will resize your image to the specified size, and permute the channels to match PyTorch's 
requirements.

Policies are the core of how an agent operates: given an observation, they produce an action to take. Policies also 
specify what type of EpisodeRunner they should be run with.

The available EpisodeRunners are:
1. EpisodeRunnerSync: Runs environments individually, synchronously. Passes a single observation into 
policy.compute_action
2. EpisodeRunnerBatch: Passes a batch of observations to policy.compute_action, and runs the environments in parallel
3. EpisodeRunnerFullParallel: Spins up a specified number of processes, and a specified number of threads per process, 
and runs the environment for n steps (or until the end of the episode) separately on each. Passes a single 
observation into policy.compute_action

### Creating a new Policy
[TODO]


### Contributing to the repository
Yes, please do! Pull requests encouraged. Please run pytest on the repository before submitting.


