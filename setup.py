from setuptools import setup, find_packages

setup(
    name='continual_rl',
    version='1.0',
    description='Continual reinforcement learning baselines and standard experiments.',
    author='Sam Powers',
    author_email='snpowers@cs.cmu.edu',
    packages=find_packages(),
    py_modules=['continual_rl.available_policies', 'continual_rl.experiment_specs'],
    install_requires=['uuid',
                      'numpy',
                      'yappi',
                      'sklearn',
                      'tensorboard',
                      'torch-ac',
                      'gym-minigrid',
                      'gym[atari]',
                      'moviepy',
                      'dotmap',
                      'procgen',
                      'ai2thor']
)
