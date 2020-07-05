from setuptools import setup

setup(
   name='continual_rl',
   version='1.0',
   description='Continual reinforcement learning baselines and standard experiments.',
   author='Sam Powers',
   author_email='snpowers@cs.cmu.edu',
   packages=['continual_rl', 'continual_rl.experiments', 'continual_rl.policies'],
   py_modules=['continual_rl.available_policies', 'continual_rl.experiment_specs'],
   install_requires=['gym[atari]',
                     'uuid',
                     'numpy',
                     'tensorboardX',
                     'yappi',
                     'sklearn',
                     'gym-minigrid']
)
