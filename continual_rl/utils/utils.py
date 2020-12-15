import logging
import types
import gym
import numpy as np
import random
import torch
import os


class Utils(object):

    @classmethod
    def create_logger(cls, file_path):
        """
        The path must be unique to the logger you're creating, otherwise you're grabbing an existing logger.
        """
        logger = logging.getLogger(file_path)

        # Since getLogger will always retrieve the same logger, we need to make sure we don't add many duplicate handlers
        # Check if we've set this up before by seeing if handlers already exist
        if len(logger.handlers) == 0:
            formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.DEBUG)

        return logger


    @classmethod
    def make_env(cls, env_spec, create_seed=False, seed_to_set=None):
        """
        Seeding is done at the time of environment creation partially to make sure that every env gets its own seed.
        If you seed before forking processes, the processes will all be seeded the same way, which is generally
        undesirable.
        If create_seed is False and seed_to_set is None, no seed will be set at all.
        :param env_spec: The specification used to create an env. Can be a name (for OpenAI gym envs) or a lambda,
        which will get called each time creation is desired.
        :param create_seed: If True, a new seed will be created and set (for numpy, random, torch, and the env)
        :param seed_to_set: If not None, a seed will be set to the same locations as create_seed
        :return: (env, seed): The environment and the seed that was set (None if no seed was set)
        """
        seed = None

        if isinstance(env_spec, types.LambdaType):
            env = env_spec()
        else:
            env = gym.make(env_spec)

        if create_seed or seed_to_set is not None:
            assert not (create_seed and seed_to_set is not None), \
                "If create_seed is True and a seed_to_set is specified, it is unclear which is desired."
            seed = cls.seed(env, seed=seed_to_set)

        return env, seed

    @classmethod
    def seed(cls, env=None, seed=None):
        # Use the operating system to generate a seed for us, as it is not useful to seed a randomizer with itself
        # (Such a seed would be undesirably the same across forked processes.)
        if seed is None:
            seed = int.from_bytes(os.urandom(4), byteorder="little")

        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        # In theory we should be able to call torch.seed, but that breaks on all but the most recent builds of pytorch.
        # https://github.com/pytorch/pytorch/issues/33546
        # So in the meantime do it manually. Tracked by issue 52
        # torch.seed()
        torch.manual_seed(seed)

        if env is not None:
            env.seed(seed)

        return seed
