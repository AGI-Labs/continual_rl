# MIT License
#
# Copyright (c) 2019 Lucas Willems
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# From: https://github.com/lcswillems/torch-ac/blob/64833c6c72b61a3295eb3b7104f2bda65a5477cb/torch_ac/utils/penv.py
# And modified


from multiprocessing import Process, Pipe
import gym
import cloudpickle
from continual_rl.utils.utils import Utils


def worker(conn, env_spec, output_dir):
    env_spec = cloudpickle.loads(env_spec)
    env, seed = Utils.make_env(env_spec, create_seed=True)

    if output_dir is not None:
        logger = Utils.create_logger(f"{output_dir}/env.log")
        logger.info(f"Created env with seed {seed}")

    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        elif cmd == "kill":
            env.close()
            return
        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs, output_dir):
        assert len(envs) >= 1, "No environment given."

        self._env_specs = envs

        # The first env is local. This helps with testing, and also makes the sync runner easier
        # Downside: slightly different code paths for 1st as opposed to rest.
        self._local_env, seed = Utils.make_env(self._env_specs[0], create_seed=True)
        self.observation_space = self._local_env.observation_space
        self.action_space = self._local_env.action_space

        if output_dir is not None:
            logger = Utils.create_logger(f"{output_dir}/env.log")
            logger.info(f"Created env with seed {seed}")

        self.locals = []
        for env_spec in self._env_specs[1:]:
            local, remote = Pipe()
            self.locals.append(local)

            pickled_spec = cloudpickle.dumps(env_spec)
            p = Process(target=worker, args=(remote, pickled_spec, output_dir))
            p.daemon = True
            p.start()
            remote.close()

    def __del__(self):
        self.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self._local_env.reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, done, info = self._local_env.step(actions[0])
        if done:
            obs = self._local_env.reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError

    def close(self):
        self._local_env.close()
        for local in self.locals:
            local.send(("kill", None))
