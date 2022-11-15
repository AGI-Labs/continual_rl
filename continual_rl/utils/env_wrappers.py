# The MIT License
#
# Copyright (c) 2017 OpenAI (http://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Taken from
#   https://raw.githubusercontent.com/openai/baselines/9b68103b737ac46bc201dfb3121cfa5df2127e53/baselines/common/wrappers.py
#   https://raw.githubusercontent.com/openai/baselines/7c520852d9cf4eaaad326a3d548efc915dc60c10/baselines/common/atari_wrappers.py
#   https://github.com/facebookresearch/torchbeast/blob/542c607cfe4adbc1967c213e8c248f29b13b64b6/torchbeast/atari_wrappers.py
# and slightly modified.

import numpy as np
import os
os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym import spaces
import torch
import cv2
cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.

        This wrapper should come before any reward-modifying wrappers, so the score is maintained.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True
        self.real_episode_return = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        self.real_episode_return += reward
        episode_return_to_report = None

        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True

        if self.was_real_done:
            episode_return_to_report = self.real_episode_return
            self.real_episode_return = 0

        # Since the consumer of the env has no ability to tell a "real" done from a fake one, put the real return
        # on the info object (or a dummy placeholder to tell the caller to wait for it), but ensure we're not
        # overwriting anything.
        assert "episode_return" not in info, "Attempting to overwrite an existing episode return."
        info["episode_return"] = episode_return_to_report

        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None, resize_interp_method="INTER_AREA"):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.

        Note that for tasks with >4 channels resize_interp_method must be changed, as INTER_AREA does not support it.
        (E.g. to INTER_LINEAR). Method is the string name of the cv2 enum (so consumers don't need to import cv2).
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        self._resize_interp_method = resize_interp_method
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = self.observation_space.shape[-1]  # If we've concatenated a goal, for instance, we might have >3 channels

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale and frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Convert the string to the cv2 int value (avoiding an over-reliance on cv2)
        interp_method = cv2.__dict__[self._resize_interp_method]
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=interp_method
        )
        if self._grayscale and len(frame.shape) == 2:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """
        Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        The original version of this stacked into the last dimension (i.e. the new channels would be c * k).
        This version puts it into a new, first, dimension, so the output is [k, *image_dim]

        This version also assumes the input is a pytorch Tensor (see note in ImageToPytorch). So far this means
        that the environment should first be wrapped in ImageToPytorch. As we add non-image environments, more
        converters may need to be added.

        See Also: LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=env.observation_space.low.min(), high=env.observation_space.high.max(),
                                            shape=(k, *shp), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was.

        The original version of this concatenated into the last dimension. The current one stacks into the first.
        This version also assumes the input is a pytorch Tensor (see note in ImageToPytorch).
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = torch.stack(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def to_tensor(self):
        """
        Ideally LazyFrames would just be interchangeable with Tensors, but in practice that isn't true.
        This forces the retrieval of the Tensor version of the LazyFrames. Know that using this negates the memory
        savings of LazyFrames.
        """
        frames = self._force()
        return frames


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        import numpy as np
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """
    # TODO (Issue 50): If used after LazyFrames, seems to negate the point of LazyFrames
    # As in, LazyFrames provides no benefit?
    # For now switching this to return a Tensor and calling it *before* FrameStack...

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        processed_observation = torch.as_tensor(observation)
        processed_observation = processed_observation.permute(2, 0, 1)
        return processed_observation


def wrap_pytorch(env):
    return ImageToPyTorch(env)


class FixedSetWrapper(gym.Wrapper):
    def __init__(self, env, seeds):
        super().__init__(env)
        self._env = env
        self._seeds = seeds

    def reset(self):
        seed = np.random.choice(self._seeds)
        self._env.seed(int(seed))
        return self._env.reset()
