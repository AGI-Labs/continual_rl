# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Taken from https://github.com/facebookresearch/torchbeast/blob/3f3029cf3d6d488b8b8f952964795f451a49048f/torchbeast/core/environment.py
# and modified slightly
"""The environment class for MonoBeast."""

import torch
import numpy as np
from continual_rl.utils.env_wrappers import LazyFrames


def _format_frame(frame):
    # TODO: This can be complicated per-environment, so let the Task Preprocessors handle it as necessary...?
    # But in some ways expecting the shape to be T, B, S, C, W, H (T=timesteps in collection, S=stacked frames) is monobeast-specific
    # so just doing it here for now...
    if isinstance(frame, LazyFrames):
        frame = frame.to_tensor()

    if isinstance(frame, dict):
        for key in frame.keys():
            if isinstance(frame[key], np.ndarray):  # TODO: probably remove this
                frame[key] = np.expand_dims(np.expand_dims(frame[key], 0), 0)
            else:
                frame[key] = frame[key].unsqueeze(0).unsqueeze(0)
    else:
        frame = frame.unsqueeze(0).unsqueeze(0)

    return frame


class Environment:
    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.zeros(1, 1, dtype=torch.uint8)  # Originally this was ones, which makes there be 0 reward episodes
        initial_frame = _format_frame(self.gym_env.reset())
        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
        )

    def step(self, action):
        frame, reward, done, prior_info = self.gym_env.step(action.item())
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        # If our environment is keeping track of this for us (EpisodicLifeEnv) use that return instead.
        if "episode_return" in prior_info:
            # The episode_return will be None until the episode is done. We make it a NaN so we can still use the
            # numpy buffer.
            prior_return = prior_info["episode_return"]
            episode_return = torch.tensor(prior_return if prior_return is not None else np.nan)
            self.episode_return = episode_return
            prior_info.pop("episode_return")

        frame = _format_frame(frame)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        env_output = dict(
            frame=frame,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
        )

        # Add anything extra that prior_info has added
        env_output.update(prior_info)
        
        return env_output

    def close(self):
        self.gym_env.close()
