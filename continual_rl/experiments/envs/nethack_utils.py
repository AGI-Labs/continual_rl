import torch
import numpy as np


def get_hunger(observation):
    return observation["internal"].squeeze(0).squeeze(0)[7]


def get_ac(observation):
    return observation["blstats"].squeeze(0).squeeze(0)[16]


def get_hp(observation):
    return observation["blstats"].squeeze(0).squeeze(0)[10]

def get_depth(observation):
    return observation["blstats"].squeeze(0).squeeze(0)[12]


def get_nle_stats(observations_to_render):
    # TODO: currently assumes the observations comprise a trajectory which is...not necessarily true (is for monobeast, isn't for hackrl)
    nle_stats = []

    # TODO: more generally, and de-dupe with hackrl
    """hunger_delta = get_hunger(observations_to_render[-1]) - get_hunger(observations_to_render[0])
    nle_stats.append({"type": "scalar", "tag": "hunger_delta", "value": hunger_delta})

    # TODO: more generally. Also note this doesn't include the last value, so ac-env won't look quite right

    ac_delta = get_ac(observations_to_render[-1]) - get_ac(observations_to_render[0])
    nle_stats.append({"type": "scalar", "tag": "ac_delta", "value": ac_delta})

    hp_delta = get_hp(observations_to_render[-1]) - get_hp(observations_to_render[0])
    nle_stats.append({"type": "scalar", "tag": "hp_delta", "value": hp_delta})

    nle_stats.append({"type": "scalar", "tag": "episode_len", "value": len(observations_to_render)})"""

    episode_returns = []
    final_observations = []
    last_obs = None
    for obs in observations_to_render:
        # If we found a "done" state, add the last obs (since the new one is from a new episode)
        # TODO: envs aren't always putting "done" on the obs. (MiniHack seems to be stripping it off...?)
        if "done" in obs and obs["done"] and last_obs is not None:
            final_observations.append(last_obs)

        # Sometimes the return used for training isn't necessarily the one we care about (e.g. innate vs external)
        if "nle_episode_ret3urn" in obs:
            episode_return = obs["nle_episode_ret3urn"].squeeze()
            if not torch.isnan(episode_return):
                episode_returns.append(episode_return)

        last_obs = obs

    # Final values, for innate reward
    if len(final_observations) > 0:
        nle_stats.append({"type": "scalar", "tag": "num_final_obs", "value": len(final_observations)})

        if "blstats" in observations_to_render[-1]:  # TODO: not 100% sure why this is happening
            nle_stats.append({"type": "scalar", "tag": "final_ac", "value": np.array([get_ac(obs) for obs in final_observations]).mean()})
            nle_stats.append({"type": "scalar", "tag": "final_hp", "value": np.array([get_hp(obs) for obs in final_observations]).mean()})
            nle_stats.append({"type": "scalar", "tag": "final_depth", "value": np.array([get_depth(obs) for obs in final_observations]).mean()})

        if "internal" in observations_to_render[-1]:
            nle_stats.append({"type": "scalar", "tag": "final_hunger", "value": np.array([get_hunger(obs) for obs in final_observations]).mean()})

        if "nle_innate_reward" in observations_to_render[-1]:
            nle_stats.append({"type": "scalar", "tag": "final_nle_innate_reward", "value": np.array([obs["nle_innate_rew3ard"] for obs in final_observations]).mean()})

    if len(episode_returns) > 0:
        nle_stats.append({"type": "scalar", "tag": "mean_nle_episode_return", "value": np.array(episode_returns).mean()})

    return nle_stats
