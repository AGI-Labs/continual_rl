import torch
import numpy as np


def get_nle_stats(observations_to_render):
    nle_stats = []

    # TODO: more generally, and de-dupe with hackrl
    def get_hunger(observation):
        return observation["internal"].squeeze(0).squeeze(0)[7]
    hunger_delta = get_hunger(observations_to_render[-1]) - get_hunger(observations_to_render[0])
    nle_stats.append({"type": "scalar", "tag": "hunger_delta", "value": hunger_delta})

    # TODO: more generally. Also note this doesn't include the last value, so ac-env won't look quite right
    def get_ac(observation):
        return observation["blstats"].squeeze(0).squeeze(0)[16]
    ac_delta = get_ac(observations_to_render[-1]) - get_ac(observations_to_render[0])
    nle_stats.append({"type": "scalar", "tag": "ac_delta", "value": ac_delta})

    def get_hp(observation):
        return observation["blstats"].squeeze(0).squeeze(0)[10]
    hp_delta = get_hp(observations_to_render[-1]) - get_hp(observations_to_render[0])
    nle_stats.append({"type": "scalar", "tag": "hp_delta", "value": hp_delta})

    nle_stats.append({"type": "scalar", "tag": "episode_len", "value": len(observations_to_render)})

    # Final values, for innate reward
    nle_stats.append({"type": "scalar", "tag": "final_ac", "value": get_ac(observations_to_render[-1])})
    nle_stats.append({"type": "scalar", "tag": "final_hp", "value": get_hp(observations_to_render[-1])})
    nle_stats.append({"type": "scalar", "tag": "final_hunger", "value": get_hunger(observations_to_render[-1])})

    if "innate_reward" in observations_to_render[-1]:
        nle_stats.append({"type": "scalar", "tag": "final_innate_reward", "value": observations_to_render[-1]["innate_reward"]})

    # Sometimes the return used for training isn't necessarily the one we care about (e.g. innate vs external)
    episode_returns = []
    for obs in observations_to_render:
        if "episode_return" in obs:
            episode_return = obs["episode_return"].squeeze()
            if not torch.isnan(episode_return):
                episode_returns.append(episode_return)

    if len(episode_returns) > 0:
        nle_stats.append({"type": "scalar", "tag": "mean_episode_return", "value": np.array(episode_returns).mean()})

    return nle_stats
