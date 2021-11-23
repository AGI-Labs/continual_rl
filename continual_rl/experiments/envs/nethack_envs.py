from minihack import MiniHackSkill, LevelGenerator, RewardManager
from minihack.reward_manager import Event
from minihack.base import MH_DEFAULT_OBS_KEYS
from gym.envs import registration
import nle
from typing import List
import re
import numpy as np
import copy

from nle.env.tasks import NetHackScore

DUNGEON_SHAPE = (76, 21)


class RegexMessageEvent(Event):
    def __init__(self, *args, messages: List[str]):
        super().__init__(*args)
        self.messages = messages

    def check(self, env, previous_observation, action, observation) -> float:
        curr_msg = (
            observation[env._original_observation_keys.index("message")]  # TODO: why isn't the dict version of the observation what gets passed in in the first place?
            .tobytes()
            .decode("utf-8")
        )
        for msg in self.messages:
            if re.match(msg, curr_msg):
                return self._set_achieved()
        return 0.0


class FoodEatenEvent(Event):
    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_hunger_from_obs(cls, env, observation):
        hunger_index = 7  # See nethack winrl.cc
        return observation[env._original_observation_keys.index("internal")][hunger_index]

    def check(self, env, previous_observation, action, observation) -> float:
        last_hunger = self.get_hunger_from_obs(env, previous_observation)
        curr_hunger = self.get_hunger_from_obs(env, observation)
        reward = 0.0

        if curr_hunger > last_hunger:
            #print(f"last: {last_hunger}, curr: {curr_hunger}")
            reward = self._set_achieved()

        return reward


class BetterArmorPutOnEvent(Event):
    def __init__(self, *args):
        super().__init__(*args)
        self._min_ac = None  # Nethack is opposite of 5e dnd: lower is better

    @classmethod
    def get_ac_from_obs(cls, env, observation):
        ac_index = 16  # Armor class
        return observation[env._original_observation_keys.index("blstats")][ac_index]

    @classmethod
    def get_hp_from_obs(cls, env, observation):
        hp_index = 10  # Hit points
        return observation[env._original_observation_keys.index("blstats")][hp_index]

    def check(self, env, previous_observation, action, observation) -> float:
        curr_ac = self.get_ac_from_obs(env, observation)
        curr_hp = self.get_hp_from_obs(env, observation)
        reward = 0.0
        print(f"curr ac: {curr_ac}, min: {self._min_ac}, action: {action}, hp: {curr_hp}")

        # If the agent prays for death, they die and their AC is marked as 0, so they get reward. Check for the death condition
        # Also, if the agent goes up the stairs, the episode ends (hp=0 and ac=0)
        if self._min_ac is not None and curr_ac < self._min_ac and curr_hp > 0:
            reward = self._set_achieved()

        if self._min_ac is None or curr_ac < self._min_ac:  # Second clause not currently really used, but...keeping it for the moment (TODO)
            self._min_ac = curr_ac

        return reward

    def reset(self):
        self._min_ac = None
        return super().reset()


class InnateDriveEvent(Event):
    def __init__(self, *args, depth_coefficient=0, time_bias=0):
        super().__init__(*args)

        self._last_hunger = None
        self._last_ac = None
        self._last_hp = None

        self._depth_coefficient = depth_coefficient
        self._last_depth = None

        self._time_bias = time_bias

    @classmethod
    def get_hunger_from_obs(cls, env, observation):
        hunger_index = 7
        return observation["internal"][hunger_index]

    @classmethod
    def get_hp_from_obs(cls, env, observation):
        hp_index = 10  # TODO: normalize by max or just go for total?
        return observation["blstats"][hp_index]

    @classmethod
    def get_hpmax_from_obs(cls, env, observation):
        hp_index = 11
        return observation["blstats"][hp_index]

    @classmethod
    def get_ac_from_obs(cls, env, observation):
        ac_index = 16  # Armor class
        return observation["blstats"][ac_index]

    @classmethod
    def get_depth_from_obs(cls, observation):
        index = 12  # Depth
        return observation["blstats"][index]

    def check(self, env, previous_observation, action, observation) -> float:
        assert previous_observation is None  # TODO: just making sure my call and usage are on the same page
        hunger = self.get_hunger_from_obs(env, observation)
        hp = self.get_hp_from_obs(env, observation)
        ac = self.get_ac_from_obs(env, observation)
        depth = self.get_depth_from_obs(observation)

        max_hp = self.get_hpmax_from_obs(env, observation)
        assert not max_hp == 0 or hp == 0, f"Max hp was {max_hp} while hp was {hp}"  # I *think* max hp 0 only happens when hp is 0... (TODO: last max hp instead?)
        max_hp = max_hp if max_hp != 0 else 10
        max_hunger = 1000  # Rough order of magnitude scaling
        max_ac = 20  # Rough order of magnitude scaling

        if self._last_hp is not None:  # TODO: just being lazy
            # High "hunger" is good, high hp is good, but lower ac is better
            # Add a time bias so the agent learns that living longer is better
            reward = (hp - self._last_hp)/max_hp + (hunger - self._last_hunger)/max_hunger - (ac - self._last_ac)/max_ac + (depth - self._last_depth) + self._time_bias
        else:
            reward = 0

        self._last_hp = hp
        self._last_ac = ac
        self._last_hunger = hunger
        self._last_depth = depth

        return reward

    def reset(self):
        self._last_hunger = None
        self._last_ac = None
        self._last_hp = None
        self._last_depth = None
        return super().reset()

class MiniHackPickupRingLev(MiniHackSkill):
    """Environment for "put on" task."""
    REGEX_MESSAGES = [
        r".* ring."
    ]

    def __init__(self, *args, **kwargs):
        kwargs["autopickup"] = True
        lvl_gen = LevelGenerator(w=5, h=5, lit=True)
        lvl_gen.add_object("levitation", "=", cursestate="blessed")  # Ring of levitation
        des_file = lvl_gen.get_des()

        reward_manager = RewardManager()
        regex_message_event = RegexMessageEvent(
                1.0,  # reward
                False,  # repeatable TODO: not exactly sure what this means
                True,  # terminal_required
                True,  # terminal_sufficient
                messages=self.REGEX_MESSAGES
        )
        reward_manager.add_event(regex_message_event)

        super().__init__(
            *args, des_file=des_file, reward_manager=reward_manager, **kwargs
        )


class MiniHackPickupEatFood(MiniHackSkill):
    """Environment for eating food."""
    def __init__(self, *args, w=9, h=9, premapped=False, **kwargs):
        kwargs["autopickup"] = True
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 250)

        des_file = f"""
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
-------
|.....|
|.....|
|.....|
|.....|
|.....|
--------
ENDMAP
REGION:(0,0,6,6),lit,"ordinary"
OBJECT: '%', random
"""

        """if premapped:
            flags = ("hardfloor", "premapped")
        else:
            flags = ("hardfloor",)

        if (w, h) != DUNGEON_SHAPE:
            # Fill the level with concrete walls " " surrounded by regular walls
            w, h = w + 2, h + 2
            lvl_gen = LevelGenerator(w=w, h=h, flags=flags)
            lvl_gen.fill_terrain("rect", "-", 0, 0, w - 1, h - 1)
            lvl_gen.fill_terrain("fillrect", " ", 1, 1, w - 2, h - 2)
        else:
            lvl_gen = LevelGenerator(w=w, h=h, fill=" ", flags=flags)

        lvl_gen.add_mazewalk()
        lvl_gen.footer += f"OBJECT:'%',random
        des_file = lvl_gen.get_des()"""

        # Add the rewards
        reward_manager = RewardManager()
        food_eaten_event = FoodEatenEvent(
                1.0,  # reward
                False,  # repeatable
                True,  # terminal_required
                True,  # terminal_sufficient
        )
        reward_manager.add_event(food_eaten_event)

        # TODO: requires obs keys passed in which is suboptimal
        observation_keys = list(kwargs["observation_keys"])
        if "internal" not in observation_keys:
            observation_keys.append("internal")
        if "glyphs" not in observation_keys:  # TODO: why was this necessary, again?
            observation_keys.append("glyphs")
        observation_keys.extend(MH_DEFAULT_OBS_KEYS)
        observation_keys = list(set(observation_keys))

        kwargs["observation_keys"] = observation_keys
        #actions = kwargs.pop("actions", nle.env.tasks.TASK_ACTIONS)
        actions = kwargs.pop("actions", nle.env.base.FULL_ACTIONS)

        super().__init__(*args, des_file=des_file, reward_manager=reward_manager, actions=actions, **kwargs)


class MiniHackPickupQuaffPotion(MiniHackSkill):
    """Environment for learning to consume potions."""
    def __init__(self, *args, w=9, h=9, premapped=False, **kwargs):
        kwargs["autopickup"] = True
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 250)

        potions = ["healing", "levitation", "speed", "gain level"]  # Should have AC > 2 (to be better than starting armor), I think
        potions_list = ", ".join([f"('!', \"{w}\")" for w in potions])

        des_file = f"""
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
-------
|.....|
|.....|
|.....|
|.....|
|.....|
--------
ENDMAP
REGION:(0,0,6,6),lit,"ordinary"

$potion_names = object: {{  {potions_list} }}
SHUFFLE: $potion_names
OBJECT: $potion_names[0], random
"""

        reward_manager = RewardManager()
        regex_message_event = RegexMessageEvent(
                1.0,  # reward
                False,  # repeatable TODO: not exactly sure what this means
                True,  # terminal_required
                True,  # terminal_sufficient
                messages=["You feel better", "You start to float", "You are suddenly moving much faster", "You feel more experienced"],
        )
        reward_manager.add_event(regex_message_event)

        # TODO: requires obs keys passed in which is suboptimal
        observation_keys = list(kwargs["observation_keys"])
        if "internal" not in observation_keys:
            observation_keys.append("internal")
        if "glyphs" not in observation_keys:  # TODO: why was this necessary, again?
            observation_keys.append("glyphs")
        observation_keys.extend(MH_DEFAULT_OBS_KEYS)
        observation_keys = list(set(observation_keys))

        kwargs["observation_keys"] = observation_keys
        #actions = kwargs.pop("actions", nle.env.tasks.TASK_ACTIONS)
        actions = kwargs.pop("actions", nle.env.base.FULL_ACTIONS)

        super().__init__(*args, des_file=des_file, reward_manager=reward_manager, actions=actions, **kwargs)


class MiniHackPickupWearArmor(MiniHackSkill):
    """Environment for wearing armor."""
    def __init__(self, *args, w=9, h=9, premapped=False, **kwargs):
        kwargs["autopickup"] = True
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 250)

        armor = ["plate mail", "splint mail", "chain mail", "studded leather armor"]  # Should have AC > 2 (to be better than starting armor), I think
        armor_list = ", ".join([f"('[', \"{w}\")" for w in armor])

        des_file = f"""
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
-------
|.....|
|.....|
|.....|
|.....|
|.....|
--------
ENDMAP
REGION:(0,0,6,6),lit,"ordinary"

$armor_names = object: {{  {armor_list} }}
SHUFFLE: $armor_names
OBJECT: $armor_names[0], random
"""
        # Add the rewards
        reward_manager = RewardManager()
        armor_event = BetterArmorPutOnEvent(
                1.0,  # reward
                False,  # repeatable
                True,  # terminal_required
                True,  # terminal_sufficient
        )
        reward_manager.add_event(armor_event)

        # TODO: requires obs keys passed in which is suboptimal
        observation_keys = list(kwargs["observation_keys"])
        if "blstats" not in observation_keys:
            observation_keys.append("blstats")
        if "glyphs" not in observation_keys:
            observation_keys.append("glyphs")
        observation_keys.extend(MH_DEFAULT_OBS_KEYS)
        observation_keys = list(set(observation_keys))

        kwargs["observation_keys"] = observation_keys
        actions = kwargs.pop("actions", nle.env.base.FULL_ACTIONS)  # ...TODO

        super().__init__(*args, des_file=des_file, reward_manager=reward_manager, actions=actions, **kwargs)


class MiniHackPickupEquipWeapon(MiniHackSkill):
    """Environment for learning to wield weapons"""

    def __init__(self, *args, **kwargs):
        kwargs["autopickup"] = True
        # Only choosing weapons with no synonyms (i.e. "katana" being "samurai sword" breaks reward logic)
        weapons = ["dagger", "spear", "trident", "axe", "short sword", "long sword", "broadsword", "crysknife", "lance", "mace"]
        weapon_list = ", ".join([f"(')', \"{w}\")" for w in weapons])

        done_messages = [fr".* {w} welds itself to your hand!" for w in weapons]
        done_messages += [fr".* {w} \(weapon in hand\)." for w in weapons]

        des_file = f"""
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
-------
|.....|
|.....|
|.....|
|.....|
|.....|
--------
ENDMAP
REGION:(0,0,6,6),lit,"ordinary"

$weapon_names = object: {{  {weapon_list} }}
SHUFFLE: $weapon_names
OBJECT: $weapon_names[0], random
"""
        reward_manager = RewardManager()
        regex_message_event = RegexMessageEvent(
                1.0,  # reward
                False,  # repeatable TODO: not exactly sure what this means
                True,  # terminal_required
                True,  # terminal_sufficient
                messages=done_messages
        )
        reward_manager.add_event(regex_message_event)

        super().__init__(
            *args, des_file=des_file, reward_manager=reward_manager, **kwargs
        )


class MiniHackWCLevitateRingInv(MiniHackSkill):  # "Water" Cross -- TODO: Water doesn't kill, so you just going to the right is enough
    """
    River is 2 wide so the agent can't make it across without levitating
    """
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)
        kwargs["autopickup"] = kwargs.pop("autopickup", True)
        des_file = """
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
-------------
|.....WW.....|
|.....WW.....|
|.....WW.....|
|.....WW.....|
|.....WW.....|
-------------
ENDMAP
REGION:(0,0,12,6),lit,"ordinary"
$right_bank = selection:fillrect (8,1,12,5)
OBJECT:('=',"levitation"),(2,2),blessed
BRANCH:(2,2,2,2),(0,0,0,0)
STAIR:rndcoord($right_bank),down
"""
        super().__init__(*args, des_file=des_file, **kwargs)


class MiniHackWCLevitatePotionInv(MiniHackSkill):
    """
    If the agent tries to swim, the potion gets replaced with water.
    We use 2-wide so the agent doesn't make it across without levitating
    """
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)
        kwargs["autopickup"] = kwargs.pop("autopickup", True)
        des_file = """
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
-------------
|.....WW.....|
|.....WW.....|
|.....WW.....|
|.....WW.....|
|.....WW.....|
-------------
ENDMAP
REGION:(0,0,12,6),lit,"ordinary"
$right_bank = selection:fillrect (8,1,12,5)
OBJECT:('!',"levitation"),(2,2),blessed
BRANCH:(2,2,2,2),(0,0,0,0)
STAIR:rndcoord($right_bank),down
"""
        super().__init__(*args, des_file=des_file, **kwargs)


class MiniHackWCLevitateRingInvRotated(MiniHackSkill):  # "Water" Cross -- TODO: Water doesn't kill, so you just going to the right is enough
    """
    River is 2 wide so the agent can't make it across without levitating
    """
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)
        kwargs["autopickup"] = kwargs.pop("autopickup", True)
        des_file = """
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
--------------
|............|
|............|
|............|
|WWWWWWWWWWWW|
|WWWWWWWWWWWW|
|............|
|............|
|............|
--------------
ENDMAP
REGION:(0,0,12,9),lit,"ordinary"
$bottom_bank = selection:fillrect (0,6,12,9)
OBJECT:('=',"levitation"),(2,2),blessed
BRANCH:(2,2,2,2),(0,0,0,0)
STAIR:rndcoord($bottom_bank),down
"""
        super().__init__(*args, des_file=des_file, **kwargs)


class MiniHackWCLevitatePotionInvRotated(MiniHackSkill):
    """
    If the agent tries to swim, the potion gets replaced with water.
    We use 2-wide so the agent doesn't make it across without levitating
    """
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)
        kwargs["autopickup"] = kwargs.pop("autopickup", True)
        des_file = """
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
--------------
|............|
|............|
|............|
|WWWWWWWWWWWW|
|WWWWWWWWWWWW|
|............|
|............|
|............|
--------------
ENDMAP
REGION:(0,0,12,9),lit,"ordinary"
$bottom_bank = selection:fillrect (0,6,12,8)
OBJECT:('!',"levitation"),(2,2),blessed
BRANCH:(2,2,2,2),(0,0,0,0)
STAIR:rndcoord($bottom_bank),down
"""
        super().__init__(*args, des_file=des_file, **kwargs)


class InnateDriveNethackEnv(NetHackScore):
    def __init__(self, *args, depth_coefficient=0.0, time_bias=0.0, external_reward_scale=1.0, 
                 penalty_mode="constant", penalty_step: float = -0.01, penalty_time: float = -0, **kwargs):
        super().__init__(*args, penalty_mode=penalty_mode, penalty_step=penalty_step, penalty_time=penalty_time, **kwargs)
        reward_manager = RewardManager()
        reward_event = InnateDriveEvent(  # TODO: not actually using the set_achieved
                1.0,  # reward
                True,  # repeatable
                False,  # terminal_required
                False,  # terminal_sufficient
                depth_coefficient=depth_coefficient,
                time_bias=time_bias
        )
        reward_manager.add_event(reward_event)

        self._innate_reward_manager = reward_manager
        self._cumulative_episode_reward = 0
        self._external_reward_scale = external_reward_scale
        self._done_step_returns = None

    def _get_observation(self, observation):
        observation = super()._get_observation(observation)

        # Initialize the custom obs -- will be replaced in step(), if that gets called
        if "nle_innate_reward" not in observation.keys():
            observation["nle_innate_reward"] = np.array([0])

        if "nle_episode_return" not in observation.keys():
            observation["nle_episode_return"] = np.array([np.nan])

        return observation

    def step(self, action: int):
        # We need last-frame information (since we put useful info on the observation in the last step). But Environment overwrites the last frame
        # with the first frame of the next episode. Instead we (falsely) extend the episode by one frame, just duplicating info, so we see it at least once
        if self._done_step_returns is None:
            obs, reward, done, info = super().step(action)
            if done:
                self._done_step_returns = copy.deepcopy(obs), reward, done, info
                done = False  # It'll be done next step instead
        else:
            obs, reward, done, info = self._done_step_returns
            self._done_step_returns = None

        # TODO: this pattern is weird and non-intuitive - basically just using the Event to compute the reward, and ignoring the fact that it's supposed to be a "done" check
        _ = self._innate_reward_manager.check_episode_end_call(self, None, action, obs)
        innate_reward = self._innate_reward_manager._reward
        self._innate_reward_manager._reward = 0  # TODO: so hacky, this makes it so the innate reward is stepwise instead of cumulative.... so bad
    
        # Not (directly) used in training, but logged to watch it
        info["nle_innate_reward"] = innate_reward

        self._cumulative_episode_reward += reward
        episode_return = self._cumulative_episode_reward if self._done_step_returns is not None else None
        info["nle_episode_return"] = episode_return

        # hackRL doesn't use info - instead you can put anything you want on obs
        # This is suboptimal because it's not necessarily the case that you want the agent to have direct
        # access to this info (e.g. a critic might learn to copy the value, not learning anything about the state of the env)
        obs["nle_episode_return"] = np.array([np.nan if episode_return is None else episode_return])
        obs["nle_innate_reward"] = np.array([innate_reward])

        combo_reward = self._external_reward_scale * reward + innate_reward
        
        return obs, combo_reward, done, info

    def reset(self, wizkit_items=None):
        self._cumulative_episode_reward = 0
        return super().reset(wizkit_items=wizkit_items)
        

registration.register(
    id="NetHackScoreInnateDrive-v0",
    entry_point="continual_rl.experiments.envs.nethack_envs:InnateDriveNethackEnv",
)
registration.register(
    id="NetHackScoreInnateDrive100-v0",
    entry_point="continual_rl.experiments.envs.nethack_envs:InnateDriveNethackEnv",
    kwargs={"external_reward_scale": 100.0}
)
registration.register(
    id="NetHackScoreInnateDrive0.1-v0",
    entry_point="continual_rl.experiments.envs.nethack_envs:InnateDriveNethackEnv",
    kwargs={"external_reward_scale": 0.1}
)
registration.register(
    id="NetHackScoreInnateDrive0.1_depth1.0-v0",
    entry_point="continual_rl.experiments.envs.nethack_envs:InnateDriveNethackEnv",
    kwargs={"external_reward_scale": 0.1, "depth_coefficient": 1.0}
)
registration.register(
    id="NetHackScoreInnateDrive0.1_time0.1-v0",
    entry_point="continual_rl.experiments.envs.nethack_envs:InnateDriveNethackEnv",
    kwargs={"external_reward_scale": 0.1, "time_bias": 0.1}
)
registration.register(
    id="MiniHack-PickupEatFood-v0",
    entry_point="continual_rl.experiments.envs.nethack_envs:MiniHackPickupEatFood",
)
registration.register(
    id="MiniHack-PickupQuaffPotion-v0",
    entry_point="continual_rl.experiments.envs.nethack_envs:MiniHackPickupQuaffPotion",
)
registration.register(
    id="MiniHack-PickupEquipWeapon-v0",
    entry_point="continual_rl.experiments.envs.nethack_envs:MiniHackPickupEquipWeapon",
)
registration.register(
    id="MiniHack-PickupWearArmor-v0",
    entry_point="continual_rl.experiments.envs.nethack_envs:MiniHackPickupWearArmor",
)
registration.register(
    id="MiniHack-PickupRingLev-v0",
    entry_point="continual_rl.experiments.envs.nethack_envs:MiniHackPickupRingLev",
)
registration.register(
    id="MiniHack-WaterCross-Levitate-Ring-Inv-v0",
    entry_point="continual_rl.experiments.envs.nethack_envs:MiniHackWCLevitateRingInv",
)
registration.register(
    id="MiniHack-WaterCross-Levitate-Potion-Inv-v0",
    entry_point="continual_rl.experiments.envs.nethack_envs:MiniHackWCLevitatePotionInv",
)
registration.register(
    id="MiniHack-WaterCross-Levitate-Ring-Inv-Rotated-v0",
    entry_point="continual_rl.experiments.envs.nethack_envs:MiniHackWCLevitateRingInvRotated",
)
registration.register(
    id="MiniHack-WaterCross-Levitate-Potion-Inv-Rotated-v0",
    entry_point="continual_rl.experiments.envs.nethack_envs:MiniHackWCLevitatePotionInvRotated",
)
