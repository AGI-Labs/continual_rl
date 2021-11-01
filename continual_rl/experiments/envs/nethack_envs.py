from minihack import MiniHackSkill, LevelGenerator, RewardManager
from minihack.reward_manager import Event
from minihack.base import MH_DEFAULT_OBS_KEYS
from gym.envs import registration
import nle
from typing import List
import re

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
        hunger_index = 7
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
        self._min_ac = None

    @classmethod
    def get_ac_from_obs(cls, env, observation):
        ac_index = 16  # Armor class
        return observation[env._original_observation_keys.index("blstats")][ac_index]

    def check(self, env, previous_observation, action, observation) -> float:
        curr_ac = self.get_ac_from_obs(env, observation)
        reward = 0.0
        #print(f"curr ac: {curr_ac}, min: {self._min_ac}")

        if self._min_ac is not None and curr_ac < self._min_ac:  # Nethack is opposite of 5e dnd: lower is better
            reward = self._set_achieved()

        if self._min_ac is None or curr_ac < self._min_ac:  # Second clause not currently really used, but...keeping it for the moment (TODO)
            self._min_ac = curr_ac

        return reward

    def reset(self):
        self._min_ac = None
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
        actions = kwargs.pop("actions", nle.env.base.FULL_ACTIONS)

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


registration.register(
    id="MiniHack-PickupEatFood-v0",
    entry_point="continual_rl.experiments.envs.nethack_envs:MiniHackPickupEatFood",
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
