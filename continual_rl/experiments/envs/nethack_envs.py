from minihack import MiniHackSkill, LevelGenerator, RewardManager
from minihack.reward_manager import Event
from gym.envs import registration
from typing import List
import re


class RegexMessageEvent(Event):
    def __init__(self, *args, messages: List[str]):
        super().__init__(*args)
        self.messages = messages

    def check(self, env, previous_observation, action, observation) -> float:
        curr_msg = (
            observation[env._original_observation_keys.index("message")]
            .tobytes()
            .decode("utf-8")
        )
        for msg in self.messages:
            if re.match(msg, curr_msg):
                return self._set_achieved()
        return 0.0


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