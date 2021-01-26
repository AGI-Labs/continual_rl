from continual_rl.experiments.experiment import Experiment
from continual_rl.experiments.tasks.image_task import ImageTask
from continual_rl.experiments.tasks.minigrid_task import MiniGridTask
from continual_rl.utils.env_wrappers import wrap_deepmind, make_atari
from continual_rl.available_policies import LazyDict
from continual_rl.envs.minigrid_envs import DynamicObstaclesRandomEnv8x8, SimpleChoiceEnv, OddManOutEnv, AssociationEnv


def get_single_atari_task(action_space_id, env_name, num_timesteps, max_episode_steps=None, clip_rewards=False):
    """
    Wrap the task creation in a scope so the env_name in the lambda doesn't change out from under us.
    The atari max step default is 100k.
    """
    return ImageTask(action_space_id=action_space_id,
                     env_spec=lambda: wrap_deepmind(
                         make_atari(env_name, max_episode_steps=max_episode_steps),
                         clip_rewards=clip_rewards,
                         frame_stack=False,  # Handled separately
                         scale=False,
                     ),
                     num_timesteps=num_timesteps, time_batch_size=4, eval_mode=False,
                     image_size=[84, 84], grayscale=True)


def create_atari_cycle_loader(max_episode_steps, game_names, num_timesteps, continual_testing_freq=5e4):
    return lambda: Experiment(tasks=[
        get_single_atari_task(action_id, name, num_timesteps=num_timesteps, max_episode_steps=max_episode_steps)
        for action_id, name in enumerate(game_names)
    ], continual_testing_freq=continual_testing_freq, cycle_count=5)


def create_atari_single_game_loader(env_name, clip_rewards=False):
    return lambda: Experiment(tasks=[
        get_single_atari_task(0, env_name, num_timesteps=5e7, max_episode_steps=10000, clip_rewards=clip_rewards)
    ])


def get_single_minigrid_task(action_space_id, env_name, timesteps, mask_object_type=False):
    """
    Wrap the task creation in a scope so the env_name in the lambda doesn't change out from under us.
    """
    return MiniGridTask(action_space_id=action_space_id, env_spec=env_name,
                                num_timesteps=timesteps, time_batch_size=1,
                                eval_mode=False, mask_object_type=mask_object_type)


def create_minigrid_tasks_loader(task_data, continual_testing_freq=10000):
    return lambda: Experiment(tasks=[get_single_minigrid_task(*task_info) for task_info in task_data],
                              continual_testing_freq=continual_testing_freq)


def load_easy_coinrun():
    import gym
    return Experiment(tasks=[ImageTask(action_space_id=0,
                                       env_spec=lambda: gym.make("procgen:procgen-coinrun-v0", distribution_mode='easy'),
                                       num_timesteps=100e6, time_batch_size=4, eval_mode=False, image_size=(84, 84),
                                       grayscale=True)])


def create_easy_coinrun_climber_jumper_loader(num_timesteps):
    import gym
    return lambda: Experiment(tasks=[ImageTask(action_space_id=0,
                                       env_spec=lambda: gym.make("procgen:procgen-coinrun-v0",
                                                                 distribution_mode='easy'),
                                       num_timesteps=num_timesteps, time_batch_size=4, eval_mode=False, image_size=(84, 84),
                                       grayscale=True),
                             ImageTask(action_space_id=0,
                                       env_spec=lambda: gym.make("procgen:procgen-climber-v0",
                                                                 distribution_mode='easy'),
                                       num_timesteps=num_timesteps, time_batch_size=4, eval_mode=False, image_size=(84, 84),
                                       grayscale=True),
                             ImageTask(action_space_id=0,
                                       env_spec=lambda: gym.make("procgen:procgen-jumper-v0",
                                                                 distribution_mode='easy'),
                                       num_timesteps=num_timesteps, time_batch_size=4, eval_mode=False, image_size=(84, 84),
                                       grayscale=True)
                             ], continual_testing_freq=50000)


def load_thor_find_pick_place_fridge_goal_conditioned():
    from continual_rl.envs.thor_env_find_pick_place import ThorFindPickPlaceEnv
    return Experiment(tasks=[
            ImageTask(action_space_id=0, env_spec=lambda: ThorFindPickPlaceEnv(scene_name="FloorPlan21", objects_to_find=["Apple"], goal_conditioned=True, clear_receptacle_object=True, receptacle_object="Fridge"),
                      num_timesteps=500000, time_batch_size=1,
                      eval_mode=False, image_size=[84, 84], grayscale=False),
            ImageTask(action_space_id=0, env_spec=lambda: ThorFindPickPlaceEnv(scene_name="FloorPlan21", objects_to_find=["Bowl"], goal_conditioned=True, clear_receptacle_object=True, receptacle_object="Fridge"),
                      num_timesteps=500000, time_batch_size=1,
                      eval_mode=False, image_size=[84, 84], grayscale=False),
            ImageTask(action_space_id=0, env_spec=lambda: ThorFindPickPlaceEnv(scene_name="FloorPlan21", objects_to_find=["Bread"], goal_conditioned=True, clear_receptacle_object=True, receptacle_object="Fridge"),
                      num_timesteps=500000, time_batch_size=1,
                      eval_mode=False, image_size=[84, 84], grayscale=False),
        ], continual_testing_freq=10000)


def load_thor_find_pick_place_fridge_apple_goal_conditioned():
    from continual_rl.envs.thor_env_find_pick_place import ThorFindPickPlaceEnv
    return Experiment(tasks=[
            ImageTask(action_space_id=0, env_spec=lambda: ThorFindPickPlaceEnv(scene_name="FloorPlan21", objects_to_find=["Apple"], goal_conditioned=True, clear_receptacle_object=True, receptacle_object="Fridge"),
                      num_timesteps=500000, time_batch_size=1,
                      eval_mode=False, image_size=[84, 84], grayscale=False)
        ], continual_testing_freq=10000)

def get_available_experiments():

    experiments = LazyDict({
        "adventure": create_atari_single_game_loader("AdventureNoFrameskip-v4"),
        "air_raid": create_atari_single_game_loader("AirRaidNoFrameskip-v4"),
        "alien": create_atari_single_game_loader("AlienNoFrameskip-v4"),
        "amidar": create_atari_single_game_loader("AmidarNoFrameskip-v4"),
        "assault": create_atari_single_game_loader("AssaultNoFrameskip-v4"),
        "asterix": create_atari_single_game_loader("AsterixNoFrameskip-v4"),
        "asteroids": create_atari_single_game_loader("AsteroidsNoFrameskip-v4"),
        "atlantis": create_atari_single_game_loader("AtlantisNoFrameskip-v4"),
        "bank_heist": create_atari_single_game_loader("BankHeistNoFrameskip-v4"),
        "battle_zone": create_atari_single_game_loader("BattleZoneNoFrameskip-v4"),
        "beam_rider": create_atari_single_game_loader("BeamRiderNoFrameskip-v4"),
        "berzerk": create_atari_single_game_loader("BerzerkNoFrameskip-v4"),
        "bowling": create_atari_single_game_loader("BowlingNoFrameskip-v4"),
        "boxing": create_atari_single_game_loader("BoxingNoFrameskip-v4"),
        "breakout": create_atari_single_game_loader("BreakoutNoFrameskip-v4"),
        "carnival": create_atari_single_game_loader("CarnivalNoFrameskip-v4"),
        "centipede": create_atari_single_game_loader("CentipedeNoFrameskip-v4"),
        "chopper_command": create_atari_single_game_loader("ChopperCommandNoFrameskip-v4"),
        "crazy_climber": create_atari_single_game_loader("CrazyClimberNoFrameskip-v4"),
        "demon_attack": create_atari_single_game_loader("DemonAttackNoFrameskip-v4"),
        "double_dunk": create_atari_single_game_loader("DoubleDunkNoFrameskip-v4"),
        "elevator_action": create_atari_single_game_loader("ElevatorActionNoFrameskip-v4"),
        "fishing_derby": create_atari_single_game_loader("FishingDerbyNoFrameskip-v4"),
        "frostbite": create_atari_single_game_loader("FrostbiteNoFrameskip-v4"),
        "gopher": create_atari_single_game_loader("GopherNoFrameskip-v4"),
        "gravitar": create_atari_single_game_loader("GravitarNoFrameskip-v4"),
        "hero": create_atari_single_game_loader("HeroNoFrameskip-v4"),
        "ice_hockey": create_atari_single_game_loader("IceHockeyNoFrameskip-v4"),
        "james_bond": create_atari_single_game_loader("JamesbondNoFrameskip-v4"),
        "journey_escape": create_atari_single_game_loader("JourneyEscapeNoFrameskip-v4"),
        "kangaroo": create_atari_single_game_loader("KangarooNoFrameskip-v4"),
        "krull": create_atari_single_game_loader("KrullNoFrameskip-v4"),
        "kung_fu_master": create_atari_single_game_loader("KungFuMasterNoFrameskip-v4"),
        "montezuma_revenge": create_atari_single_game_loader("MontezumaRevengeNoFrameskip-v4"),
        "ms_pacman": create_atari_single_game_loader("MsPacmanNoFrameskip-v4"),
        "name_this_game": create_atari_single_game_loader("NameThisGameNoFrameskip-v4"),
        "phoenix": create_atari_single_game_loader("PhoenixNoFrameskip-v4"),
        "pitfall": create_atari_single_game_loader("PitfallNoFrameskip-v4"),
        "pong": create_atari_single_game_loader("PongNoFrameskip-v4"),
        "pooyan": create_atari_single_game_loader("PooyanNoFrameskip-v4"),
        "private_eye": create_atari_single_game_loader("PrivateEyeNoFrameskip-v4"),
        "qbert": create_atari_single_game_loader("QbertNoFrameskip-v4"),
        "riverraid": create_atari_single_game_loader("RiverraidNoFrameskip-v4"),
        "road_runner": create_atari_single_game_loader("RoadRunnerNoFrameskip-v4"),
        "robotank": create_atari_single_game_loader("RobotankNoFrameskip-v4"),
        "seaquest": create_atari_single_game_loader("SeaquestNoFrameskip-v4"),
        "space_invaders": create_atari_single_game_loader("SpaceInvadersNoFrameskip-v4"),
        "star_gunner": create_atari_single_game_loader("StarGunnerNoFrameskip-v4"),
        "tennis": create_atari_single_game_loader("TennisNoFrameskip-v4"),
        "time_pilot": create_atari_single_game_loader("TimePilotNoFrameskip-v4"),
        "tutankham": create_atari_single_game_loader("TutankhamNoFrameskip-v4"),
        "up_n_down": create_atari_single_game_loader("UpNDownNoFrameskip-v4"),
        "video_pinball": create_atari_single_game_loader("VideoPinballNoFrameskip-v4"),
        "wizard_of_wor": create_atari_single_game_loader("WizardOfWorNoFrameskip-v4"),
        "yars_revenge": create_atari_single_game_loader("YarsRevengeNoFrameskip-v4"),
        "zaxxon": create_atari_single_game_loader("ZaxxonNoFrameskip-v4"),

        "mini_atari_cycle": create_atari_cycle_loader(10000, ['SpaceInvadersNoFrameskip-v4',
                                                              "KrullNoFrameskip-v4",
                                                              "BeamRiderNoFrameskip-v4"], num_timesteps=1e7),
        "mini_atari_cycle_2": create_atari_cycle_loader(10000, ["HeroNoFrameskip-v4",
                                                                "StarGunnerNoFrameskip-v4",
                                                                "MsPacmanNoFrameskip-v4"], num_timesteps=1e7),
        "mini_atari_cycle_full": create_atari_cycle_loader(1e4, ['SpaceInvadersNoFrameskip-v4',
                                                                   "KrullNoFrameskip-v4",
                                                                   "BeamRiderNoFrameskip-v4",
                                                                   "HeroNoFrameskip-v4",
                                                                   "StarGunnerNoFrameskip-v4",
                                                                   "MsPacmanNoFrameskip-v4"], num_timesteps=1e7),
        "mini_atari_cycle_no_krull": create_atari_cycle_loader(1e4, ['SpaceInvadersNoFrameskip-v4',
                                                                   "BeamRiderNoFrameskip-v4",
                                                                   "HeroNoFrameskip-v4",
                                                                   "StarGunnerNoFrameskip-v4",
                                                                   "MsPacmanNoFrameskip-v4"], num_timesteps=1e7,
                                                               continual_testing_freq=200000),
        "atari_cycle": create_atari_cycle_loader(10000, ['SpaceInvadersNoFrameskip-v4',
                                                         "KrullNoFrameskip-v4",
                                                         "BeamRiderNoFrameskip-v4",
                                                         "HeroNoFrameskip-v4",
                                                         "StarGunnerNoFrameskip-v4",
                                                         "MsPacmanNoFrameskip-v4"
                                                         ], num_timesteps=5e7, continual_testing_freq=200000),
        "atari_cycle_no_krull": create_atari_cycle_loader(10000, ['SpaceInvadersNoFrameskip-v4',
                                                                  "BeamRiderNoFrameskip-v4",
                                                                  "HeroNoFrameskip-v4",
                                                                  "StarGunnerNoFrameskip-v4",
                                                                  "MsPacmanNoFrameskip-v4"
                                                                  ], num_timesteps=5e7, continual_testing_freq=6e6),
        "minier_atari_cycle": create_atari_cycle_loader(10000, ['SpaceInvadersNoFrameskip-v4',
                                                                   "KrullNoFrameskip-v4",
                                                                   "BeamRiderNoFrameskip-v4"], num_timesteps=1e5),
        "mini_atari_cycle_6act": create_atari_cycle_loader(10000, ['SpaceInvadersNoFrameskip-v4',
                                                                   "PongNoFrameskip-v4",
                                                                   "QbertNoFrameskip-v4"], num_timesteps=5e6),

        "minigrid_oddmanout": create_minigrid_tasks_loader(
            [(0, lambda: OddManOutEnv(correct_color='blue', incorrect_color='yellow'), 600000, True),
             (0, lambda: OddManOutEnv(correct_color='yellow', incorrect_color='blue'), 600000, True)]),
        "minigrid_oddmanout_obst": create_minigrid_tasks_loader(
            [(0, lambda: OddManOutEnv(correct_color='blue', incorrect_color='yellow'), 600000, True),
             (0, lambda: OddManOutEnv(correct_color='yellow', incorrect_color='blue'), 600000, True),
             (1, 'MiniGrid-Dynamic-Obstacles-6x6-v0', 750000, True)]),
        "minigrid_oddmanout_quad": create_minigrid_tasks_loader(
            [(0, lambda: OddManOutEnv(correct_color='blue', incorrect_color='yellow'), 600000, True),
             (0, lambda: OddManOutEnv(correct_color='yellow', incorrect_color='blue'), 600000, True),
             (0, lambda: OddManOutEnv(correct_color='green', incorrect_color='purple'), 600000, True),
             (0, lambda: OddManOutEnv(correct_color='purple', incorrect_color='green'), 750000, True)]),
        "minigrid_association": create_minigrid_tasks_loader(
            [
                (0, lambda: AssociationEnv(association_pairs=[('blue', 'yellow'),
                                                              ('yellow', 'purple'),
                                                              ('green', 'blue'),
                                                              ('purple', 'green')], indicator_color='blue'), 600000, True),
                (0, lambda: AssociationEnv(association_pairs=[('blue', 'blue'),
                                                              ('yellow', 'green'),
                                                              ('green', 'purple'),
                                                              ('purple', 'yellow')], indicator_color='yellow'), 750000, True)
            ]
        ),
        "minigrid_association_2": create_minigrid_tasks_loader(
            [
                (0, lambda: AssociationEnv(association_pairs=[('blue', 'yellow'),
                                                              ('yellow', 'purple')], indicator_color='blue'), 600000, True),
                (0, lambda: AssociationEnv(association_pairs=[('blue', 'blue'),
                                                              ('yellow', 'green')], indicator_color='yellow'), 750000, True)
            ]
        ),

        "easy_coinrun": load_easy_coinrun,
        "easy_coinrun_climber_jumper": create_easy_coinrun_climber_jumper_loader(30e6),
        "easy_coinrun_climber_jumper_short": create_easy_coinrun_climber_jumper_loader(5e6),
        "thor_find_pick_place_fridge_goal_conditioned": load_thor_find_pick_place_fridge_goal_conditioned,
        "thor_find_pick_place_fridge_apple_goal_conditioned": load_thor_find_pick_place_fridge_apple_goal_conditioned,

        "hero_clip_rewards": create_atari_single_game_loader("HeroNoFrameskip-v4", clip_rewards=True)
    })

    return experiments
