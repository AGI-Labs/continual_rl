from continual_rl.experiments.experiment import Experiment
from continual_rl.experiments.tasks.image_task import ImageTask
from continual_rl.experiments.tasks.minigrid_task import MiniGridTask
from continual_rl.utils.env_wrappers import wrap_deepmind, make_atari
from continual_rl.available_policies import LazyDict


def load_mini_atari_cycle():
    return Experiment(tasks=[
                ImageTask(action_space_id=0,
                          env_spec=lambda: wrap_deepmind(
                              make_atari('SpaceInvadersNoFrameskip-v4', max_episode_steps=10000),
                              clip_rewards=False,
                              frame_stack=False,  # Handled separately
                              scale=False,
                          ),
                          num_timesteps=10000000, time_batch_size=4, eval_mode=False,
                          image_size=[84, 84], grayscale=True),
                   ImageTask(action_space_id=2,
                             env_spec=lambda: wrap_deepmind(
                                 make_atari('KrullNoFrameskip-v4', max_episode_steps=10000),
                                 clip_rewards=False,
                                 frame_stack=False,  # Handled separately
                                 scale=False,
                             ), num_timesteps=10000000, time_batch_size=4, eval_mode=False,
                             image_size=[84, 84], grayscale=True),
                   ImageTask(action_space_id=4,
                             env_spec=lambda: wrap_deepmind(
                                 make_atari('BeamRiderNoFrameskip-v4', max_episode_steps=10000),
                                 clip_rewards=False,
                                 frame_stack=False,  # Handled separately
                                 scale=False,
                             ), num_timesteps=10000000, time_batch_size=4, eval_mode=False,
                             image_size=[84, 84], grayscale=True)
            ], continual_testing_freq=50000, cycle_count=5)


def load_atari_single_game(env_name):
    return Experiment(tasks=[
                ImageTask(action_space_id=0,
                          env_spec=lambda: wrap_deepmind(
                              make_atari(env_name),
                              clip_rewards=False,
                              frame_stack=False,  # Handled separately
                              scale=False,
                          ),
                          num_timesteps=5e7, time_batch_size=4, eval_mode=False,
                          image_size=[84, 84], grayscale=True)
            ], continual_testing_freq=50000, cycle_count=5)


def load_minigrid_empty8x8_unlock():
    return Experiment(tasks=[MiniGridTask(action_space_id=0, env_spec='MiniGrid-Empty-8x8-v0', num_timesteps=150000,
                                           time_batch_size=1, eval_mode=False),
                              MiniGridTask(action_space_id=0, env_spec='MiniGrid-Unlock-v0', num_timesteps=500000,
                                           time_batch_size=1, eval_mode=False),
                              MiniGridTask(action_space_id=0, env_spec='MiniGrid-Empty-8x8-v0', num_timesteps=10000,
                                           time_batch_size=1, eval_mode=True)
                              ])


def get_available_experiments():

    experiments = LazyDict({
        "adventure": load_atari_single_game("AdventureNoFrameSkip-v4"),
        "air_raid": load_atari_single_game("AirRaidNoFrameSkip-v4"),
        "alien": load_atari_single_game("AlienNoFrameSkip-v4"),
        "amidar": load_atari_single_game("AmidarNoFrameSkip-v4"),
        "asault": load_atari_single_game("AssaultNoFrameSkip-v4"),
        "asterix": load_atari_single_game("AsterixNoFrameSkip-v4"),
        "asteroids": load_atari_single_game("AsteroidsNoFrameSkip-v4"),
        "atlantis": load_atari_single_game("AtlantisNoFrameSkip-v4"),
        "bank_heist": load_atari_single_game("BankHeistNoFrameSkip-v4"),
        "battle_zone": load_atari_single_game("BattleZoneNoFrameSkip-v4"),
        "beam_rider": load_atari_single_game("BeamRiderNoFrameSkip-v4"),
        "berzerk": load_atari_single_game("BerzerkNoFrameSkip-v4"),
        "bowling": load_atari_single_game("BowlingNoFrameSkip-v4"),
        "boxing": load_atari_single_game("BoxingNoFrameSkip-v4"),
        "breakout": load_atari_single_game("BreakoutNoFrameSkip-v4"),
        "carnival": load_atari_single_game("CarnivalNoFrameSkip-v4"),
        "centipede": load_atari_single_game("CentipedeNoFrameSkip-v4"),
        "chopper_command": load_atari_single_game("ChopperCommandNoFrameSkip-v4"),
        "crazy_climber": load_atari_single_game("CrazyClimberNoFrameSkip-v4"),
        "demon_attack": load_atari_single_game("DemonAttackNoFrameSkip-v4"),
        "double_dunk": load_atari_single_game("DoubleDunkNoFrameSkip-v4"),
        "elevator_action": load_atari_single_game("ElevatorActionNoFrameSkip-v4"),
        "fishing_derby": load_atari_single_game("FishingDerbyNoFrameSkip-v4"),
        "frostbite": load_atari_single_game("FrostbiteNoFrameSkip-v4"),
        "gopher": load_atari_single_game("GopherNoFrameSkip-v4"),
        "gravitar": load_atari_single_game("GravitarNoFrameSkip-v4"),
        "hero": load_atari_single_game("HeroNoFrameSkip-v4"),
        "ice_hockey": load_atari_single_game("IceHockeyNoFrameSkip-v4"),
        "james_bond": load_atari_single_game("JamesbondNoFrameSkip-v4"),
        "journey_escape": load_atari_single_game("JourneyEscapeNoFrameSkip-v4"),
        "kangaroo": load_atari_single_game("KangarooNoFrameSkip-v4"),
        "krull": load_atari_single_game("KrullNoFrameSkip-v4"),
        "kung_fu_master": load_atari_single_game("KungFuMasterNoFrameSkip-v4"),
        "montezuma_revenge": load_atari_single_game("MontezumaRevengeNoFrameSkip-v4"),
        "ms_pacman": load_atari_single_game("MsPacmanNoFrameSkip-v4"),
        "name_this_game": load_atari_single_game("NameThisGameNoFrameSkip-v4"),
        "phoenix": load_atari_single_game("PhoenixNoFrameSkip-v4"),
        "pitfall": load_atari_single_game("PitfallNoFrameSkip-v4"),
        "pong": load_atari_single_game("PongNoFrameSkip-v4"),
        "pooyan": load_atari_single_game("PooyanNoFrameSkip-v4"),
        "private_eye": load_atari_single_game("PrivateEyeNoFrameSkip-v4"),
        "qbert": load_atari_single_game("QbertNoFrameSkip-v4"),
        "riverraid": load_atari_single_game("RiverraidNoFrameSkip-v4"),
        "road_runner": load_atari_single_game("RoadRunnerNoFrameSkip-v4"),
        "robotank": load_atari_single_game("RobotankNoFrameSkip-v4"),
        "seaquest": load_atari_single_game("SeaquestNoFrameSkip-v4"),
        "space_invaders": load_atari_single_game("SpaceInvadersNoFrameSkip-v4"),
        "star_gunner": load_atari_single_game("StarGunnerNoFrameSkip-v4"),
        "tennis": load_atari_single_game("TennisNoFrameSkip-v4"),
        "time_pilot": load_atari_single_game("TimePilotNoFrameSkip-v4"),
        "tutankham": load_atari_single_game("TutankhamNoFrameSkip-v4"),
        "up_n_down": load_atari_single_game("UpNDownNoFrameSkip-v4"),
        "video_pinball": load_atari_single_game("VideoPinballNoFrameSkip-v4"),
        "wizard_of_wor": load_atari_single_game("WizardOfWorNoFrameSkip-v4"),
        "yars_revenge": load_atari_single_game("YarsRevengeNoFrameSkip-v4"),
        "zaxxon": load_atari_single_game("ZaxxonNoFrameSkip-v4"),

        "mini_atari_cycle": load_mini_atari_cycle,
        "minigrid_empty8x8_unlock": load_minigrid_empty8x8_unlock
    })

    return experiments
