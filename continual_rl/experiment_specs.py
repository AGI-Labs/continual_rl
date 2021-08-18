from continual_rl.experiments.experiment import Experiment
from continual_rl.experiments.tasks.make_atari_task import get_single_atari_task
from continual_rl.experiments.tasks.make_procgen_task import get_single_procgen_task
from continual_rl.experiments.tasks.make_thor_task import create_alfred_tasks_from_sequence
from continual_rl.available_policies import LazyDict

import os
import json


def create_atari_cycle_loader(game_names, num_timesteps, max_episode_steps=None, continual_testing_freq=5e4, cycle_count=5, full_action_space=False):
    return lambda: Experiment(tasks=[
        get_single_atari_task(
            action_space_id,
            name,
            num_timesteps,
            max_episode_steps=max_episode_steps,
            full_action_space=full_action_space
        ) for action_space_id, name in enumerate(game_names)
    ], continual_testing_freq=continual_testing_freq, cycle_count=cycle_count)


def create_atari_single_game_loader(env_name):
    return lambda: Experiment(tasks=[
        get_single_atari_task(0, env_name, num_timesteps=5e7, max_episode_steps=10000)
    ])


def create_procgen_cycle_loader(
    game_names,
    num_timesteps,
    cycle_count=5,
    continual_testing_freq=5e4,
    task_params={},
    add_eval_task=True,
    eval_task_override_params={},
):
    tasks = []
    for action_space_id, name in enumerate(game_names):
        task = get_single_procgen_task(
            action_space_id,
            name,
            num_timesteps,
            **task_params,
        )
        tasks.append(task)

        if add_eval_task:
            eval_task = get_single_procgen_task(
                action_space_id,
                name,
                0,  # not training with this task
                eval_mode=True,
                **{**task_params, **eval_task_override_params}  # order matters, overriding params
            )
            tasks.append(eval_task)

    return lambda: Experiment(tasks=tasks, continual_testing_freq=continual_testing_freq, cycle_count=cycle_count)


def create_alfred_demo_based_thor_loader(
    cycle_count=1,
    continual_testing_freq=5e4,
    num_timesteps=2e6,
    max_episode_steps=1000,
    sequence_file_name="alfred_task_sequences.json"
):
    tasks = create_alfred_tasks_from_sequence(sequence_file_name, num_timesteps, max_episode_steps)
    return lambda: Experiment(tasks=tasks, continual_testing_freq=continual_testing_freq, cycle_count=cycle_count)


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

        # # Default action space for games in used in atari_cycle
        # {
        #     0: 'SpaceInvadersNoFrameskip-v4',
        #     1: 'KrullNoFrameskip-v4',
        #     2: 'BeamRiderNoFrameskip-v4',
        #     3: 'HeroNoFrameskip-v4',
        #     4: 'StarGunnerNoFrameskip-v4',
        #     5: 'MsPacmanNoFrameskip-v4'
        # }
        # {0: Discrete(6), 1: Discrete(18), 2: Discrete(9), 3: Discrete(18), 4: Discrete(18), 5: Discrete(9)}

        "atari_6_tasks_5_cycles": create_atari_cycle_loader(
            ["SpaceInvadersNoFrameskip-v4",
             "KrullNoFrameskip-v4",
             "BeamRiderNoFrameskip-v4",
             "HeroNoFrameskip-v4",
             "StarGunnerNoFrameskip-v4",
             "MsPacmanNoFrameskip-v4"],
            max_episode_steps=10000,
            num_timesteps=5e7,
            continual_testing_freq=0.25e6,
            cycle_count=5,
            full_action_space=True,
        ),

        "mini_atari_3_tasks_3_cycles": create_atari_cycle_loader(
            ["SpaceInvadersNoFrameskip-v4",
             "BeamRiderNoFrameskip-v4",
             "MsPacmanNoFrameskip-v4"],
            max_episode_steps=10000,
            num_timesteps=5e7,
            continual_testing_freq=0.25e6,
            cycle_count=3,
            full_action_space=True,
        ),

        "procgen_6_tasks_5_cycles": create_procgen_cycle_loader(
            # using same games as section 5.3 of https://openreview.net/pdf?id=Qun8fv4qSby
            ["climber-v0",
             "dodgeball-v0",
             "ninja-v0",
             "starpilot-v0",
             "bigfish-v0",
             "fruitbot-v0"],
            # 25M steps total per environment
            cycle_count=5,
            num_timesteps=5e6,
            continual_testing_freq=0.25e6,
            task_params=dict(
                num_levels=200,
                start_level=0,
                distribution_mode="easy",
            ),
            add_eval_task=True,
            eval_task_override_params=dict(
                num_levels=0,  # full distribution
            ),
        ),

        "alfred_demo_based_thor": create_alfred_demo_based_thor_loader(num_timesteps=1.35e6),

        "alfred_demo_based_thor_no_crl": create_alfred_demo_based_thor_loader(
            continual_testing_freq=None
        ),

        "alfred_demo_based_thor_250_steps": create_alfred_demo_based_thor_loader(num_timesteps=1.35e6, max_episode_steps=250),
        "alfred_demo_based_thor_250_steps_2": create_alfred_demo_based_thor_loader(num_timesteps=2e6, max_episode_steps=250, sequence_file_name='alfred_task_sequences_2.json'),

    })

    return experiments
