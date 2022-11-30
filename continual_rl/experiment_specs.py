import os
from continual_rl.experiments.experiment import Experiment
from continual_rl.experiments.tasks.make_atari_task import get_single_atari_task
from continual_rl.experiments.tasks.make_procgen_task import get_single_procgen_task
from continual_rl.experiments.tasks.make_chores_task import create_chores_tasks_from_sequence
from continual_rl.experiments.tasks.make_minihack_task import get_single_minihack_task
from continual_rl.available_policies import LazyDict
from continual_rl.experiments.tasks.state_task import StateTask
from continual_rl.experiments.tasks.image_task import ImageTask
from continual_rl.experiments.tasks.multigoal_robot_task import MultiGoalRobotTask
from continual_rl.envs.robot_demonstration_env import RobotDemonstrationEnv
#from continual_rl.envs.franka.franka_env import FrankaEnv, FrankaScoopEnv
from continual_rl.experiments.tasks.state_image_task import StateImageTask
from continual_rl.envs.maniskill_demonstration_env import ManiskillDemonstrationEnv, ManiskillEnv
from continual_rl.envs.ravens_demonstration_env import RavensSimEnvironment, RavensDemonstrationEnv
from home_robot.ros.stretch_demo_env import StretchOfflineDemoEnv, StretchOnlineDemoEnv, StretchLiveEnv


def create_atari_sequence_loader(
    task_prefix,
    game_names,
    num_timesteps=5e7,
    max_episode_steps=None,
    full_action_space=False,
    continual_testing_freq=1000,
    cycle_count=1,
):
    def loader():
        tasks = [
            get_single_atari_task(
                f"{task_prefix}_{action_space_id}",
                action_space_id,
                name,
                num_timesteps,
                max_episode_steps=max_episode_steps,
                full_action_space=full_action_space
            ) for action_space_id, name in enumerate(game_names)
        ]

        return Experiment(
            tasks,
            continual_testing_freq=continual_testing_freq,
            cycle_count=cycle_count,
        )
    return loader


def create_atari_single_game_loader(env_name):
    return lambda: Experiment(tasks=[
        # Use the env name as the task_id if it's a 1:1 mapping between env and task (as "single game" implies)
        get_single_atari_task(env_name, 0, env_name, num_timesteps=5e7, max_episode_steps=10000)
    ])


def create_procgen_sequence_loader(
    task_prefix,
    game_names,
    num_timesteps=5e6,
    task_params=None,
    add_eval_task=True,
    eval_task_override_params=None,
    continual_testing_freq=1000,
    cycle_count=1,
    start_level_ids=None
):
    task_params = task_params if task_params is not None else {}
    eval_task_override_params = eval_task_override_params if eval_task_override_params is not None else {}
    num_timesteps = num_timesteps if isinstance(num_timesteps, list) else [num_timesteps for _ in range(len(game_names))]

    def loader():
        tasks = []
        for action_space_id, name in enumerate(game_names):
            if start_level_ids is not None and "start_level" in task_params:
                task_params["start_level"] = start_level_ids[action_space_id]

            task = get_single_procgen_task(
                f"{task_prefix}_{action_space_id}",
                action_space_id,
                name,
                num_timesteps[action_space_id],
                **task_params,
            )
            tasks.append(task)

            if add_eval_task:
                eval_task = get_single_procgen_task(
                    f"{task_prefix}_{action_space_id}_eval",
                    action_space_id,
                    name,
                    0,  # not training with this task
                    eval_mode=True,
                    **{**task_params, **eval_task_override_params}  # order matters, overriding params
                )
                tasks.append(eval_task)

        return Experiment(
            tasks,
            continual_testing_freq=continual_testing_freq,
            cycle_count=cycle_count,
        )
    return loader


def create_chores_sequence_loader(
    task_prefix,
    cycle_count=1,
    continual_testing_freq=5e4,
    num_timesteps=2e6,
    max_episode_steps=1000,
    sequence_file_name="alfred_task_sequences.json"
):
    def loader():
        tasks = create_chores_tasks_from_sequence(task_prefix, sequence_file_name, num_timesteps, max_episode_steps)
        return Experiment(
            tasks,
            continual_testing_freq=continual_testing_freq,
            cycle_count=cycle_count,
        )
    return loader


def create_minihack_loader(
    task_prefix,
    env_name_pairs,
    num_timesteps=10e6,
    task_params=None,
    continual_testing_freq=1000,
    cycle_count=1,
):
    task_params = task_params if task_params is not None else {}

    def loader():
        tasks = []
        for action_space_id, pairs in enumerate(env_name_pairs):
            train_task = get_single_minihack_task(f"{task_prefix}_{action_space_id}", action_space_id, pairs[0],
                                                  num_timesteps, **task_params)
            eval_task = get_single_minihack_task(f"{task_prefix}_{action_space_id}_eval", action_space_id, pairs[1],
                                                 0, eval_mode=True, **task_params)

            tasks += [train_task, eval_task]

        return Experiment(
            tasks,
            continual_testing_freq=continual_testing_freq,
            cycle_count=cycle_count,
        )
    return loader


def create_continuous_control_tasks_loader(task_names, env_specs, demonstration_tasks, eval_modes, num_timesteps,
                                           continual_testing_freq=10000, cycle_count=1, use_state=True, image_size=[84, 84]):
    # See: https://stackoverflow.com/questions/15933493/pygame-error-no-available-video-device (maybe only necessary for CarRacing?)
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    task_class = StateImageTask if use_state else ImageTask
    kwargs = {} if use_state else {"dict_space_key": "image"}  # TODO: dict_space_key breaks StateImageTask

    def loader():
        tasks = []
        for id, task_name in enumerate(task_names):
            task = task_class(task_names[id], action_space_id=0, env_spec=env_specs[id], num_timesteps=num_timesteps[id],
                                               time_batch_size=1, eval_mode=eval_modes[id], image_size=image_size, grayscale=False,
                                               demonstration_task=demonstration_tasks[id], resize_interp_method="INTER_LINEAR", **kwargs)
            tasks.append(task)

        return Experiment(tasks=tasks, continual_testing_freq=continual_testing_freq, cycle_count=cycle_count)

    return loader


def create_continuous_control_tasks_loader_pymultigoal(task_name, num_timesteps=10e6, continual_testing_freq=10000, cycle_count=1):
    def create_env():
        import pybullet_multigoal_gym as pmg
        # Install matplotlib if you want to use imshow to view the goal images
        #import matplotlib.pyplot as plt

        # For mujoco version (in progress, not sure if all necessary):
        # pip install mujoco
        # pip install gym-robotics
        # pip install mujoco-py
        # wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz (for linux, see: https://github.com/openai/mujoco-py/ for others)
        # tar -xvf mujoco{...}.tar.gz
        # Move folder to ~/.mujoco/mujoco210
        # ...locally it is having gcc issues, so nevermind for right now (leaving instructions for now)

        camera_setup = [  # TODO: not sure why it's doubled. Also the 84 dim was originally 128
            {
                'cameraEyePosition': [-1.0, 0.25, 0.6],
                'cameraTargetPosition': [-0.6, 0.05, 0.2],
                'cameraUpVector': [0, 0, 1],
                'render_width': 84,
                'render_height': 84
            },
            {
                'cameraEyePosition': [-1.0, -0.25, 0.6],
                'cameraTargetPosition': [-0.6, -0.05, 0.2],
                'cameraUpVector': [0, 0, 1],
                'render_width': 84,
                'render_height': 84
            }
        ]

        env_fn = lambda: pmg.make_env(
            # task args ['reach', 'push', 'slide', 'pick_and_place',
            #            'block_stack', 'block_rearrange', 'chest_pick_and_place', 'chest_push']
            task=task_name,
            gripper='parallel_jaw',
            num_block=4,  # only meaningful for multi-block tasks, up to 5 blocks
            render=False,
            binary_reward=True,
            max_episode_steps=50,
            # image observation args
            image_observation=True,
            depth_image=False,
            goal_image=True,
            visualize_target=True,
            camera_setup=camera_setup,
            observation_cam_id=[0],
            goal_cam_id=1)

        return Experiment(tasks=[MultiGoalRobotTask(task_name, action_space_id=0, env_spec=env_fn, num_timesteps=num_timesteps,
                                                   time_batch_size=1, eval_mode=False, image_size=[84, 84], grayscale=False)],
                                  continual_testing_freq=continual_testing_freq, cycle_count=cycle_count)

    return create_env


def create_continuous_control_state_tasks_loader(task_name, num_timesteps=10e6, continual_testing_freq=10000, cycle_count=1):
    return lambda: Experiment(tasks=[StateTask(task_name, action_space_id=0, env_spec=task_name, num_timesteps=num_timesteps,
                                               time_batch_size=1, eval_mode=False)],
                              continual_testing_freq=continual_testing_freq, cycle_count=cycle_count)


def get_available_experiments():

    experiments = LazyDict({

        # ===============================
        # ============ Atari ============
        # ===============================

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

        "atari_6_tasks_5_cycles": create_atari_sequence_loader(
            "atari_6_tasks_5_cycles",
            ["SpaceInvadersNoFrameskip-v4",
             "KrullNoFrameskip-v4",
             "BeamRiderNoFrameskip-v4",
             "HeroNoFrameskip-v4",
             "StarGunnerNoFrameskip-v4",
             "MsPacmanNoFrameskip-v4"],
            max_episode_steps=10000,
            num_timesteps=5e7,
            full_action_space=True,
            continual_testing_freq=0.25e6,
            cycle_count=5,
         ),

        "mini_atari_3_tasks_3_cycles": create_atari_sequence_loader(
            "mini_atari_3_tasks_3_cycles",
            ["SpaceInvadersNoFrameskip-v4",
             "BeamRiderNoFrameskip-v4",
             "MsPacmanNoFrameskip-v4"],
            max_episode_steps=10000,
            num_timesteps=5e7,
            full_action_space=True,
            continual_testing_freq=0.25e6,
            cycle_count=3,
        ),

        # ===============================
        # ============ Procgen ==========
        # ===============================

        "procgen_6_tasks_5_cycles": create_procgen_sequence_loader(
            # using same games as section 5.3 of https://openreview.net/pdf?id=Qun8fv4qSby
            "procgen_6_tasks_5_cycles",
            ["climber-v0",
             "dodgeball-v0",
             "ninja-v0",
             "starpilot-v0",
             "bigfish-v0",
             "fruitbot-v0"],
            num_timesteps=5e6, # 25M steps total per environment
            task_params=dict(
                num_levels=200,
                start_level=0,
                distribution_mode="easy",
            ),
            add_eval_task=True,
            eval_task_override_params=dict(
                num_levels=0,  # full distribution
            ),
            continual_testing_freq=0.25e6,
            cycle_count=5,
        ),

        "procgen_climber_fixed_seq": create_procgen_sequence_loader(
            "procgen_climber_fixed_seq",
            ["climber-v0" for _ in range(4)],
            num_timesteps=3e6,
            task_params=dict(
                num_levels=1,
                start_level=0,
                distribution_mode="easy"
            ),
            add_eval_task=False,
            continual_testing_freq=0.1e6,
            cycle_count=3,
            start_level_ids=[3, 16, 42, 46]
        ),

        "procgen_fruitbot_fixed_seq": create_procgen_sequence_loader(
            "procgen_fruitbot_fixed_seq",
            ["fruitbot-v0" for _ in range(5)],
            num_timesteps=3e6,
            task_params=dict(
                num_levels=1,
                start_level=0,
                distribution_mode="easy"
            ),
            add_eval_task=False,
            continual_testing_freq=0.1e6,
            cycle_count=3,
            start_level_ids=[1, 10, 11, 12, 14]
        ),

        "procgen_miner_fixed_seq": create_procgen_sequence_loader(
            "procgen_miner_fixed_seq",
            ["miner-v0" for _ in range(4)],
            num_timesteps=3e6,
            task_params=dict(
                num_levels=1,
                start_level=0,
                distribution_mode="easy"
            ),
            add_eval_task=False,
            continual_testing_freq=0.1e6,
            cycle_count=3,
            start_level_ids=[16, 25, 29, 31]
        ),
        # ===============================
        # ============ MiniHack =========
        # ===============================

        "minihack_nav_paired_2_cycles": create_minihack_loader(
            "minihack_nav_paired_2_cycles",
            [
                ("Room-Random-5x5-v0", "Room-Random-15x15-v0"),
                ("Room-Dark-5x5-v0", "Room-Dark-15x15-v0"),
                ("Room-Monster-5x5-v0", "Room-Monster-15x15-v0"),
                ("Room-Trap-5x5-v0", "Room-Trap-15x15-v0"),
                ("Room-Ultimate-5x5-v0", "Room-Ultimate-15x15-v0"),
                ("Corridor-R2-v0", "Corridor-R5-v0"),
                ("Corridor-R3-v0", "Corridor-R5-v0"),
                ("KeyRoom-S5-v0", "KeyRoom-S15-v0"),
                ("KeyRoom-Dark-S5-v0", "KeyRoom-Dark-S15-v0"),
                ("River-Narrow-v0", "River-v0"),
                ("River-Monster-v0", "River-MonsterLava-v0"),
                ("River-Lava-v0", "River-MonsterLava-v0"),
                ("HideNSeek-v0", "HideNSeek-Big-v0"),
                ("HideNSeek-Lava-v0", "HideNSeek-Big-v0"),
                ("CorridorBattle-v0", "CorridorBattle-Dark-v0")
            ],
            num_timesteps=10e6,
            continual_testing_freq=1e6,
            cycle_count=2,
        ),

        # ===============================
        # ============ CHORES ===========
        # ===============================

        # Verified set, using replay_checks
        "chores_vary_objects_sequential_clean": create_chores_sequence_loader(
            "chores_vary_objects_sequential_clean",
            num_timesteps=1e6,
            max_episode_steps=1000,
            sequence_file_name='chores/vary_objects.json',
            cycle_count=2),
        "chores_vary_tasks_sequential_tp": create_chores_sequence_loader(
            "chores_vary_tasks_sequential_tp",
            num_timesteps=1e6,
            max_episode_steps=1000,
            sequence_file_name='chores/vary_tasks.json',
            cycle_count=2),
        "chores_vary_envs_sequential_pick_handtowel": create_chores_sequence_loader(
            "chores_vary_envs_sequential_pick_handtowel",
            num_timesteps=1e6,
            max_episode_steps=1000,
            sequence_file_name='chores/vary_envs.json',
            cycle_count=2),
        "chores_sequential_multi_traj": create_chores_sequence_loader(
            "chores_sequential_multi_traj",
            num_timesteps=1e6,
            max_episode_steps=1000,
            sequence_file_name='chores/multi_traj.json',
            cycle_count=2),

        # ===============================
        # ============ Continuous Action Space Environments ===========
        # ===============================

        "continuous_car_racing": create_continuous_control_tasks_loader(["CarRacing-v1"], ["CarRacing-v1"],
                                                                        demonstration_tasks=[False], eval_modes=[False],
                                                                        num_timesteps=[10e6],
                                                                        continual_testing_freq=None),
        "continuous_robot_demos": create_continuous_control_tasks_loader(
            ["FrankaTrain", "FrankaTest"],
            [lambda: RobotDemonstrationEnv(os.getenv("FRANKA_DEMOS_PATH"), (None, -100)),
            lambda: RobotDemonstrationEnv(os.getenv("FRANKA_DEMOS_PATH"), (-100, None))],
            demonstration_tasks=[True, True],
            eval_modes=[False, True],
            num_timesteps=[10e6, 1e5],
            continual_testing_freq=20000),
        "continuous_robot_1demo": create_continuous_control_tasks_loader(
            ["FrankaTrain", "FrankaTest"],
            [lambda: RobotDemonstrationEnv(os.getenv("FRANKA_DEMOS_PATH"), (None, 1)),
             lambda: RobotDemonstrationEnv(os.getenv("FRANKA_DEMOS_PATH"), (1, None))],
            demonstration_tasks=[True, True],
            eval_modes=[False, True],
            num_timesteps=[1e6, 1e5],
            continual_testing_freq=20000),
        "continuous_robot_1demo_long": create_continuous_control_tasks_loader(
            ["FrankaTrain", "FrankaTest"],
            [lambda: RobotDemonstrationEnv(os.getenv("FRANKA_DEMOS_PATH"), (None, 1)),
             lambda: RobotDemonstrationEnv(os.getenv("FRANKA_DEMOS_PATH"), (1, None))],
            demonstration_tasks=[True, True],
            eval_modes=[False, True],
            num_timesteps=[10e6, 1e4],
            continual_testing_freq=20000),
        "continuous_robot_1demo_short": create_continuous_control_tasks_loader(
            ["FrankaTrain", "FrankaTest"],
            [lambda: RobotDemonstrationEnv(os.getenv("FRANKA_DEMOS_PATH"), (None, 1)),
             lambda: RobotDemonstrationEnv(os.getenv("FRANKA_DEMOS_PATH"), (1, None))],
            demonstration_tasks=[True, True],
            eval_modes=[False, True],
            num_timesteps=[1e5, 1e4],
            continual_testing_freq=20000),
        "continuous_robot_1demo_mid": create_continuous_control_tasks_loader(
            ["FrankaTrain", "FrankaTest"],
            [lambda: RobotDemonstrationEnv(os.getenv("FRANKA_DEMOS_PATH"), (None, 1)),
             lambda: RobotDemonstrationEnv(os.getenv("FRANKA_DEMOS_PATH"), (1, None))],
            demonstration_tasks=[True, True],
            eval_modes=[False, True],
            num_timesteps=[5e5, 1e4],
            continual_testing_freq=20000),
        "continuous_franka_control": create_continuous_control_tasks_loader(
            ["FrankaControl"],
            env_specs=[lambda: FrankaScoopEnv()],
            demonstration_tasks=[False, False],
            eval_modes=[True],
            num_timesteps=[10e6],
            continual_testing_freq=None),

        "maniskill_pick_cube": create_continuous_control_tasks_loader(
            ["ManiSkillPickCube"],
            env_specs=[lambda: ManiskillDemonstrationEnv(os.getenv("MANISKILL_PATH"), "PickCube-v0",
                               valid_dataset_indices=(None, -100))],
            demonstration_tasks=[True],
            eval_modes=[False],
            num_timesteps=[10e6],
            continual_testing_freq=None),

        "maniskill_pick_cube_eval": create_continuous_control_tasks_loader(
            ["ManiSkillPickCube", "ManiSkillPickCubeEval"],
            env_specs=[lambda: ManiskillDemonstrationEnv(os.getenv("MANISKILL_PATH"), "PickCube-v0",
                                                         valid_dataset_indices=(None, -100)),
                       lambda: ManiskillDemonstrationEnv(os.getenv("MANISKILL_PATH"), "PickCube-v0",
                                                         valid_dataset_indices=(-100, None))
                       ],
            demonstration_tasks=[True, True],
            eval_modes=[False, True],
            num_timesteps=[10e6, 1e3],
            continual_testing_freq=5e4),

        "maniskill_pick_cube_sim_eval": create_continuous_control_tasks_loader(
            ["ManiSkillPickCube", "ManiSkillPickCubeSimEval"],
            env_specs=[lambda: ManiskillDemonstrationEnv(os.getenv("MANISKILL_PATH"), "PickCube-v0",
                                                         valid_dataset_indices=(None, -100)),
                       lambda: ManiskillEnv("PickCube-v0")
                       ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[10e6, 1e3],
            continual_testing_freq=5e4),

        "maniskill_pick_cube_demo_finetune": create_continuous_control_tasks_loader(
            ["ManiSkillPickCube", "ManiSkillPickCubeFineTune"],
            env_specs=[lambda: ManiskillDemonstrationEnv(os.getenv("MANISKILL_PATH"), "PickCube-v0",
                                                         valid_dataset_indices=(None, -100)),
                       lambda: ManiskillEnv("PickCube-v0")
                       ],
            demonstration_tasks=[True, False],
            eval_modes=[False, False],
            num_timesteps=[2e6, 10e6],
            continual_testing_freq=5e4),

        "maniskill_pick_cube_no_demo": create_continuous_control_tasks_loader(
            ["ManiSkillPickCubeNoDemo"],
            env_specs=[lambda: ManiskillEnv("PickCube-v0")
                       ],
            demonstration_tasks=[False],
            eval_modes=[False],
            num_timesteps=[10e6],
            continual_testing_freq=5e4),

        "ravens_put_block_base": create_continuous_control_tasks_loader(
            ["RavensPutBlockBase"],
            env_specs=[lambda: RavensSimEnvironment(assets_root="/home/spowers/Git/ravens_visual_foresight/ravens/environments/assets", task_name="put-block-base-mcts")
                       ],
            demonstration_tasks=[False],
            eval_modes=[False],
            num_timesteps=[10e6],
            continual_testing_freq=5e4,
            use_state=False),

        "ravens_put_block_base_demos": create_continuous_control_tasks_loader(
            ["RavensPutBlockBaseDemos", "RavensPutBlockBaseSim"],
            env_specs=[lambda: RavensDemonstrationEnv(
                assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "data_train/put-block-base-mcts-pp-train"), valid_dataset_indices=(None, -100),
                task_name="put-block-base-mcts"),
                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           task_name="put-block-base-mcts")
                       ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[10e6, 1e5],
            continual_testing_freq=3e3,
            use_state=False,
            image_size=[640, 480]),

        "ravens_put_block_base_stack_square_demos": create_continuous_control_tasks_loader(
            ["RavensPutBlockBaseDemos", "RavensPutBlockBaseSim", "RavensStackSquareDemos", "RavensStackSquareSim"],
            env_specs=[lambda: RavensDemonstrationEnv(
                assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "data_train/put-block-base-mcts-pp-train"),
                valid_dataset_indices=(None, -100), task_name="put-block-base-mcts", use_goal_image=True),
                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                            data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "data_train/put-block-base-mcts-pp-train"),
                           task_name="put-block-base-mcts", use_goal_image=True),

                       lambda: RavensDemonstrationEnv(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_train/stack-square-mcts-pp-train"),
                           valid_dataset_indices=(None, -100), task_name="stack-square-mcts", use_goal_image=True),
                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_train/stack-square-mcts-pp-train"),
                           task_name="stack-square-mcts", use_goal_image=True)
                       ],
            demonstration_tasks=[True, False, True, False],
            eval_modes=[False, True, False, True],
            num_timesteps=[1e4, 1e1, 1e4, 1e1],
            continual_testing_freq=3e3,
            use_state=False,
            image_size=[160, 320]),   # TODO: backwards?

        "ravens_put_block_base_debug": create_continuous_control_tasks_loader(
            ["RavensPutBlockBaseDemos", "RavensPutBlockBaseEvalDemos"],
            env_specs=[lambda: RavensDemonstrationEnv(
                assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "data_train/put-block-base-mcts-pp-train"),
                valid_dataset_indices=(0, 1), task_name="put-block-base-mcts", use_goal_image=True),
                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                            data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "data_train/put-block-base-mcts-pp-train"),
                           task_name="put-block-base-mcts", use_goal_image=True, seeds=[0]),
                       ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[1e4, 1e1],
            continual_testing_freq=3e3,
            use_state=False,
            image_size=[160, 320]),  # TODO: backwards?

        "ravens_put_block_base_debug_seed_norandact": create_continuous_control_tasks_loader(
            ["RavensPutBlockBaseDemos", "RavensPutBlockBaseEvalDemos"],
            env_specs=[lambda: RavensDemonstrationEnv(
                assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "data_train_seed_norandact/put-block-base-mcts-pp-train"),
                valid_dataset_indices=(0, 1), task_name="put-block-base-mcts", use_goal_image=True, n_demos=10),

                       lambda: RavensDemonstrationEnv(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_test_seed_norandact/put-block-base-mcts-pp-test"),
                           valid_dataset_indices=(0, 1), task_name="put-block-base-mcts", use_goal_image=True,
                           n_demos=10),
                       ],
            demonstration_tasks=[True, True],
            eval_modes=[False, True],
            num_timesteps=[1e4, 1e1],
            continual_testing_freq=3e3,
            use_state=False,
            image_size=[160, 320]),  # TODO: backwards?

        "ravens_put_block_base_debug_seed_norandact_nodisp_sim": create_continuous_control_tasks_loader(
            ["RavensPutBlockBaseDemos", "RavensPutBlockBaseEvalDemos"],
            env_specs=[lambda: RavensDemonstrationEnv(
                assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                      "data_train_seed_nodisp/put-block-base-mcts-pp-train"),
                valid_dataset_indices=(0, 1), task_name="put-block-base-mcts", use_goal_image=True, n_demos=10),

                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_train_seed_nodisp/put-block-base-mcts-pp-train"),
                           task_name="put-block-base-mcts", use_goal_image=True, seeds=[0], n_demos=10),
                       ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[1e4, 1e1],
            continual_testing_freq=3e3,
            use_state=False,
            image_size=[160, 320]),  # TODO: backwards?

        "ravens_put_block_base_stack_square_seed_norandact_nodisp_sim_10": create_continuous_control_tasks_loader(
            ["RavensPutBlockBaseDemos", "RavensPutBlockBaseEvalSim", "RavensStackSquareDemos", "RavensStackSquareSim"],
            env_specs=[lambda: RavensDemonstrationEnv(
                assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                      "data_train_nodisp_norand/put-block-base-mcts-pp-train"),
                valid_dataset_indices=(None, 10), task_name="put-block-base-mcts", use_goal_image=True, n_demos=10),

                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_train_nodisp_norand/put-block-base-mcts-pp-train"),
                           task_name="put-block-base-mcts", use_goal_image=True, seeds=[i*2 for i in range(10)], n_demos=10),

                       lambda: RavensDemonstrationEnv(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_train_nodisp_norand/stack-square-mcts-pp-train"),
                           valid_dataset_indices=(None, 10), task_name="stack-square-mcts", use_goal_image=True,
                           n_demos=10),

                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_train_nodisp_norand/stack-square-mcts-pp-train"),
                           task_name="stack-square-mcts", use_goal_image=True, seeds=[i*2 for i in range(10)], n_demos=10),
                       ],
            demonstration_tasks=[True, False, True, False],
            eval_modes=[False, True, False, True],
            num_timesteps=[2e4, 1e1, 2e4, 1e1],
            continual_testing_freq=3e3,
            use_state=False,
            image_size=[160, 320]),  # TODO: backwards?

        "ravens_put_block_base_stack_square_seed_norandact_nodisp_sim_100": create_continuous_control_tasks_loader(
            ["RavensPutBlockBaseDemos", "RavensPutBlockBaseEvalSim", "RavensStackSquareDemos", "RavensStackSquareSim"],
            env_specs=[lambda: RavensDemonstrationEnv(
                assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                      "data_train_nodisp_norand/put-block-base-mcts-pp-train"),
                valid_dataset_indices=(None, 100), task_name="put-block-base-mcts", use_goal_image=True, n_demos=100),

                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_train_nodisp_norand/put-block-base-mcts-pp-train"),
                           task_name="put-block-base-mcts", use_goal_image=True, seeds=[i * 2 for i in range(100)],
                           n_demos=100),

                       lambda: RavensDemonstrationEnv(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_train_nodisp_norand/stack-square-mcts-pp-train"),
                           valid_dataset_indices=(None, 100), task_name="stack-square-mcts", use_goal_image=True,
                           n_demos=100),

                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_train_nodisp_norand/stack-square-mcts-pp-train"),
                           task_name="stack-square-mcts", use_goal_image=True, seeds=[i * 2 for i in range(100)],
                           n_demos=100),
                       ],
            demonstration_tasks=[True, False, True, False],
            eval_modes=[False, True, False, True],
            num_timesteps=[2e4, 1e1, 2e4, 1e1],
            continual_testing_freq=3e3,
            use_state=False,
            image_size=[160, 320]),  # TODO: backwards?

        "cliport_stack-block-pyramid_10": create_continuous_control_tasks_loader(
            ["CliportStackBlockPyramidDemos", "CliportStackBlockPyramidSim"],
            env_specs=[lambda: RavensDemonstrationEnv(
                assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                data_dir=os.path.join(os.getenv('CLIPORT_ROOT'), "data/stack-block-pyramid-train"),
                valid_dataset_indices=(None, 10), task_name="stack-block-pyramid", use_goal_image=True, n_demos=10),

                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('CLIPORT_ROOT'), "data/stack-block-pyramid-train"),
                           task_name="stack-block-pyramid", use_goal_image=True, seeds=[i * 2 for i in range(10)],
                           n_demos=10),

                       ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e5, 1e1],
            continual_testing_freq=1e3,
            use_state=False,
            image_size=[320, 160]),  # TODO: backwards?

        "cliport_stack-block-pyramid_1": create_continuous_control_tasks_loader(
            ["CliportStackBlockPyramidDemos", "CliportStackBlockPyramidSim"],
            env_specs=[lambda: RavensDemonstrationEnv(
                assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                data_dir=os.path.join(os.getenv('CLIPORT_ROOT'), "data/stack-block-pyramid-train"),
                valid_dataset_indices=(None, 1), task_name="stack-block-pyramid", use_goal_image=True, n_demos=1),

                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('CLIPORT_ROOT'), "data/stack-block-pyramid-train"),
                           task_name="stack-block-pyramid", use_goal_image=True, seeds=[i * 2 for i in range(1)],
                           n_demos=1),

                       ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e4, 1e1],
            continual_testing_freq=1e3,
            use_state=False,
            image_size=[320, 160]),  # TODO: backwards?

        "stretch_oven": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[#lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_10_2022_20.56.56.054503.h5"),
                       #lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_11_2022_17.45.14.346868.h5"),
                       # 5 pulls: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_11_2022_18.16.56.805601.h5"),
                       # 1 simple pull: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_13_2022_15.11.34.383975.h5"),
                       # 5 "augmented" pulls: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_13_2022_16.35.22.349653.h5"),
                       lambda: StretchOfflineDemoEnv(demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/oven_0"),
                       lambda: StretchLiveEnv(
                           demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/oven_0", use_true_action=False),
                       ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=5e3,
            use_state=True,
            image_size=[224, 224]),  # TODO

        "stretch_oven_1": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                # lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_10_2022_20.56.56.054503.h5"),
                # lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_11_2022_17.45.14.346868.h5"),
                # 5 pulls: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_11_2022_18.16.56.805601.h5"),
                # 1 simple pull: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_13_2022_15.11.34.383975.h5"),
                # 5 "augmented" pulls: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_13_2022_16.35.22.349653.h5"),
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/oven_1"),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/oven_1",
                    use_true_action=False),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=5e3,
            use_state=True,
            image_size=[224, 224]),  # TODO

        "stretch_oven_debug": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                # lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_10_2022_20.56.56.054503.h5"),
                # lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_11_2022_17.45.14.346868.h5"),
                # 5 pulls: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_11_2022_18.16.56.805601.h5"),
                # 1 simple pull: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_13_2022_15.11.34.383975.h5"),
                # 5 "augmented" pulls: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_13_2022_16.35.22.349653.h5"),
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/oven_debug_2"),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/oven_debug_2",
                    use_true_action=False),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=2e5,
            use_state=True,
            image_size=[224, 224]),  # TODO

        "stretch_oven_1_agg": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                # lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_10_2022_20.56.56.054503.h5"),
                # lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_11_2022_17.45.14.346868.h5"),
                # 5 pulls: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_11_2022_18.16.56.805601.h5"),
                # 1 simple pull: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_13_2022_15.11.34.383975.h5"),
                # 5 "augmented" pulls: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_13_2022_16.35.22.349653.h5"),
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/oven_1_agg", state_augmentation_scale=0),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/oven_1_agg",
                    use_true_action=False),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=1e5,
            use_state=True,
            image_size=[224, 224]),  # TODO

        "stretch_oven_1_progressive": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                # lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_10_2022_20.56.56.054503.h5"),
                # lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_11_2022_17.45.14.346868.h5"),
                # 5 pulls: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_11_2022_18.16.56.805601.h5"),
                # 1 simple pull: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_13_2022_15.11.34.383975.h5"),
                # 5 "augmented" pulls: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_13_2022_16.35.22.349653.h5"),
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/oven_1_progressive"),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/oven_1_progressive",
                    use_true_action=False),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=1e5,
            use_state=True,
            image_size=[224, 224]),  # TODO

        "stretch_oven_1_progressive_reuse": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                # lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_10_2022_20.56.56.054503.h5"),
                # lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_11_2022_17.45.14.346868.h5"),
                # 5 pulls: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_11_2022_18.16.56.805601.h5"),
                # 1 simple pull: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_13_2022_15.11.34.383975.h5"),
                # 5 "augmented" pulls: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_13_2022_16.35.22.349653.h5"),
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/oven_1_progressive"),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/oven_1_progressive",
                    use_true_action=False),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=2e5,
            use_state=True,
            image_size=[224, 224]),  # TODO

        "stretch_oven_3_progressive_long": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                # lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_10_2022_20.56.56.054503.h5"),
                # lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_11_2022_17.45.14.346868.h5"),
                # 5 pulls: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_11_2022_18.16.56.805601.h5"),
                # 1 simple pull: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_13_2022_15.11.34.383975.h5"),
                # 5 "augmented" pulls: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_13_2022_16.35.22.349653.h5"),
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/oven_3_progressive"),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/oven_3_progressive",
                    use_true_action=False),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=1e5,
            use_state=True,
            image_size=[224, 224]),  # TODO

        "stretch_oven_progressive_debug": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                # lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_10_2022_20.56.56.054503.h5"),
                # lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_11_2022_17.45.14.346868.h5"),
                # 5 pulls: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_11_2022_18.16.56.805601.h5"),
                # 1 simple pull: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_13_2022_15.11.34.383975.h5"),
                # 5 "augmented" pulls: lambda: StretchOfflineDemoEnv(demo_path="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/demo_Oct_13_2022_16.35.22.349653.h5"),
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/progressive_demos_debug"),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/progressive_demos_debug",
                    use_true_action=False),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=1e5,
            use_state=True,
            image_size=[224, 224]),  # TODO

        "stretch_oven_4_dagger": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/oven_4_dagger",
                    state_augmentation_scale=1),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/oven_4_dagger",
                    use_true_action=False),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=1e5,
            use_state=True,
            image_size=[224, 224]),

        "stretch_oven_key_frames": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/key_frames",
                    state_augmentation_scale=0, use_key_frames=True),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/key_frames",
                    use_true_action=False, use_key_frames=True),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=15000,
            use_state=True,
            image_size=[224, 224]),

        "stretch_oven_key_frames_absolute": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/key_frames",
                    state_augmentation_scale=0, use_key_frames=True, command_absolute=True),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/key_frames",
                    use_true_action=False, use_key_frames=True, command_absolute=True),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=15000,
            use_state=True,
            image_size=[224, 224]),

        "stretch_oven_key_frames_absolute_camera_state": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/key_frames",
                    state_augmentation_scale=0, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/key_frames",
                    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=15000,
            use_state=True,
            image_size=[1280, 720]),  # TODO: ...inverted?

        "stretch_oven_key_frames_camera_state": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/key_frames",
                    state_augmentation_scale=0, use_key_frames=True, command_absolute=False, camera_info_in_state=True),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/key_frames",
                    use_true_action=False, use_key_frames=True, command_absolute=False, camera_info_in_state=True),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=15000,
            use_state=True,
            image_size=[1280, 720]),  # TODO: ...inverted?

        "stretch_oven_key_frames_2_camera_state": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/key_frames_2",
                    state_augmentation_scale=0, use_key_frames=True, command_absolute=False, camera_info_in_state=True),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/key_frames_2",
                    use_true_action=False, use_key_frames=True, command_absolute=False, camera_info_in_state=True),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=5000,
            use_state=True,
            image_size=[1280, 720]),  # TODO: ...inverted?

        "stretch_oven_key_frames_3_camera_state": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/key_frames_3",
                    state_augmentation_scale=0, use_key_frames=True, command_absolute=False, camera_info_in_state=True),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/key_frames_3",
                    use_true_action=False, use_key_frames=True, command_absolute=False, camera_info_in_state=True),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=5000,
            use_state=True,
            image_size=[1280, 720]),  # TODO: ...inverted?

        "stretch_oven_key_frames_4_camera_state": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/key_frames_4",
                    state_augmentation_scale=0, use_key_frames=True, command_absolute=False, camera_info_in_state=True),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/key_frames_4",
                    use_true_action=False, use_key_frames=True, command_absolute=False, camera_info_in_state=True),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=500000,
            use_state=True,
            image_size=[1280, 720]),  # TODO: ...inverted?

        "stretch_oven_key_frames_2_absolute_camera_state": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/key_frames_2",
                    state_augmentation_scale=0, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/key_frames_2",
                    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=5000,
            use_state=True,
            image_size=[1280, 720]),  # TODO: ...inverted?

        "stretch_right_oven_key_frames_2_absolute_camera_state": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames",
                    state_augmentation_scale=0, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames",
                    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=5000,
            use_state=True,
            image_size=[1280, 720]),  # TODO: ...inverted?

        "stretch_right_oven_key_frames_2_camera_state": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames",
                    state_augmentation_scale=0, use_key_frames=True, command_absolute=False, camera_info_in_state=True),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames",
                    use_true_action=False, use_key_frames=True, command_absolute=False, camera_info_in_state=True),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=5000,
            use_state=True,
            image_size=[1280, 720]),  # TODO: ...inverted?

        "stretch_right_oven_key_frames_2_camera_state_perturb_live": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames/single",
                    state_augmentation_scale=0, use_key_frames=True, command_absolute=False, camera_info_in_state=True),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames/single",
                    use_true_action=False, use_key_frames=True, command_absolute=False, camera_info_in_state=True,
                    perturb_start_state=True),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=10000,
            use_state=True,
            image_size=[1280, 720]),  # TODO: ...inverted?

        "stretch_right_oven_only": create_continuous_control_tasks_loader(
            # TODO: perturb is a lie
            ["StretchOvenOfflineDemosRight", "StretchPredictedAction"],
            env_specs=[
                lambda: StretchOfflineDemoEnv(
                    #demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames/single",
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames_2_campose",
                    state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
                #lambda: StretchOfflineDemoEnv(
                #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames",
                #    state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
                lambda: StretchLiveEnv(
                    #demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames/single",
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames_2_campose",
                    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
                    perturb_start_state=False),
                #lambda: StretchLiveEnv(
                #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames",
                #    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
                #    perturb_start_state=True),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],  # 25000
            continual_testing_freq=5000,
            use_state=True,
            image_size=[1280, 720],
            cycle_count=2),

        "stretch_right_oven_key_frames_2_camera_state_perturb_live_abs": create_continuous_control_tasks_loader(  # TODO: perturb is a lie
            ["StretchOvenOfflineDemosLeft", "StretchOvenOfflineDemosRight", "StretchPredictedAction"],
            env_specs=[
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames/single",
                    state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames",
                    state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
                #lambda: StretchLiveEnv(
                #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames/single",
                #    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
                #    perturb_start_state=False),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames",
                    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
                    perturb_start_state=True),
            ],
            demonstration_tasks=[True, True, False],
            eval_modes=[False, False, True],
            num_timesteps=[25000, 25000, 1e1],
            continual_testing_freq=5000,
            use_state=True,
            image_size=[1280, 720],
        cycle_count=2),

        "stretch_two_oven_offline_only": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemosLeft", "StretchOvenOfflineDemosRight", "LiveRight", "LiveLeft"],
            env_specs=[
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames_2_campose",
                    state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames",
                    state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
                 lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames_2_campose",
                    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
                    perturb_start_state=False),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames",
                    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
                    perturb_start_state=False),
            ],
            demonstration_tasks=[True, True, False, False],
            eval_modes=[False, False, True, True],
            num_timesteps=[5000, 5000, 1e1, 1e1],
            #num_timesteps=[5000, 5000, 1e1],
            continual_testing_freq=1000,
            use_state=True,
            image_size=[1280, 720],
            cycle_count=1),

        "stretch_left_oven_ee": create_continuous_control_tasks_loader(
            # TODO: perturb is a lie
            ["StretchOvenOfflineDemosLeft", "StretchOvenOfflineDemosRight", "StretchPredictedAction"],
            env_specs=[
                #lambda: StretchOfflineDemoEnv(
                #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames/single",
                #    state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames",
                    state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
                # lambda: StretchLiveEnv(
                #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames/single",
                #    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
                #    perturb_start_state=False),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames",
                    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
                    perturb_start_state=False),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[25000, 1e1],
            continual_testing_freq=5000,
            use_state=True,
            image_size=[1280, 720],
            cycle_count=2),

        "stretch_left_oven_ee_shift": create_continuous_control_tasks_loader(
            # TODO: perturb is a lie
            ["StretchOvenOfflineDemosLeft", "StretchPredictedAction", "Live"],
            env_specs=[
                # lambda: StretchOfflineDemoEnv(
                #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames/single",
                #    state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames_shift",
                    state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames_2_campose",
                    state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
                # lambda: StretchLiveEnv(
                #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames/single",
                #    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
                #    perturb_start_state=False),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames_shift",
                    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
                    perturb_start_state=False),
            ],
            demonstration_tasks=[True, True, False],
            eval_modes=[False, False, True],
            num_timesteps=[10000, 10000, 1e1],
            continual_testing_freq=1000,
            use_state=True,
            image_size=[1280, 720],
            cycle_count=2),

    "stretch_large_waffle_iron": create_continuous_control_tasks_loader(
        ["StretchOvenOfflineDemosLeft", "Live"],
        env_specs=[
            lambda: StretchOfflineDemoEnv(
                demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/large_waffle_iron",
                state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
            lambda: StretchLiveEnv(
                demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/large_waffle_iron",
                use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
                perturb_start_state=False),
            #lambda: StretchOfflineDemoEnv(
            #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames_shift",
            #    state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
            #lambda: StretchOfflineDemoEnv(
            #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames_2_campose",
            #    state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
            # lambda: StretchLiveEnv(
            #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames/single",
            #    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
            #    perturb_start_state=False),
            #lambda: StretchLiveEnv(
            #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames_shift",
            #    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
            #    perturb_start_state=False),
        ],
        demonstration_tasks=[True, False],
        eval_modes=[False, True],
        num_timesteps=[10000, 1e1],
        continual_testing_freq=1000,
        use_state=True,
        image_size=[1280, 720],
        cycle_count=2),

    "stretch_large_waffle_iron_left_oven_shift": create_continuous_control_tasks_loader(
        ["StretchDemo1", "StretchDemo2", "Live1"], #, "Live2"],
        env_specs=[
            lambda: StretchOfflineDemoEnv(
                demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/large_waffle_iron",
                state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
            #lambda: StretchOfflineDemoEnv(
            #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames_shift",
            #    state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
            lambda: StretchOfflineDemoEnv(
                demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames_shift",
                state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
            # lambda: StretchLiveEnv(
            #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames/single",
            #    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
            #    perturb_start_state=False),
            #lambda: StretchLiveEnv(
            #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames_shift",
            #    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
            #    perturb_start_state=False),
            lambda: StretchLiveEnv(
                demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/large_waffle_iron",
                use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
                perturb_start_state=False),
            lambda: StretchLiveEnv(
                demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames_shift",
                use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
                perturb_start_state=False),
        ],
        demonstration_tasks=[True, True, False, False],
        eval_modes=[False, False, True, True],
        num_timesteps=[5000, 5000, 1e1, 1e1],
        continual_testing_freq=1000,
        use_state=True,
        image_size=[1280, 720],
        cycle_count=1),

    "stretch_large_waffle_iron_shift_left_oven_shift": create_continuous_control_tasks_loader(
        ["StretchDemo1", "StretchDemo2", "Live1"], #, "Live2"],
        env_specs=[
            lambda: StretchOfflineDemoEnv(
                demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/large_waffle_iron_shift_2",
                state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
            #lambda: StretchOfflineDemoEnv(
            #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames_shift",
            #    state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
            lambda: StretchOfflineDemoEnv(
                demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames_shift",
                state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
            # lambda: StretchLiveEnv(
            #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames/single",
            #    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
            #    perturb_start_state=False),
            #lambda: StretchLiveEnv(
            #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames_shift",
            #    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
            #    perturb_start_state=False),
            #lambda: StretchLiveEnv(
            #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/large_waffle_iron",
            #    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
            #    perturb_start_state=False),
            lambda: StretchLiveEnv(
                demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames_shift",
                use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
                perturb_start_state=False),
        ],
        demonstration_tasks=[True, True, False, False],
        eval_modes=[False, False, True, True],
        num_timesteps=[25000, 25000, 1e1, 1e1],
        continual_testing_freq=1000,
        use_state=True,
        image_size=[1280, 720],
        cycle_count=1),

    "stretch_large_waffle_iron_right_oven": create_continuous_control_tasks_loader(
        ["StretchDemo1", "StretchDemo2", "Live1", "Live2"],
        env_specs=[
            lambda: StretchOfflineDemoEnv(
                demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/large_waffle_iron",
                state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
            #lambda: StretchOfflineDemoEnv(
            #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames_shift",
            #    state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
            lambda: StretchOfflineDemoEnv(
                demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames_2_campose",
                state_augmentation_scale=3, use_key_frames=True, command_absolute=True, camera_info_in_state=True),
            # lambda: StretchLiveEnv(
            #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames/single",
            #    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
            #    perturb_start_state=False),
            #lambda: StretchLiveEnv(
            #    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/left_oven_key_frames_shift",
            #    use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
            #    perturb_start_state=False),
            lambda: StretchLiveEnv(
                demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/large_waffle_iron",
                use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
                perturb_start_state=False),
            lambda: StretchLiveEnv(
                demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames_2_campose",
                use_true_action=False, use_key_frames=True, command_absolute=True, camera_info_in_state=True,
                perturb_start_state=False),
        ],
        demonstration_tasks=[True, True, False, False],
        eval_modes=[False, False, True, True],
        num_timesteps=[15000, 18000, 1e1, 1e1],
        continual_testing_freq=1000,
        use_state=True,
        image_size=[1280, 720],
        cycle_count=1),

    "stretch_right_oven_key_frames_2_224x224": create_continuous_control_tasks_loader(
            ["StretchOvenOfflineDemos", "StretchPredictedAction"],
            env_specs=[
                lambda: StretchOfflineDemoEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames",
                    state_augmentation_scale=0, use_key_frames=True, command_absolute=False, camera_info_in_state=False),
                lambda: StretchLiveEnv(
                    demo_dir="/home/spowers/Git/home_robot/src/home_robot/ros/tmp/demo_data/right_oven_key_frames",
                    use_true_action=False, use_key_frames=True, command_absolute=False, camera_info_in_state=False),
            ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[2e6, 1e1],
            continual_testing_freq=25000,
            use_state=True,
            image_size=[224, 224]),  # TODO: ...inverted?

        "ravens_put_block_base_debug_demos_seed": create_continuous_control_tasks_loader(
            ["RavensPutBlockBaseDemos", "RavensPutBlockBaseEvalDemos"],
            env_specs=[lambda: RavensDemonstrationEnv(
                assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                      "data_train_seed/put-block-base-mcts-pp-train"),
                valid_dataset_indices=(0, 1), task_name="put-block-base-mcts", use_goal_image=True, n_demos=10),
                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_test_seed/put-block-base-mcts-pp-train"),
                           task_name="put-block-base-mcts", use_goal_image=True, seeds=[0], n_demos=10),
                       ],
            demonstration_tasks=[True, False],
            eval_modes=[False, True],
            num_timesteps=[1e4, 1e1],
            continual_testing_freq=3e3,
            use_state=False,
            image_size=[160, 320]),  # TODO: backwards?

        "ravens_seq_demos": create_continuous_control_tasks_loader(
            ["RavensStackTowerDemos", "RavensStackTowerSim", "RavensPutBlockBaseDemos", "RavensPutBlockBaseSim", "RavensStackSquareDemos", "RavensStackSquareSim"],
            env_specs=[lambda: RavensDemonstrationEnv(
                assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "data_train/stack-tower-mcts-pp-train"),
                valid_dataset_indices=(None, -100), task_name="stack-tower-mcts", use_goal_image=True),
                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                            data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "data_train/stack-tower-mcts-pp-train"),
                           task_name="stack-tower-mcts", use_goal_image=True),

                lambda: RavensDemonstrationEnv(
                assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "data_train/put-block-base-mcts-pp-train"),
                valid_dataset_indices=(None, -100), task_name="put-block-base-mcts", use_goal_image=True),
                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                            data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "data_train/put-block-base-mcts-pp-train"),
                           task_name="put-block-base-mcts", use_goal_image=True),

                       lambda: RavensDemonstrationEnv(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_train/stack-square-mcts-pp-train"),
                           valid_dataset_indices=(None, -100), task_name="stack-square-mcts", use_goal_image=True),
                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_train/stack-square-mcts-pp-train"),
                           task_name="stack-square-mcts", use_goal_image=True)
                       ],
            demonstration_tasks=[True, False, True, False, True, False],
            eval_modes=[False, True, False, True, False, True],
            num_timesteps=[1e4, 1e1, 1e4, 1e1, 1e4, 1e1],
            continual_testing_freq=5e2,
            use_state=False,
            image_size=[160, 320]),

        "ravens_seq_demos_infreq_cl": create_continuous_control_tasks_loader(
            ["RavensStackTowerDemos", "RavensStackTowerSim", "RavensPutBlockBaseDemos", "RavensPutBlockBaseSim",
             "RavensStackSquareDemos", "RavensStackSquareSim"],
            env_specs=[lambda: RavensDemonstrationEnv(
                        assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                        data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "data_train/stack-tower-mcts-pp-train"),
                        valid_dataset_indices=(None, -100), task_name="stack-tower-mcts", use_goal_image=True),
                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_train/stack-tower-mcts-pp-train"),
                           task_name="stack-tower-mcts", use_goal_image=True),

                       lambda: RavensDemonstrationEnv(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_train/put-block-base-mcts-pp-train"),
                           valid_dataset_indices=(None, -100), task_name="put-block-base-mcts", use_goal_image=True),
                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_train/put-block-base-mcts-pp-train"),
                           task_name="put-block-base-mcts", use_goal_image=True),

                       lambda: RavensDemonstrationEnv(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_train/stack-square-mcts-pp-train"),
                           valid_dataset_indices=(None, -100), task_name="stack-square-mcts", use_goal_image=True),
                       lambda: RavensSimEnvironment(
                           assets_root=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'), "ravens/environments/assets"),
                           data_dir=os.path.join(os.getenv('RAVENS_FORESIGHT_DIR'),
                                                 "data_train/stack-square-mcts-pp-train"),
                           task_name="stack-square-mcts", use_goal_image=True)
                       ],
            demonstration_tasks=[True, False, True, False, True, False],
            eval_modes=[False, True, False, True, False, True],
            num_timesteps=[2e4, 1e1, 2e4, 1e1, 2e4, 1e1],
            continual_testing_freq=3e3,
            use_state=False,
            image_size=[160, 320]),

        "continuous_pendulum": create_continuous_control_state_tasks_loader("Pendulum-v1", continual_testing_freq=None),
        "continuous_mountaincar": create_continuous_control_state_tasks_loader("MountainCarContinuous-v0", continual_testing_freq=None),
        "pymultigoal_stack": create_continuous_control_tasks_loader_pymultigoal("block_stack", continual_testing_freq=None),
        "pymultigoal_reach": create_continuous_control_tasks_loader_pymultigoal("reach", continual_testing_freq=None)

    })

    return experiments
