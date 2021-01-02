import numpy as np
import torch
from ete3 import Tree, NodeStyle, TreeStyle
from PIL import Image, ImageDraw
import moviepy.editor as mpy
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.config_base import UnknownExperimentConfigEntry
from continual_rl.experiments.environment_runners.environment_runner_batch import EnvironmentRunnerBatch
from continual_rl.experiments.environment_runners.full_parallel.environment_runner_full_parallel import EnvironmentRunnerFullParallel
from continual_rl.policies.sane.sane_policy_config import SanePolicyConfig
from continual_rl.policies.sane.sane_timestep_data import SaneTimestepDataBatch
from continual_rl.policies.sane.hypothesis_directory.directory_data import DirectoryData
from continual_rl.policies.sane.hypothesis_directory.directory_usage_accessor import DirectoryUsageAccessor
from continual_rl.policies.sane.hypothesis_directory.directory_updater import DirectoryUpdater
from continual_rl.policies.sane.hypothesis_directory.utils import Utils

import matplotlib
try:
    matplotlib.use("Qt5Agg")  # Otherwise ETE's render on mac throws an exception
except ImportError:
    pass  # E.g. in headless environments, that won't work, so just leave it


class SanePolicy(PolicyBase):
    """
    Layered Actor-Critic Ensemble Networks implementation.
    """
    def __init__(self, config: SanePolicyConfig, observation_space, action_spaces):  # Switch to your config type
        super().__init__()
        self._config = config

        # Needs to run once, not totally sure of the best place
        #multiprocessing.set_start_method('spawn')
        torch.multiprocessing.set_sharing_strategy('file_system')  # Attempting to bypass "too many open files". (May also be lack of deepcopy in replay buffer)

        self._action_size_map = action_spaces
        self._common_action_size = np.array([space.n for space in action_spaces.values()]).max()
        self._num_parallel_envs = config.num_parallel_envs
        self._timesteps_per_collection = config.timesteps_per_collection
        self._render_freq = 500000 // config.num_parallel_envs  # timesteps
        self._random_action_rate = config.random_action_rate  # Not applied in test
        self._total_timesteps = 0

        self._directory_data = DirectoryData(config.use_cuda, config.output_dir, observation_space,
                                             self._common_action_size, config,
                                             config.replay_buffer_size, config.filter_learning_rate,
                                             config.is_sync)
        self._directory_updater = DirectoryUpdater(self._directory_data)
        self._directory_usage_accessor = DirectoryUsageAccessor(self._directory_data)

        # State info
        self._update_processes_state_info = None
        self._last_info_to_store_override = None

    @property
    def _logger(self):
        logger = Utils.create_logger(f"{self._config.output_dir}/policy.log")
        return logger

    def get_environment_runner(self, task_spec):
        if task_spec.eval_mode:
            # During eval mode, don't get the update bundles for the processes that are generated during train
            num_parallel_envs = 1
            create_update_bundle = None
            receive_update_bundle = None
        else:
            num_parallel_envs = self._num_parallel_envs
            create_update_bundle = self.create_update_bundle
            receive_update_bundle = self.receive_update_bundle

        if self._config.env_mode == "parallel":
            # Usage process doesn't need the hypothesis updater, which has Queues, which make starting new Processes sad
            # TODO: more neatly
            updater = self._directory_updater
            self._directory_updater = None

            environment_runner = EnvironmentRunnerFullParallel(self, num_parallel_processes=num_parallel_envs,
                                                               timesteps_per_collection=self._timesteps_per_collection,
                                                               render_collection_freq=self._render_freq,
                                                               create_update_process_bundle=create_update_bundle,
                                                               receive_update_process_bundle=receive_update_bundle,
                                                               output_dir=self._config.output_dir)

            self._directory_updater = updater

        elif self._config.env_mode == "batch":  # TODO: test....?
            environment_runner = EnvironmentRunnerBatch(self, num_parallel_envs=num_parallel_envs,
                                                        timesteps_per_collection=self._timesteps_per_collection,
                                                        render_collection_freq=self._render_freq,
                                                        output_dir=self._config.output_dir)
        else:
            raise UnknownExperimentConfigEntry(f"Env_mode {self._config.env_mode} not recognized")

        return environment_runner

    def receive_update_bundle(self, update_bundle):
        directory_data, last_info_to_store = update_bundle
        self._directory_data.set_from(directory_data)
        self._last_info_to_store_override = last_info_to_store

    def create_update_bundle(self):
        bundle = self._update_processes_state_info
        self._update_processes_state_info = None
        return bundle

    def _compute_actions_internal(self, x, action_size, eval_mode, counter_lock, per_episode_storage):
        """
        Compute the correct action to take, given the input.
        :param x: The observation *without* a batch dimension. This is to enforce that we are only computing one timestep at a time here. (No parallelization of env support here.)
        :param action_size: Used because we might have multiple tasks, and we want to know what subset to sample over.
        :return: selected_action, policy_info: The action to execute, and a black box of information that should be saved off
            in a buffer as a list of (policy_info, reward) and then passed back into train()
        """
        assert len(per_episode_storage) == len(x) == 1, "Only supporting 1 environment"

        hypothesis, step_creation_buffer = self._directory_usage_accessor.get(x, eval_mode, per_episode_storage[0])  # Get() is idempotent (or...should be, double check TODO)
        policy, value, replay_entry = self._directory_usage_accessor.hypothesis_accessor.forward(hypothesis, x,
                                                                                  eval_mode=eval_mode, counter_lock=counter_lock)  # We don't use hypothesis.train()/eval() because it's more burdensome/error-prone to set it for all hypotheses than just using this bool

        assert not torch.isnan(policy).any(), f"Found a NaN in hypothesis {hypothesis.friendly_name}, aborting."
        random_action_rate = self._random_action_rate if not eval_mode else 0
        log_probs, selected_action, _ = Utils.get_log_probs(hypothesis, policy, action_size,
                                                            random_action_rate=random_action_rate, selected_action=None, verbose=False)

        # Policy info is all of the stuff we will need for training the policy. Collecting it like this just so the only object
        # that needs to care about what is inside of it is this HypothesisPolicy object - TODO: why is it being picky about numpy? My other procs aren't
        converted_replay = replay_entry.input_state.cpu().numpy() if replay_entry is not None else None
        policy_info = (hypothesis.unique_id, selected_action.unsqueeze(0).detach().cpu().numpy(), action_size,
                       value.unsqueeze(0).detach().cpu().numpy(), converted_replay)

        selected_actions = [selected_action.cpu().numpy()]  # TODO: otherwise corrupted somewhere?
        policy_infos = [policy_info]

        return selected_actions, policy_infos, step_creation_buffer

    def _get_episode_storages(self, num_envs, last_info_to_store):
        per_episode_storages = []
        for index in range(num_envs):
            # Maintain episode storage except when crossing episode boundaries
            if last_info_to_store is None or last_info_to_store.done[index]:
                per_episode_storage = {}
            else:
                per_episode_storage = last_info_to_store.per_episode_storage[index]

            per_episode_storages.append(per_episode_storage)

        return per_episode_storages

    def compute_action(self, observation, task_id, last_info_to_store, eval_mode):
        # We may have received update data from the main process, so override our current info to store.
        # Specifically this is currently manipulating the creation_buffer, so we don't infinitely create data.
        if self._last_info_to_store_override is not None:
            last_info_to_store = self._last_info_to_store_override
            self._last_info_to_store_override = None

        per_episode_storages = self._get_episode_storages(len(observation), last_info_to_store)
        action_size = self._action_size_map[task_id].n

        # Single shared creation buffer - grab it or create it, as necessary
        if last_info_to_store is None:
            creation_buffer = {}
        else:
            creation_buffer = last_info_to_store.creation_buffer  # last_info_to_store's creation buffer is the same object as the current one's

        # We'll put usage bundles on the InfoToStore (i.e. updated per timestep)
        actions, policy_infos, step_creation_buffer = self._compute_actions_internal(observation, action_size, eval_mode, None, per_episode_storages)

        # Just using a helper function
        DirectoryUpdater.update_creation_buffer(creation_buffer, step_creation_buffer)

        data_to_store = SaneTimestepDataBatch(policy_infos, per_episode_storages)
        data_to_store.creation_buffer = creation_buffer

        return actions, data_to_store

    def _print_hierarchy(self, layer_id, directory, tree=None):  # TODO: these names are terrible
        if tree is None:
            tree = Tree()

        for entry in directory:
            #policy = entry.prototype.policy if entry.is_long_term else entry.policy
            policy = entry.policy
            self._logger.info(f"Layer {layer_id} ({entry.friendly_name}: usage {entry.usage_count}, non-decayed: {entry.non_decayed_usage_count}): {policy}")

            # Style node so it decays from bright pink to blue
            scaled_non_decayed = self._directory_updater._merge_manager._scale_usage_count(entry.non_decayed_usage_count)
            node_style = NodeStyle()
            node_style["fgcolor"] = f"#{int((1-scaled_non_decayed/100)*255):02x}00FF"  # Shades of purple (some blue so it doesn't fade entirely to white)
            node_style["size"] = 10

            tree_node = tree.add_child()
            tree_node.set_style(node_style)

            if entry.is_long_term:
                self._print_hierarchy(layer_id + 1, entry.short_term_versions, tree_node)

        return tree

    def _save_tree(self, tree, frame_timestep_id):  # TODO: these names are terrible
        style = TreeStyle()
        style.show_scale = False
        style.branch_vertical_margin = 3

        # In theory can pass %%return to get render to directly return the image, but QByteArray isn't being fun to work with
        # So writing to a file then reading it back in
        tmp_path = f"{self._config.output_dir}/tmp.jpg"
        tree.render(tmp_path, tree_style=style)
        pil_image = Image.open(tmp_path)
        ImageDraw.Draw(pil_image).text(
            (0, 0),  # Coordinates
            f"Timestep {frame_timestep_id}",  # Text
            (255, 170, 25)  # Color
        )

        # TODO: maybe something like this instead? If I can't get ImageDraw to be higher res...
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #position = (5, 5)
        #cv2.putText(image, int_rew_str, position, font, 0.4, text_color, 1, cv2.LINE_AA)

        np_image = np.array(pil_image)
        image_clip = mpy.ImageClip(np_image, duration=0.1)
        self._directory_updater._tree_video_frames.append(image_clip)

        # Overwriting the video each step because there doesn't seem to be a convenient way to just append a frame
        # to an existing video.
        # "Compose" will find the max size and create the video with that
        # For codec selection see https://zulko.github.io/moviepy/ref/VideoClip/VideoClip.html?highlight=codec
        concat_clip = mpy.concatenate_videoclips(self._directory_updater._tree_video_frames, method="compose")
        concat_clip.write_videofile(f"{self._config.output_dir}/tree.mp4", fps=30, codec="mpeg4", audio=False)

        torch_image = torch.Tensor(np_image)
        torch_image = torch_image.permute(2, 0, 1)
        return torch_image

    def train(self, storage_buffer):
        self._total_timesteps += len(storage_buffer) * len(storage_buffer[0])
        num_envs_per_proc = 1
        num_hypotheses_created = 0

        self._logger.info("=========")  # TODO: temporary location for this so it doesn't get logged so many times
        tree = self._print_hierarchy(0, self._directory_data._long_term_directory)  # TODO: rename...
        #new_tree_frame = self._save_tree(tree, self._total_timesteps)

        env_sorted_data_blobs = [[] for _ in range(num_envs_per_proc * len(storage_buffer))]
        for proc_id, proc_storage_buffer in enumerate(storage_buffer):
            env_offset_id = proc_id * num_envs_per_proc

            for info_to_store in proc_storage_buffer:
                env_sorted_singles = info_to_store.convert_to_array_of_singles()

                for env_id, data_blob_entry in enumerate(env_sorted_singles):
                    offset_env_id = env_id + env_offset_id
                    # Kill two birds with one stone - reorganize and update_core_process_data
                    env_sorted_data_blobs[offset_env_id].append(data_blob_entry)

                    hypothesis_id, _, _, _, _ = data_blob_entry.data_blob

                    try:
                        self._directory_updater.set_update_core_process_data(({}, [hypothesis_id]))
                    except AssertionError:
                        self._logger.warn(f"Either hypothesis {hypothesis_id} or its parent is missing.... I do not know why")
                        pass  # TODO: I don't understand why the hypothesis is getting deleted between usage and getting here

            # To save space, we only keep one creation buffer and keep updating it
            num_hypotheses_created = len(proc_storage_buffer[-1].creation_buffer)  # TODO: this isn't super clean
            self._directory_updater.set_update_core_process_data((proc_storage_buffer[-1].creation_buffer, []))
            proc_storage_buffer[-1].creation_buffer.clear()  # So we don't explode

        policy_loss, total_num_hypotheses = self._directory_updater.update(env_sorted_data_blobs)

        # TODO: absolutely 100% ensure that the order is always consistent (ie with uuids)
        self._update_processes_state_info = [(self._directory_data, proc_storage[-1])
                                             for proc_storage in storage_buffer]

        logs = [{"type": "scalar", "tag": f"total_num_hypotheses", "value": total_num_hypotheses},
                {"type": "scalar", "tag": f"num_hypotheses_created", "value": num_hypotheses_created}]
                #{"type": "image", "tag": f"tree", "value": new_tree_frame}]
        if policy_loss is not None:  # If every hypothesis that was used had been deleted before update, for instance.
            logs.append({"type": "scalar", "tag": f"policy_loss", "value": policy_loss})
        return logs

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
