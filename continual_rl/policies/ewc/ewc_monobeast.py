import numpy as np
import torch
import threading
import json
import shutil
import os
from continual_rl.policies.impala.torchbeast.monobeast import Monobeast, Buffers
from continual_rl.utils.utils import Utils


class EWCTaskInfo(object):
    def __init__(self, model_flags, buffer_specs, entries_per_buffer, task_name):
        # Variables used on both the main process and shared processes
        # Technically only the replay_buffers probably need to be file-backed, but may as well handle everything the
        # same, for consistency.

        # We want the replay buffers to be created in the large_file_path,
        # but in a place characteristic to this experiment.
        # Be careful if the output_dir specified is very nested
        # (ie. Windows has max path length of 260 characters)
        # Could hash output_dir_str if this is a problem.
        output_dir_str = os.path.normpath(model_flags.output_dir).replace(os.path.sep, '-')
        permanent_path = os.path.join(
            model_flags.large_file_path,
            "file_backed",
            output_dir_str,
            task_name,
        )
        buffers_existed = os.path.exists(permanent_path)
        os.makedirs(permanent_path, exist_ok=True)

        self.replay_buffers, self.temp_files = self._create_replay_buffers(
            model_flags, buffer_specs, entries_per_buffer, permanent_path
        )
        self.total_steps, _, total_step_file = Utils.create_file_backed_tensor(
            permanent_path, (1,), dtype=torch.int64, permanent_file_name="total_steps.fbt"
        )
        self.replay_buffer_counters, _, replay_counter_file = Utils.create_file_backed_tensor(
            permanent_path,
            (model_flags.num_actors,),
            dtype=torch.int64,
            permanent_file_name="replay_counters.fbt",
        )

        self.temp_files.append(total_step_file)
        self.temp_files.append(replay_counter_file)

        if not buffers_existed:
            # Set to 0, since they're both counters.
            self.total_steps.zero_()
            self.replay_buffer_counters.zero_()

        # Main-process only variables
        self.ewc_regularization_terms = None

    def _create_replay_buffers(self, model_flags, specs, entries_per_buffer, permanent_path):
        """
        Key differences from normal buffers:
        1. File-backed, so we can store more at a time
        2. Structured so that there are num_actors buffers, each with entries_per_buffer entries

        Each buffer entry has unroll_length size, so the number of frames stored is (roughly, because of integer
        rounding): num_actors * entries_per_buffer * unroll_length
        """
        buffers: Buffers = {key: [] for key in specs}

        # Hold on to the file handle so it does not get deleted. Technically optional, as at least linux will
        # keep the file open even after deletion, but this way it is still visible in the location it was created
        temp_files = []

        for actor_id in range(model_flags.num_actors):
            for key in buffers:
                shape = (entries_per_buffer, *specs[key]["size"])
                permanent_file_name = f"replay_{actor_id}_{key}.fbt"
                new_tensor, _, temp_file = Utils.create_file_backed_tensor(
                    permanent_path,
                    shape,
                    specs[key]["dtype"],
                    permanent_file_name=permanent_file_name,
                )
                buffers[key].append(new_tensor.share_memory_())
                temp_files.append(temp_file)

        return buffers, temp_files


class EWCMonobeast(Monobeast):
    """
    An implementation of Elastic Weight Consolidation: https://arxiv.org/pdf/1612.00796.pdf.
    The original used DQN, but this is a variant that uses IMPALA, which we believe to be in-line with what is
    described in Progress and Compress: https://arxiv.org/pdf/1805.06370.pdf. Additionally, online EWC is as described
    in P&C. The full P&C method is available in progress_and_compress.
    """

    def __init__(self, model_flags, observation_space, action_spaces, policy_class):
        super().__init__(model_flags, observation_space, action_spaces, policy_class)

        # LSTMs not supported largely because they have not been validated; nothing extra is stored for them.
        assert not model_flags.use_lstm, "EWC does not presently support using LSTMs."

        self._model_flags = model_flags
        self._observation_space = observation_space
        self._action_space = Utils.get_max_discrete_action_space(action_spaces)

        self._entries_per_buffer = int(
            model_flags.replay_buffer_frames // (model_flags.unroll_length * model_flags.num_actors)
        )
        self._prev_task_id = None
        self._checkpoint_lock = threading.Lock()
        self._collection_paused = False

        self._tasks = None  # If you observe this never getting set, make sure initialize_tasks is getting called

    def save(self, output_path):
        super().save(output_path)
        ewc_metadata_path = os.path.join(output_path, "ewc_metadata.tar")

        # Back up previous model (sometimes they can get corrupted)
        if os.path.exists(ewc_metadata_path):
            shutil.copyfile(ewc_metadata_path, os.path.join(output_path, "ewc_metadata_bak.tar"))

        metadata = {"prev_task_id": self._prev_task_id}
        per_task_metadata = {}

        for task_id, task_info in self._tasks.items():
            per_task_metadata[task_id] = task_info.ewc_regularization_terms

        metadata["per_task_metadata"] = per_task_metadata
        torch.save(metadata, ewc_metadata_path)

    def load(self, output_path):
        super().load(output_path)
        ewc_metadata_path = os.path.join(output_path, "ewc_metadata.tar")
        metadata = None

        if os.path.exists(ewc_metadata_path):
            self.logger.info(f"Loading ewc metdata from {ewc_metadata_path}")
            metadata = torch.load(ewc_metadata_path)
            key_fn = lambda x: x

        if metadata is not None:
            self._prev_task_id = metadata["prev_task_id"]
            per_task_metadata = metadata["per_task_metadata"]

            for task_id in self._tasks.keys():
                self._tasks[task_id].ewc_regularization_terms = per_task_metadata[key_fn(task_id)]

    def set_pause_collection_state(self, state):
        self._collection_paused = state

    def initialize_tasks(self, task_ids):
        # Initialize the tensor containers for all storage for each task. By using tensors we can avoid
        # having to pass information around by queue, instead just updating the shared tensor directly.
        specs = self.create_buffer_specs(
            self._model_flags.unroll_length, self._observation_space.shape, self._action_space.n
        )

        if self._model_flags.online_ewc:
            self._tasks = {
                "online": EWCTaskInfo(self._model_flags, specs, self._entries_per_buffer, "online")
            }
        else:
            self._tasks = {
                id: EWCTaskInfo(self._model_flags, specs, self._entries_per_buffer, f"task_{id}")
                for id in task_ids
            }

    def _compute_ewc_loss(self, task_flags, model):
        ewc_loss = 0
        num_tasks_included = 0

        # For each task, incorporate its regularization terms. If online ewc, then there should only be one "task"
        for task_id, task_info in self._tasks.items():
            if task_info.ewc_regularization_terms is not None and (
                not self._model_flags.omit_ewc_for_current_task
                or task_info != self._get_task(task_flags.task_id)
            ):
                self.logger.info("EWC regularization terms found: computing loss")
                task_param, importance = task_info.ewc_regularization_terms
                task_reg_loss = 0
                for n, p in model.named_parameters():
                    mean = task_param[n]
                    fisher = importance[n]
                    task_reg_loss_delta = fisher.detach() * (p - mean.detach()) ** 2
                    if self._model_flags.use_ewc_mean:
                        task_reg_loss = task_reg_loss + task_reg_loss_delta.mean()
                    else:
                        task_reg_loss = task_reg_loss + task_reg_loss_delta.sum()

                ewc_loss = ewc_loss + task_reg_loss
                num_tasks_included += 1

        if self._model_flags.scale_ewc_by_num_tasks and num_tasks_included != 0:
            # Scale by the number of tasks whose losses we're including, so the scale is roughly consistent
            final_ewc_loss = ewc_loss / num_tasks_included
        else:
            final_ewc_loss = ewc_loss

        return final_ewc_loss / 2.0

    def custom_loss(self, task_flags, model, initial_agent_state, batch, vtrace_returns):
        """
        Use the learner_model to save off Fisher information/mean params (via "checkpointing"), and use those
        to compute the EWC loss. Both use the learner_model for consistency (specifically device consistency).
        """
        # If we've moved to a new task, save off what we need to for ewc loss computation
        # Don't let multiple learner threads trigger the checkpointing
        with self._checkpoint_lock:
            cur_task_id = task_flags.task_id
            if self._prev_task_id is not None and cur_task_id != self._prev_task_id:
                # Note: task_flags passed in here are only pseudo-used. Consider using prev task flags if this changes
                self.logger.info(f"EWC: checkpointing {self._prev_task_id}")

                # EWC checkpointing can take some time, so attempting to pause stats reporting
                # so not just reporting nans. Still works without this, but cuts down on the
                # nans logged so to not appear like actors are dead. 
                with self._stats_lock:
                    self.checkpoint_task(self._prev_task_id, task_flags, model, online=self._model_flags.online_ewc)
            self._prev_task_id = cur_task_id

        if self._model_flags.online_ewc or self._get_task(cur_task_id).total_steps >= self._model_flags.ewc_per_task_min_frames:
            ewc_loss = self._model_flags.ewc_lambda * self._compute_ewc_loss(task_flags, model)
            stats = {"ewc_loss": ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else ewc_loss}
        else:
            ewc_loss = 0.0
            stats = {"ewc_loss": 0.0}

        return ewc_loss, stats

    def checkpoint_task(self, task_id, task_flags, model, online=False):
        # save model weights for task (MAP estimate)
        task_params = {}
        for n, p in model.named_parameters():
            task_params[n] = p.detach().clone()

        importance = {}
        # initialize to zeros
        for n, p in model.named_parameters():
            if p.requires_grad:
                importance[n] = p.detach().clone().fill_(0)  # initialize to zeros

        # estimate Fisher information matrix
        for i in range(self._model_flags.n_fisher_samples):
            task_replay_batch = self._sample_from_task_replay_buffer(task_id, self._model_flags.batch_size)

            # NOTE: setting initial_agent_state to an empty list, not sure if this is correct?
            # Calling Monobeast's loss explicitly to make sure the loss is the right one (PnC overrides it)
            # This uses pg_loss and baseline_loss as the signals for importance of parameters (omitting entropy)
            _, stats, pg_loss, baseline_loss = super().compute_loss(self._model_flags, task_flags, model, task_replay_batch, [], with_custom_loss=False)
            loss = pg_loss + baseline_loss
            self.optimizer.zero_grad()
            loss.backward()

            for n, p in model.named_parameters():
                assert p.grad is not None, f"Parameter {n} did not have a gradient when computing the Fisher. Therefore it will not be saved correctly."
                importance[n] = importance[n] + p.grad.detach().clone() ** 2

        # Normalize by sample size used for estimation
        task_info = self._get_task(task_id)
        importance = {n: p / self._model_flags.n_fisher_samples for n, p in importance.items()}

        if online and task_info.ewc_regularization_terms is not None:
            _, old_importance = task_info.ewc_regularization_terms

            for name, old_importance_entry in old_importance.items():
                # see eq. 9 in Progress & Compress
                new_importance_entry = self._model_flags.online_gamma * old_importance_entry + importance[name]
                importance[name] = new_importance_entry

        if self._model_flags.normalize_fisher_method:
            for name in importance.keys():
                v = importance[name]

                # conv filter: (O, C, W, H)
                # linear weight: (O, I)
                # bias: (O)
                if self._model_flags.normalize_fisher_method == "full":
                    importance[name] /= torch.norm(v)
                elif self._model_flags.normalize_fisher_method == "row":
                    d = 1 if len(p.shape) != 1 else 0  # also consider bias, conv filters
                    importance[name] = torch.nn.functional.normalize(v, dim=d)
                elif self._model_flags.normalize_fisher_method == "col":
                    importance[name] = torch.nn.functional.normalize(v, dim=0)
                else:
                    raise ValueError(f"Unsupported fisher normalization method {self._model_flags.normalize_fisher_method}.")

        task_info.ewc_regularization_terms = (task_params, importance)

    def on_act_unroll_complete(self, task_flags, actor_index, agent_output, env_output, new_buffers):
        if not self._collection_paused:
            task_info = self._get_task(task_flags.task_id)

            # update the tasks's total_steps
            task_info.total_steps += self._model_flags.unroll_length

            # update the task replay buffer
            to_populate_replay_index = task_info.replay_buffer_counters[actor_index] % self._entries_per_buffer
            for key in new_buffers.keys():
                task_info.replay_buffers[key][actor_index][to_populate_replay_index][...] = new_buffers[key]

            # should only be getting 1 unroll for any key
            task_info.replay_buffer_counters[actor_index] += 1

    def _get_task(self, task_id):
        task_lookup_label = "online" if self._model_flags.online_ewc else task_id
        return self._tasks[task_lookup_label]

    def _sample_from_task_replay_buffer(self, task_id, batch_size):
        task_info = self._get_task(task_id)
        replay_entry_count = batch_size
        shuffled_subset = []  # Will contain a list of tuples of (actor_index, buffer_index)
        random_state = np.random.RandomState()

        # Select a random actor, and from that, a random buffer entry.
        for _ in range(replay_entry_count):
            actor_index = random_state.randint(0, self._model_flags.num_actors)

            # We may not have anything in this buffer yet, so check for that (randint complains)
            entries_in_buffer = min(task_info.replay_buffer_counters[actor_index], self._entries_per_buffer)
            if entries_in_buffer > 0:
                buffer_index = random_state.randint(0, entries_in_buffer)
                shuffled_subset.append((actor_index, buffer_index))

        replay_batch = {
            # Get the actor_index and entry_id from the raw id
            key: torch.stack([task_info.replay_buffers[key][actor_id][buffer_id]
                              for actor_id, buffer_id in shuffled_subset], dim=1) for key in task_info.replay_buffers
        }

        replay_batch = {
            k: t.to(device=self._model_flags.device, non_blocking=True)
            for k, t in replay_batch.items()
        }
        return replay_batch
