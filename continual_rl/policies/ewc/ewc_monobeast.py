import numpy as np
import torch
import threading
from continual_rl.policies.impala.torchbeast.monobeast import Monobeast, Buffers
from continual_rl.utils.utils import Utils


class EWCMonobeast(Monobeast):

    def __init__(self, model_flags, observation_space, action_space, policy_class):
        super().__init__(model_flags, observation_space, action_space, policy_class)

        # LSTMs not supported largely because they have not been validated; nothing extra is stored for them.
        assert not model_flags.use_lstm, "CLEAR does not presently support using LSTMs."

        self._model_flags = model_flags
        self._observation_space = observation_space
        self._action_space = action_space

        self._entries_per_buffer = int(model_flags.replay_buffer_frames // (model_flags.unroll_length * model_flags.num_actors))

        self._lock = threading.Lock()
        self._prev_task = None
        self._cur_task = None
        self._replay_buffer = {}
        self._replay_buffer_counter = {}
        self._total_steps = {} # might be easier to pull this from scheduler?
        self._temp_files = {}

        self._ewc_task_reg_terms = {}

    def _compute_ewc_loss(self, model, ewc_task_reg_terms):
        ewc_loss = 0
        for k, (task_param, importance) in ewc_task_reg_terms.items(): # if online ewc, then there should only be one "task"
            task_reg_loss = 0
            for n, p in model.named_parameters():
                # if '_bias' in n or '_gain' in n: # these parameters are task specific
                #     continue

                mean = task_param[n]
                fisher = importance[n]
                task_reg_loss += (fisher * (p - mean) ** 2).sum()

            ewc_loss += task_reg_loss
        return ewc_loss / 2.

    def custom_loss(self, model, initial_agent_state):
        if not self._model_flags.online_ewc and \
         (self._total_steps[self._cur_task] > self._model_flags.it_start_ewc_per_task):
            ewc_loss = 0.
            stats = {"ewc_loss": 0.}
        else:
            ewc_loss = self._compute_ewc_loss(model, self._ewc_task_reg_terms) # model should be self.learner_model
            stats = {"ewc_loss": ewc_loss.item()}
        return self._model_flags.ewc_lambda * ewc_loss, stats

    def checkpoint_task(self, task_id, model, online=False):
        # save model weights for task (MAP estimate)
        task_params = {}
        for n, p in model.named_parameters():
            task_params[n] = p.detach().clone()

        importance = {}
        # initialize to zeros
        for n, p in model.named_parameters():
            # if '_bias' in n or '_gain' in n: # these parameters are task specific
            #     continue

            if p.requires_grad:
                importance[n] = p.detach().clone().fill_(0) # initialize to zeros

        # estimate Fisher information matrix
        for i in range(self._model_flags.n_fisher_samples):
            task_replay_buffer_counter, task_replay_buffer = self._get_task_replay_buffer(task_id) # get based on task_id
            task_replay_batch = self._sample_from_task_replay_buffer(task_replay_buffer_counter, task_replay_buffer, self._model_flags.batch_size)

            # NOTE: setting initial_agent_state to an empty list, not sure if this is correct?
            loss, stats = self.compute_loss(self._model_flags, model, task_replay_batch, [], with_custom_loss=False)
            self.optimizer.zero_grad()
            loss.backward()

            for n, p in model.named_parameters():
                # if '_bias' in n or '_gain' in n: # these parameters are task specific
                #     continue

                if p.requires_grad and p.grad is not None:
                    importance[n] += p.grad.detach() ** 2

        # Normalize by sample size used for estimation
        importance = {n: p / self._model_flags.n_fisher_samples for n, p in importance.items()}

        if online:
            old_task_params, old_importance = self._ewc_task_reg_terms[-1]
            for n, p in old_importance.items():
                v = self._model_flags.online_gamma * old_importance + importance[n]# see eq. 9 in Progress & Compress

                if self._model_flags.normalize_fisher:
                    v /= torch.norm(v)

                importance[n] = v

            self._ewc_task_reg_terms[-1] = (task_params, importance)
        else:
            self._ewc_task_reg_terms[task_id] = (task_params, importance)

    def on_act_unroll_complete(self, actor_index, agent_output, env_output, new_buffers):
        with self._lock:
            # if we've switched tasks, then checkpoint the model
            if self._prev_task is not None and self._prev_task != self._cur_task:
                # switching task
                # not using self.learner_model b/c may be getting updated in learn()
                self.checkpoint_task(self._prev_task, self.model, online=self._model_flags.online_ewc)

            if self._model_flags.online_ewc:
                buffer_key = 'online'
            else:
                buffer_key = self._cur_task

            if buffer_key not in self._total_steps.keys():
                # this is our first time seeing this task
                task_buffer_counters, task_replay_buffers, task_temp_files = self._create_replay_buffers(self._model_flags, self._observation_space.shape,
                                                                                self._action_space.n, self._entries_per_buffer)
                self._replay_buffer[buffer_key] = task_replay_buffers
                self._replay_buffer_counter[buffer_key] = task_buffer_counters
                self._temp_files[buffer_key] = task_temp_files
                self._total_steps[buffer_key] = 0

            self._prev_task = self._cur_task

            # update the tasks's total_steps
            self._total_steps[buffer_key] += self._model_flags.unroll_length

            to_populate_replay_index = self._replay_buffer_counter[buffer_key][actor_index] % self._entries_per_buffer
            # update the task replay buffer
            for key in new_buffers.keys():
                self._replay_buffer[buffer_key][key][actor_index][to_populate_replay_index][...] = new_buffers[key]

            # should only be getting 1 unroll for any key
            self._replay_buffer_counter[buffer_key][actor_index] += 1

    def set_current_task(self, task_id):
        self._cur_task = task_id

    def _get_task_replay_buffer(self, task_id):
        if self._model_flags.online_ewc:
            buffer_key = 'online'
        else:
            buffer_key = task_id
        return self._replay_buffer_counter[buffer_key], self._replay_buffer[buffer_key]

    def _sample_from_task_replay_buffer(self, task_replay_buffer_counter, task_replay_buffer, batch_size):
        replay_entry_count = batch_size
        shuffled_subset = np.random.randint(0, min(task_replay_buffer_counter, self._entries_per_buffer), size=replay_entry_count)

        replay_batch = {
            # Get the actor_index and entry_id from the raw id
            key: torch.stack([task_replay_buffer[key][m // self._entries_per_buffer][m % self._entries_per_buffer]
                                for m in shuffled_subset], dim=1) for key in task_replay_buffer
        }

        replay_batch = {k: t.to(device=self._model_flags.device, non_blocking=True) for k, t in replay_batch.items()}
        return replay_batch

    def _create_replay_buffers(self, model_flags, obs_shape, num_actions, entries_per_buffer):
        """
        Key differences from normal buffers:
        1. File-backed, so we can store more at a time
        2. Structured so that there are num_actors buffers, each with entries_per_buffer entries

        Each buffer entry has unroll_length size, so the number of frames stored is (roughly, because of integer
        rounding): num_actors * entries_per_buffer * unroll_length
        """
        # Get the standard specs, and also add the CLEAR-specific reservoir value
        specs = self.create_buffer_specs(model_flags.unroll_length, obs_shape, num_actions)
        buffers: Buffers = {key: [] for key in specs}

        # Hold on to the file handle so it does not get deleted. Technically optional, as at least linux will
        # keep the file open even after deletion, but this way it is still visible in the location it was created
        temp_files = []

        for _ in range(model_flags.num_actors):
            for key in buffers:
                shape = (entries_per_buffer, *specs[key]["size"])
                new_tensor, temp_file = Utils.create_file_backed_tensor(model_flags.large_file_path, shape,
                                                                        specs[key]["dtype"])
                # new_tensor.zero_()  # Ensure our new tensor is zero'd out
                buffers[key].append(new_tensor.share_memory_())
                temp_files.append(temp_file)

        # points to the next index to update
        buffer_counters = [0] * model_flags.num_actors

        return buffer_counters, buffers, temp_files
