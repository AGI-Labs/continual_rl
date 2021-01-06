import torch
from continual_rl.policies.impala.torchbeast.monobeast import Monobeast, Buffers

class EWCMonobeast(Monobeast):

    def __init__(self, model_flags, observation_space, action_space, policy_class):
        super().__init__(model_flags, observation_space, action_space, policy_class)

        # LSTMs not supported largely because they have not been validated; nothing extra is stored for them.
        assert not model_flags.use_lstm, "CLEAR does not presently support using LSTMs."

        self._model_flags = model_flags

        self.prev_task = None
        self.cur_task = None

        self.replay_buffer_ = {}
        self.total_steps_ = {} # might be easier to pull this from scheduler?

        self.ewc_task_reg_terms = {}

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
        if self.total_steps_[self.cur_task] > self._model_flags.it_start_ewc_per_task:
            ewc_loss = 0.
            stats = {"ewc_loss": 0.}
        else:
            ewc_loss = self._compute_ewc_loss(model, self.ewc_task_reg_terms) # model should be self.learner_model
            stats = {"ewc_loss": ewc_loss.item()}
        return self._model_flags.ewc_lambda * ewc_loss, stats

    def checkpoint_task(self, task_id, online=False):
        # save model weights for task (MAP estimate)
        task_params = {}
        for n, p in self.learner_model.named_parameters():
            task_params[n] = p.detach().clone()

        importance = {}
        # initialize to zeros
        for n, p in self.learner_model.named_parameters():
            # if '_bias' in n or '_gain' in n: # these parameters are task specific
            #     continue

            if p.requires_grad:
                importance[n] = p.detach().clone().fill_(0) # initialize to zeros

        # estimate Fisher information matrix
        for i in range(self._model_flags.n_fisher_samples):
            task_replay_buffer = self._get_task_replay_buffer(task_id) # get based on task_id
            task_replay_batch = self._sample_from_task_replay_buffer(task_replay_buffer, self._model_flags.batch_size)

            # NOTE: setting initial_agent_state to an empty list, not sure if this is correct?
            loss, stats = self.compute_loss(self._model_flags, self.learner_model, task_replay_batch, [], with_custom_loss=False)
            self.optimizer.zero_grad()
            loss.backward()

            for n, p in self.learner_model.named_parameters():
                # if '_bias' in n or '_gain' in n: # these parameters are task specific
                #     continue

                if p.requires_grad and p.grad is not None:
                    importance[n] += p.grad.detach() ** 2

        # Normalize by sample size used for estimation
        importance = {n: p / self._model_flags.n_fisher_samples for n, p in importance.items()}

        if online:
            old_task_params, old_importance = self.ewc_task_reg_terms[-1]
            for n, p in old_importance.items():
                v = importance[n] + self._model_flags.online_gamma * old_importance # see eq. 9 in Progress & Compress

                if self._model_flags.normalize_fisher:
                    v /= torch.norm(v)

                importance[n] = v

            self.ewc_task_reg_terms[-1] = (task_params, importance)
        else:
            self.ewc_task_reg_terms[task_id] = (task_params, importance)

    def on_act_unroll_complete(self, actor_index, agent_output, env_output, new_buffers):
        # if we've switched tasks, then checkpoint the model
        if self.prev_task is not None and self.prev_task != self.cur_task:
            # switching task
            self.checkpoint_task(self.prev_task, online=self._model_flags.online_ewc)

        if self.cur_task not in self.total_steps_.keys():
            # this is our first time seeing this task
            if self._model_flags.online_ewc and self.prev_task is not None:
                # if online ewc, then only need to keep the current task buffers
                self.replay_buffer_[-1] = self._create_replay_buffer()
                # self.replay_buffer_[self.cur_task] = self.replay_buffer_[-1] # add ref
            else:
                self.replay_buffer_[self.cur_task] = self._create_replay_buffer()
            self.total_steps_[self.cur_task] = 0

        self.prev_task = self.cur_task

        # update the task replay buffer
        raise NotImplementedError

        # update the tasks's total_steps
        raise NotImplementedError

    def set_current_task(self, task_id):
        self.cur_task = task_id

    def _get_task_replay_buffer(self, task_id):
        raise NotImplementedError

    def _sample_from_task_replay_buffer(self, task_replay_buffer, batch_size):
        raise NotImplementedError

    def _create_replay_buffers(self):
        # based on self._model_flags.task_replay_buffer_max_size
        raise NotImplementedError