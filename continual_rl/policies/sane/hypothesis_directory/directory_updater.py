import torch
import torch.optim as optim
from continual_rl.policies.sane.hypothesis.core_accessor import CoreAccessor
from continual_rl.policies.sane.hypothesis.usage_accessor import UsageAccessor
import numpy as np
import gc
from continual_rl.policies.sane.hypothesis_directory.hypothesis_lifetime_manager import HypothesisLifetimeManager
from continual_rl.policies.sane.hypothesis_directory.hypothesis_merge_manager import HypothesisMergeManager
from continual_rl.policies.sane.hypothesis.replay_buffer import ReplayEntry
from continual_rl.policies.sane.hypothesis_directory.utils import Utils
#from memory_profiler import profile
import psutil


class DirectoryUpdater(object):
    """
    Manages all aspects of updating the hypotheses: creation, training policy, training critics, merging, deletion.
    """
    def __init__(self, directory_data):
        self.hypothesis_accessor = CoreAccessor
        self._hypothesis_usage_accessor = UsageAccessor
        self._lifetime_manager = HypothesisLifetimeManager(directory_data)
        self._merge_manager = HypothesisMergeManager(directory_data, self._lifetime_manager)
        self._data = directory_data
        self._train_step = 0

        # We discover we need to create hypotheses during get(), but we want get() to be parallelizable, so defer the creation of hypotheses to the update.
        # This structure contains (directory_to_create_in, hypothesis_to_duplicate)
        self._hypothesis_creation_buffer = {}  # "None" means create new in long_term, otherwise create new as a duplicate under the STVs

        # Initialize first long-term hypothesis
        if len(self._data._long_term_directory) == 0:
            self._lifetime_manager._create_hypothesis(parent=None)

    def shutdown(self):
        self._lifetime_manager.shutdown()

    @property
    def logger(self):
        logger = Utils.create_logger(f"{self._data._output_dir}/core_process.log")
        return logger

    @classmethod
    def update_creation_buffer(cls, creation_buffer_target, creation_buffer_source):  # TODO: utility instead
        """
        Transfer elements from source into target
        """
        for hypo_id, all_creation_data in creation_buffer_source.items():
            if hypo_id not in creation_buffer_target:
                creation_buffer_target[hypo_id] = {}

            for priority, creation_data in all_creation_data.items():
                prior_count = creation_buffer_target[hypo_id].get(priority, {}).get('count', 0)

                creation_buffer_target[hypo_id][priority] = creation_data  # TODO: just takes the last, except accumulate the count
                creation_buffer_target[hypo_id][priority]['count'] += prior_count
                #self.logger.info(f"Creation buffer contains hypo {hypo_id} with priority {priority} and count {creation_buffer_target[hypo_id][priority]['count']}")

    def set_update_core_process_data(self, data_bundle):
        creation_buffer, hypothesis_update_bundles = data_bundle
        self.update_creation_buffer(self._hypothesis_creation_buffer, creation_buffer)

        for hypothesis_id in hypothesis_update_bundles:
            hypothesis = self.get_hypothesis_from_id(hypothesis_id)

            # Prototypes don't actually get updated, so their usages shouldn't either (TODO: here? or elsewhere, do the check)
            if hypothesis is None:
                self.logger.warning(f"set_update_core_process_data: Attempted to find hypothesis {hypothesis_id} that no longer exists")
            elif hypothesis.is_prototype:
                self.logger.warning(f"set_update_core_process_data: Skipping prototype {hypothesis_id}")
            else:
                self.hypothesis_accessor.increment(hypothesis, count=1)

                # Increment the whole way up the tree
                while hypothesis.parent_hypothesis is not None:
                    self.hypothesis_accessor.increment(hypothesis.parent_hypothesis, count=1)
                    hypothesis = hypothesis.parent_hypothesis

    def get_hypothesis_from_id(self, id):
        selected_hypothesis = self._data.get_hypothesis_from_id(id)

        if selected_hypothesis is None:
            self.logger.warning(f"get_hypothesis_from_id: Attempted to find hypothesis {id} that no longer exists")

        # May return None if the hypothesis was not found.
        return selected_hypothesis

    #@profile
    def update(self, all_storage_buffers):
        """
        Primary update function. Should be called periodically to ensure everything is getting updated 
        LT -> new meta, ST -> LT, ST merging, policy update, pattern filter update.
        """
        policy_loss = self._train_policy(all_storage_buffers)

        # TODO: should anchor policies update to be weighted averages of their children?
        # or would this only help the very first?

        num_samples = self._data._config.filter_train_batch_size
        long_term_forced_update_freq = 100
        create_update_freq = 1

        # Check that our hypothesis processes are still alive
        for entry in self._data.all_hypotheses:
            self._lifetime_manager.get_comms(entry).check_outgoing_queue()
            self._lifetime_manager.get_comms(entry).wait_for_train_to_complete()

        self.logger.info("Sync with train complete")

        # Everything after this may be asynchronous, and we don't want conflict between usage_count_since_last_update and the running of episodes, so clone it up front
        cloned_usage_count_since_last_update = self._clone_usage_count_since_last_update()
        #self._reset_usage_count_since_last_update()

        # Clear creation buffer before things that might take a while, since add_hypothesis is currently await'd
        # Happens after replay buffer update so we get the most recent buffer -- TODO: not doing this right now to see if I can make the duplication segfault go away...also to keep stuff async
        # Should happen after sync in update and before anything that might initiate a deletion, to remove the chance we're duplicating a hypothesis that has been deleted
        if self._train_step % create_update_freq == 0:
            self.logger.info(f"Clearing the creation buffer, with {len(self._hypothesis_creation_buffer)} entries")
            self._lifetime_manager._process_creation_buffer(self._hypothesis_creation_buffer)
        self._hypothesis_creation_buffer.clear()  # TODO: a test - if we only ever actually create from the latest set, not the aggregate...faster but just as good?

        # Make sure everyone has at least two, otherwise refactory means we basically just waste an episode telling us to go create two already
        for hypothesis in self._data.all_hypotheses:
            while hypothesis.is_long_term and len(hypothesis.short_term_versions) < 2:
                policy = hypothesis.prototype._policy
                self.logger.info(f"Long Term {hypothesis.friendly_name} needs a new STV, creating one. _Policy: {policy}")
                self._lifetime_manager._duplicate_hypothesis(hypothesis, parent=hypothesis, random_policy=False, pattern_filter_source=hypothesis)

        if self._train_step % long_term_forced_update_freq == 0:  # Slow, so only do sometimes
            self.logger.info("Sending long-term refresh cache")

            for entry in self._data.all_hypotheses:
                if entry.is_long_term:
                    cache = []
                    for short_term_entry in entry.short_term_versions:
                        num_to_get = short_term_entry._replay_buffer.maxlen//len(entry.short_term_versions)
                        #cache.extend(self._lifetime_manager.get_comms(short_term_entry).get_random_replay_buffer_entries(num_neg_to_get, id_start_frac=0, id_end_frac=1))
                        cache.append(self._lifetime_manager.get_comms(short_term_entry).get_random_replay_buffer_entries(num_to_get))

                    #self._lifetime_manager.get_comms(entry)._to_add_to_replay_cache.add_many(cache)
                    self._lifetime_manager.get_comms(entry).add_many_to_replay(cache)

        self.logger.info("Sending replay cache")
        # Update the replay buffers of each hypothesis
        for entry in self._data.all_hypotheses:
            # TODO: this is turned off because we're compacting the replay cache
            # assert len(self._hypothesis_comms[long_term_entry]._to_add_to_replay_cache) == long_term_entry.usage_count_since_last_update, f"Number of items in long-term replay cache ({len(self._hypothesis_comms[long_term_entry]._to_add_to_replay_cache)}) doesn't match number of usages counted ({long_term_entry.usage_count_since_last_update})"
            self._lifetime_manager.get_comms(entry).send_replay_cache()

        self.logger.info("Updating long term policies")
        self._update_long_term_policies()

        self.logger.info("Merging hypotheses")
        self._merge_manager._ensure_max_hypotheses_for_directory(layer_id=0, directory=self._data._long_term_directory, num_samples=num_samples)

        # Number of iterations in the outer loop so we can kick off jobs in parallel... TODO: though I think the get_negative_examples will still cause issues
        # Should happen last, because it's done asynchronously (and takes a long time), so may finish during usage
        self.logger.info("Training hypotheses")
        for entry in self._data.all_hypotheses:
            num_times_to_train = self._data._num_times_to_train_long_term if entry.is_long_term else self._data._num_times_to_train_short_term

            if cloned_usage_count_since_last_update.get(entry.unique_id, 0) > self._data._num_before_train or (self._data._always_train_all_long_term and entry.is_long_term): # and entry.usage_count > 500: # or (entry.is_long_term and self._train_step % long_term_forced_update_freq == 0):
                self._lifetime_manager.get_comms(entry).train(num_samples, id_start_frac=0, id_end_frac=1,  # TODO: 1st three currently ignored
                                                              num_times_to_train=num_times_to_train)
                entry.usage_count_since_last_update = 0

        print("Clearing caches")
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info("Directory update done")

        self._train_step += 1
        total_num_hypotheses = len(list(self._data.all_hypotheses))

        # Log out memory, to help me make sure memory is the cause of my early process death
        virtual_mem = psutil.virtual_memory()
        self.logger.info(f"Virtual mem: {virtual_mem}")

        return policy_loss, total_num_hypotheses

    def _update_long_term_policies(self):
        for hypothesis in self._data.all_hypotheses:
            if hypothesis.is_long_term:
                if len(hypothesis.short_term_versions) > 0:
                    hypothesis._policy.data = self._merge_manager._create_combined_policy(hypothesis.short_term_versions)

    def _reset_usage_count_since_last_update(self):
        for hypothesis in self._data.all_hypotheses:
            hypothesis.usage_count_since_last_update = 0

    def _clone_usage_count_since_last_update(self):
        usage_count_since_last_update_clone = {}

        for entry in self._data.all_hypotheses:
            usage_count_since_last_update_clone[entry.unique_id] = entry.usage_count_since_last_update

        return usage_count_since_last_update_clone

    def _compute_loss_and_update_replay(self, storage_buffer):
        last_reward = None
        gamma = self._data._config.reward_decay_rate
        actor_coeff = 10000 # 10000

        actor_losses = []
        all_policy_params = []
        prediction_losses = []
        hypothesis_id_cache = {}

        self.logger.info(f"Starting compute_loss with storage buffer length {len(storage_buffer)}")

        # Compute our losses and collect the parameters to train
        for entry_id, storage_entry in enumerate(reversed(storage_buffer)):
            # Unpack what we've stored
            policy_info = storage_entry.data_blob
            reward = storage_entry.reward
            hypothesis_id, selected_action, action_size, value, replay_entry_input = policy_info

            if self._data._config.scale_reward_by_max:
                # A naive attempt at auto scaling the reward
                self._data._max_reward_received *= .999  # If we go a while without seeing our top values, lower our standards again
                if np.abs(reward) > self._data._max_reward_received:
                    self._data._max_reward_received = np.abs(reward)
                reward = (reward / self._data._max_reward_received) * 10
            else:
                reward = np.clip(reward, -1, 1) * 10

            cached_hypothesis = hypothesis_id_cache.get(hypothesis_id, None)
            hypothesis = cached_hypothesis or self.get_hypothesis_from_id(hypothesis_id)

            if hypothesis is None:
                self.logger.warning(f"compute_loss_and_update_replay: Attempted to find hypothesis {hypothesis_id} that no longer exists")
                continue  # TODO: not 100% sure why this is happening, figure it out please. Okay, well now at least it's because policy is trained after filters (on purpose)

            if cached_hypothesis is None:
                hypothesis_id_cache[hypothesis_id] = hypothesis

            if storage_entry.done:
                last_reward = 0

            elif last_reward is None:  # TODO: no_grad instead?
                # TODO: accessing the usage_accessor like this instead of via directory_accessor is ...cheating?
                _, last_reward, _ = self._hypothesis_usage_accessor.forward(hypothesis,
                                                                            torch.Tensor(replay_entry_input),
                                                                            eval_mode=True,
                                                                            counter_lock=None,
                                                                            create_replay=False)
                last_reward = .9 * last_reward.detach().cpu()  # TODO: a high gamma (.999) is good for credit assignment, but bad if the value is wrong because then it takes forever to converge back to 0, from this estimation

            # TODO: I don't know why my env torch processes aren't allowing tensors, but...they aren't, so doing this dumb thing for the moment
            selected_action = torch.Tensor(selected_action)[0]
            value = torch.Tensor(value)[0]
            replay_entry = ReplayEntry(torch.Tensor(replay_entry_input))

            adjusted_reward = reward + gamma * last_reward  # For the "critic" (pattern filter)

            cumulative_advantage = adjusted_reward - value

            # We can't store anything with a grad or queue in policy_info, so re-inflate the relevant components
            policy = hypothesis.get_policy_with_entropy(torch.Tensor(replay_entry_input)).to(hypothesis._master_device)
            log_probs, _, entropy = Utils.get_log_probs(hypothesis, policy,
                                                                 random_action_rate=0, action_size=action_size,
                                                                 selected_action=selected_action)

            actor_losses.append(-(actor_coeff * cumulative_advantage.detach() * log_probs)) # - entropy_coeff * entropy)

            policy_params, filter_params = hypothesis.parameters()
            all_policy_params.extend(policy_params)

            last_reward = adjusted_reward

            # Need to add the replay entry even if we don't use the entry for the actor loss - TODO: not anymore, since I'm using log_probs
            parent_hypothesis_comms = self._lifetime_manager.get_comms(hypothesis.parent_hypothesis) if hypothesis.parent_hypothesis is not None else None
            self._lifetime_manager.get_comms(hypothesis).complete_and_register_replay_entry_to_send(replay_entry,
                                                                                                     adjusted_reward,
                                                                                                     log_probs,
                                                                                                     selected_action,
                                                                                                     parent_hypothesis_comms)

        return actor_losses, prediction_losses, all_policy_params

    def _train_policy(self, all_storage_buffers):
        # all_storage_buffers are expected to be [env, timestep]
        self.logger.info("Training policies triggered")

        self.logger.info(f"Computing loss and sending replay entries: {len(all_storage_buffers)}")
        total_loss = None
        losses = []
        all_prediction_losses = []
        all_policy_params = []

        # Do the storage buffers separately so the rewards get propagated properly... it does mean they overwrite each other?
        # These have to be done before the train_pattern_filters, because it also populates them... TODO break it up
        for storage_buffer in all_storage_buffers:
            # TODO: move into policy, too policy-specific to be out here
            actor_losses, prediction_losses, params = self._compute_loss_and_update_replay(storage_buffer)

            losses.extend(actor_losses)
            all_prediction_losses.extend(prediction_losses)
            all_policy_params.extend(params)

        if len(losses) > 0:
            total_loss = torch.stack(losses).mean()
            optimizer = optim.SGD([{'params': list(set(all_policy_params)), "lr": self._data._config.consequent_learning_rate}], lr=1e-3)
            optimizer.zero_grad()

            try:
                total_loss.backward()
                optimizer.step()
            except RuntimeError as e:
                # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
                # This can happen if only prototypes are used
                assert "does not require grad" in repr(e)
                self.logger.info(f"Skipping policy update due to no valid gradients")

            self.logger.info(f"Actor loss: {total_loss}")

        self.logger.info("Train complete")
        return total_loss
