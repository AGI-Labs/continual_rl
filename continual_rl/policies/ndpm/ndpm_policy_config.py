from continual_rl.policies.config_base import ConfigBase


class NdpmPolicyConfig(ConfigBase):
    # Based on https://github.com/soochan-lee/CN-DPM/blob/master/configs/mnist-cndpm.yaml
    # TODO = not sure at the moment if all are used in the current flow

    def __init__(self):
        super().__init__()
        self.render_collection_freq = 20000
        self.comment = ""  # No-op, just for commenting in the JSON
        
        self.data_root = "/tmp/data"
        self.batch_size = 10
        self.num_workers = 16
        self.sleep_batch_size = 50
        self.sleep_num_workers = 4
        self.eval_batch_size = 256
        self.eval_num_workers = 4

        #########
        # Model #
        #########
        self.x_c = 1
        self.x_h = 28
        self.x_w = 28
        self.y_c = 10
    
        self.device = 'cpu'
    
        self.model_name = 'ndpm_model'
        self.g = 'mlp_sharing_vae'
        self.d = 'mlp_sharing_classifier'
        self.disable_d = False
        self.vae_nf_base = 64
        self.vae_nf_ext = 16
        self.cls_nf_base = 64
        self.cls_nf_ext = 16
        self.z_dim = 16
        self.z_samples = 16
    
        self.recon_loss = 'gaussian'
        self.x_log_var_param = 0
        self.learn_x_log_var = False
        self.classifier_chill = 0.01

        #########
        # DPMoE #
        #########

        self.log_alpha = -400
        self.stm_capacity = 500
        self.stm_erase_period = 0
        self.sleep_step_g = 8000
        self.sleep_step_d = 2000
        self.sleep_summary_step = 500
        self.sleep_val_size = 0
        self.update_min_usage = 0.1

        #########
        # Train #
        #########

        self.implicit_lr_decay = False
        self.weight_decay = 0.00001

        # TODO...these are not command-line compatible
        self.optimizer_g = dict(type="Adam", options=dict(lr=0.0004))
        self.lr_scheduler_g = dict(type="MultiStepLR", options=dict(milestones=[1], gamma=1.0))

        self.optimizer_d = dict(type="Adam", options=dict(lr=0.0001))
        self.lr_scheduler_d = dict(type="MultiStepLR", options=dict(milestones=[1], gamma=1.0))

        self.clip_grad = dict(type="value", options=dict(clip_value=0.5))

        ########
        # Eval #
        ########

        self.eval_d = True
        self.eval_g = False
        self.eval_t = False

        ###########
        # Summary #
        ###########

        self.summary_step = 250
        self.eval_step = 250
        self.summarize_samples = False

    def _load_from_dict_internal(self, config_dict):
        self._auto_load_class_parameters(config_dict)
        return self
