from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
# import pytorch3d.ops as torch3d_ops  # unused, pytorch3d not installed

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder

class SimpleDP3(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            state_split=None,
            contact_gate=None,
            state_obs_dim=None,
            mask_dropout=0.0,
            mask_modality_dropout=0.0,
            aux_torque=None,
            clamp_agent_pos=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # ---- clamp normalized agent_pos into [-1,1] ------------------------------------
        # Pairs with InspireDrillDataset(torque_start=...): percentile torque scaling fixes the
        # channel's scale (std 0.106 -> 0.44) at the price of putting ~2% of frames -- the
        # effort-limit saturation frames -- outside [-1,1], up to +-8.9. Those 2% would carry
        # ~11% of the torque gradient (MSE is squared) and DDIM's clip_sample=True means the
        # model can never emit them at inference anyway. No-op under min/max normalization.
        # Applied here rather than in the dataset so predict_action gets the same treatment:
        # the normalizer travels in the checkpoint, so deploy must not need its own copy.
        self.clamp_agent_pos = bool(clamp_agent_pos)
        self._clamp_check_left = 0 if self.clamp_agent_pos else 50

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        # ---- auxiliary torque objective (TA-VLA, arXiv:2509.07962 Sec. 5) --------------------
        # The diffusion trajectory becomes Z_t = [A_t ; T_t]: the action chunk concatenated with
        # the torque chunk over the SAME horizon, denoised by the SAME UNet through the SAME
        # widened in/out conv -- deliberately not a second network or a second projection head.
        # The whole point of the auxiliary task is to shape the shared trunk, and a separate
        # network would keep the torque gradient in its own parameters where the action branch
        # never sees it. Losses stay separate: L = L_action + beta * L_torque.
        #
        # The target needs no new data: the dataset already returns the full `horizon` of obs
        # (only the first n_obs_steps are consumed for conditioning), so the future torque chunk
        # is nobs['agent_pos'][..., lo:hi] -- already normalised by the same LinearNormalizer that
        # puts the action in [-1,1], which is what makes beta a clean knob.
        #
        # aux_torque=None (default) restores the original 13-channel behaviour exactly.
        self.aux_dim = 0
        self.aux_slice = None
        self.aux_beta = 0.0
        if aux_torque is not None and bool(aux_torque.get('enabled', True)):
            lo = int(aux_torque.get('start', 13))
            hi = int(aux_torque.get('end', 26))
            if hi > lo:
                self.aux_dim = hi - lo
                self.aux_slice = (lo, hi)
                self.aux_beta = float(aux_torque.get('beta', 0.1))
                cprint(f"[SDP3] AUX torque objective: agent_pos[{lo}:{hi}] "
                       f"({self.aux_dim}d) over all {horizon} steps, beta={self.aux_beta}; "
                       f"UNet channels {action_dim} -> {action_dim + self.aux_dim}", "green")

        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])


        obs_encoder = DP3Encoder(observation_space=obs_dict,
                                                   img_crop_shape=crop_shape,
                                                out_channel=encoder_output_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                state_split=state_split,
                                                contact_gate=contact_gate,
                                                state_obs_dim=state_obs_dim,
                                                mask_dropout=mask_dropout,
                                                mask_modality_dropout=mask_modality_dropout,
                                                )

        # ---- contact-gate supervision (privileged: training only) --------------------------
        # The gate is the one module that has to make a decision INSIDE a frame -- its output
        # multiplies that frame's torque, so a wrong gate cannot be repaired downstream. Left to
        # the diffusion loss alone it collapses towards a constant, which is exactly the failure
        # arXiv:2604.01414 reports for their learned MoE router (weights 0.7634/0.2366 with
        # contact vs 0.7714/0.2286 without -- indistinguishable). The contact sensors give a
        # direct label for it, so we supervise it explicitly.
        # contact[k] and state[k] are the SAME frame by construction (collect_dp3_data.py reads
        # both after env.step() and writes them to one row), so unlike the auxiliary torque
        # target this needs NO shift.
        self.contact_groups = None
        self.contact_beta = 0.0
        self.contact_soft = False
        if contact_gate is not None and bool(contact_gate.get('enabled', False)):
            # index -> the contact-sensor columns of the links that joint drives.
            # Column order is fixed by GraspDrillEnv._get_contact_forces_obs's sensor list --
            # it is NOT the order the ContactSensorCfgs are declared in, so it must be mirrored
            # here literally rather than re-derived. 6 (thumb_proximal_base) never fires and 12
            # (hand_base) has no controllable joint to gate; both are unused.
            _scope = str(contact_gate.get('scope', 'per_finger'))
            # gate_position='feature' gates the force_mlp OUTPUT (arXiv:2604.01414 Eq.1). A 64-d
            # feature cannot be split across 6 finger gates, so that mode is a SINGLE gate whatever
            # the scope, and the supervision has to be the matching single label -- "is the hand
            # touching at all" -- rather than the six per-finger ones. Getting this wrong would not
            # crash: the shapes are checked below, but only after the labels were built, so it is
            # decided here in one place.
            _gpos = str(contact_gate.get('gate_position', 'input'))
            if _scope in ('global', 'finger_single') or _gpos == 'feature':
                # one gate, one label: "is the hand touching the drill at all". The union of every
                # live sensor -- 6 (thumb_proximal_base) never fires and is left out so it cannot
                # dilute the union.
                self.contact_groups = list(contact_gate.get(
                    'global_group', [[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]))
            else:
                self.contact_groups = list(contact_gate.get('groups', [
                    [8, 0],     # index  proximal + intermediate
                    [9, 2],     # middle
                    [11, 4],    # pinky
                    [10, 3],    # ring
                    [7, 5, 1],  # thumb yaw   -- proximal + intermediate + distal
                    [7, 5, 1],  # thumb pitch -- same links, shares the label
                ]))
            self.contact_beta = float(contact_gate.get('beta', 0.1))
            self.contact_thresh = float(contact_gate.get('threshold', 0.01))

            # ---- graded (soft) gate label ------------------------------------------------
            # The forward pass was always continuous -- phi = sigmoid(logits) mixes tau with
            # tau_free. What makes the trained gate effectively binary is THIS label: BCE against
            # a thresholded 0/1 drives phi to saturate. Contact force spans more than a decade
            # (in-contact p10 2.6-8.3 N, p50 14-29, p90 36-67, max 86.6 on norobot.zarr) and the
            # hard label collapses all of it onto 1.
            # soft_label swaps in log1p(|F|)/log1p(c_ref) clipped to [0,1]: no contact -> 0, a
            # p90-strength contact -> 1, everything between graded. log rather than linear because
            # a linear map would put a light touch (~3 N) at 0.05, indistinguishable from zero.
            # BCE is unchanged -- binary_cross_entropy_with_logits takes soft targets in [0,1] --
            # but its floor becomes the target entropy, so `contact_loss` is NOT comparable
            # against a hard-label run. Compare gate/separation and deployed success instead.
            self.contact_soft = bool(contact_gate.get('soft_label', False))
            if self.contact_soft:
                _key = ('soft_c_ref_global'
                        if (_scope in ('global', 'finger_single') or _gpos == 'feature')
                        else 'soft_c_ref')
                _ref = contact_gate.get(_key, None)
                if _ref is None:
                    raise ValueError(f"contact_gate.soft_label=true needs contact_gate.{_key} "
                                     f"(per-group reference force, same order as the groups)")
                _ref = torch.tensor([float(x) for x in _ref], dtype=torch.float32)
                if _ref.numel() != len(self.contact_groups):
                    raise ValueError(f"contact_gate.{_key} has {_ref.numel()} entries but there "
                                     f"are {len(self.contact_groups)} gate groups")
                self.register_buffer('contact_log_ref', torch.log1p(_ref))

            cprint(f"[SDP3] CONTACT GATE supervision: {len(self.contact_groups)} BCE heads, "
                   f"groups={self.contact_groups}, thresh={self.contact_thresh}, "
                   f"gamma={self.contact_beta}, "
                   f"label={'SOFT log1p(|F|)/log1p(c_ref), c_ref=' + str(_ref.tolist()) if self.contact_soft else 'HARD 0/1'}",
                   "green")

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            # ConditionalUnet1D's final conv emits `input_dim` channels too, so widening once
            # widens both ends -- the joint [action ; torque] denoising target.
            input_dim = action_dim + self.aux_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[SDP3] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[SDP3] pointnet_type: {self.pointnet_type}", "yellow")


        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            # the aux torque channels count as "action" here purely so the generator's
            # `assert D == action_dim + obs_dim` matches the widened trajectory; with
            # obs_as_global_cond the returned mask is all-False either way (nothing inpainted).
            action_dim=action_dim + self.aux_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps


        print_params(self)
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler


        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]


            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory


    def _clamp_state(self, nobs):
        """Clamp normalized agent_pos into [-1,1]; see clamp_agent_pos in __init__.

        Must run BEFORE the auxiliary torque target is sliced out of nobs['agent_pos'], so that
        the observation and the target are clamped identically.

        When it is off, the first few batches are checked for out-of-range values: that is the
        exact signature of percentile torque scaling being enabled on the dataset side without
        this flag, and the two are configured in different places, so it is worth catching.
        """
        if self.clamp_agent_pos:
            nobs['agent_pos'] = nobs['agent_pos'].clamp(-1.0, 1.0)
        elif self._clamp_check_left > 0:
            self._clamp_check_left -= 1
            if float(nobs['agent_pos'].detach().abs().max()) > 1.01:
                cprint("[SDP3] WARNING: normalized agent_pos leaves [-1,1] but "
                       "clamp_agent_pos=False. That is what task.dataset.torque_start "
                       "(percentile torque scaling) produces -- set policy.clamp_agent_pos=true, "
                       "the two switches belong together.", "red")
                self._clamp_check_left = 0
        return nobs

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # move all input tensors to the model device
        obs_dict = dict_apply(obs_dict, lambda x: x.to(device=self.device))
        # normalize input
        nobs = self._clamp_state(self.normalizer.normalize(obs_dict))
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action (+ the aux torque channels, discarded below)
            cond_data = torch.zeros(size=(B, T, Da + self.aux_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # unnormalize the auxiliary torque prediction, if this checkpoint has one. The torque
        # target was normalized as agent_pos[..., lo:hi] (see compute_loss), not by the 'action'
        # normalizer, so it must be inverted the same way: pad zeros into a full agent_pos-width
        # tensor, unnormalize with that field's scale/offset, then slice the torque channels back
        # out (SingleFieldLinearNormalizer requires the last dim to match scale.shape[0] exactly).
        torque_pred = None
        if self.aux_dim:
            lo, hi = self.aux_slice
            ntorque_pred = nsample[..., Da:Da + self.aux_dim]          # (B, T, aux_dim), normalized
            padded = torch.zeros(*ntorque_pred.shape[:-1], hi,
                                  device=ntorque_pred.device, dtype=ntorque_pred.dtype)
            padded[..., lo:hi] = ntorque_pred
            torque_pred = self.normalizer['agent_pos'].unnormalize(padded)[..., lo:hi]

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred,
        }
        if torque_pred is not None:
            result['torque_pred'] = torque_pred

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        nobs = self._clamp_state(self.normalizer.normalize(batch['obs']))
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        if self.aux_dim:
            # Z_t = [A_t ; T_t], where the torque at slot i must be the one action[i] PRODUCES.
            # The zarr row is (obs before action, action): collect_dp3_data.py reads
            # applied_torque AFTER env.step() and stores it as the *next* row's state, so
            # agent_pos[i] carries the response to action[i-1], not to action[i]. Pairing them
            # index-wise would supervise the torque that PRECEDES each action, turning the
            # auxiliary task from "predict the force this action will cause" into "reproduce the
            # force you already felt" -- and the first n_obs_steps frames of that are verbatim in
            # the conditioning, so they cost nothing to fit. Shift the target one step left to
            # line each action up with its own response. The last slot's target is agent_pos
            # [horizon], which is outside the sampled window; it is padded with the final frame
            # here and excluded from the torque loss below.
            lo, hi = self.aux_slice
            tq = nobs['agent_pos'][..., lo:hi]
            tq = torch.cat([tq[:, 1:], tq[:, -1:]], dim=1)
            trajectory = torch.cat([nactions, tq], dim=-1)
        cond_data = trajectory



        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()


        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        


        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        
        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                            local_cond=local_cond, 
                            global_cond=global_cond)


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)

        if self.aux_dim:
            # split the widened trajectory back apart and weight the two objectives separately
            # (TA-VLA: L_joint = L_action + beta * L_torque). `bc_loss` keeps meaning the action
            # term alone, so it stays comparable with runs that have no auxiliary objective.
            # The final torque slot is padding (its true target lies outside the sampled window,
            # see the shift above), so the torque term covers the first horizon-1 steps only.
            Da = self.action_dim
            l_action = reduce(loss[..., :Da], 'b ... -> b (...)', 'mean').mean()
            l_torque = reduce(loss[:, :-1, Da:], 'b ... -> b (...)', 'mean').mean()
            loss = l_action + self.aux_beta * l_torque
            loss_dict = {'bc_loss': l_action.item(), 'torque_loss': l_torque.item()}
        else:
            loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
            loss_dict = {'bc_loss': loss.item()}

        l_contact, gate_stats = self._contact_gate_loss(batch, batch_size)
        if l_contact is not None:
            loss = loss + self.contact_beta * l_contact
            loss_dict['contact_loss'] = l_contact.item()
            loss_dict.update(gate_stats)

        return loss, loss_dict

    def _contact_gate_loss(self, batch, batch_size):
        """BCE on the gate logits + the diagnostics that tell whether the gate actually works.

        Returns (loss_or_None, stats). Reads the logits the encoder stashed on its last forward,
        so it must be called after compute_loss has run the encoder.
        """
        if self.contact_groups is None:
            return None, {}
        logits = getattr(self.obs_encoder, 'last_gate_logits', None)
        if logits is None:
            return None, {}
        contact = batch.get('contact', None)
        if contact is None:
            raise RuntimeError(
                "contact_gate is enabled but the batch has no 'contact' key -- the dataset must "
                "load data/contact (collect with --save_contact)")

        # logits are (B*To, G): the encoder ran once per observation step. Slice contact to the
        # same observation steps and flatten identically so the two line up row for row.
        To = self.n_obs_steps
        c = contact[:, :To].reshape(batch_size * To, -1)             # (B*To, 13)
        # `hard` is kept even in soft mode: every diagnostic below is defined against
        # "is this group touching at all", so the numbers stay comparable with a hard-label run.
        hard = torch.stack([(c[:, g] > self.contact_thresh).any(dim=-1)
                            for g in self.contact_groups], dim=-1).float()     # (B*To, G)
        if self.contact_soft:
            # per-group contact magnitude -> graded target in [0,1]. `.max` over the group's
            # sensor columns mirrors the `.any` above, so soft and hard agree on what "this
            # group's contact" means and differ only in how strongly it is expressed.
            mag = torch.stack([c[:, g].max(dim=-1).values
                               for g in self.contact_groups], dim=-1)          # (B*To, G)
            labels = (torch.log1p(mag.clamp_min(0.0)) / self.contact_log_ref).clamp(0.0, 1.0)
        else:
            labels = hard
        if labels.shape != logits.shape:
            raise RuntimeError(f"contact label/logit shape mismatch: "
                               f"{tuple(labels.shape)} vs {tuple(logits.shape)}")

        l_contact = F.binary_cross_entropy_with_logits(logits, labels)

        # ---- gate health. `separation` is the load-bearing one: it is exactly the quantity
        # that exposed the MoE collapse in arXiv:2604.01414 (0.7634 with contact vs 0.7714
        # without -- a gate that ignores its input). Near 0 here means the same thing.
        with torch.no_grad():
            phi = torch.sigmoid(logits)
            # split by the HARD label, never by `labels`: in soft mode `labels > 0.5` would mean
            # "touching hard enough", a different question, and the metric would stop being
            # comparable with the hard-label control run.
            pos, neg = hard > 0.5, hard < 0.5
            m_c = phi[pos].mean() if pos.any() else phi.new_tensor(float('nan'))
            m_f = phi[neg].mean() if neg.any() else phi.new_tensor(float('nan'))
            stats = {
                'gate/mean_contact': m_c.item(),
                'gate/mean_free': m_f.item(),
                'gate/separation': (m_c - m_f).item(),
                'gate/std': phi.std().item(),
                'gate/pos_rate': hard.mean().item(),
            }
            if self.contact_soft:
                # mean of the graded target, and how much of phi's spread survives inside the
                # contact frames -- a soft gate that still collapses to binary shows up as
                # in_contact_std near 0 despite a graded target.
                stats['gate/label_mean'] = labels.mean().item()
                if pos.any():
                    stats['gate/in_contact_std'] = phi[pos].std().item()
            if To > 1:
                # |phi_t - phi_{t-1}| across the observation steps: if the single-frame gate
                # chatters this is where it shows, and it is the trigger for giving the gate a
                # history window instead.
                p = phi.reshape(batch_size, To, -1)
                stats['gate/jitter'] = (p[:, 1:] - p[:, :-1]).abs().mean().item()
            for i in range(hard.shape[-1]):
                pi, li = phi[:, i], hard[:, i]        # hard, for the same comparability reason
                if li.any() and (~li.bool()).any():
                    stats[f'gate/sep_{i}'] = (pi[li > 0.5].mean() - pi[li < 0.5].mean()).item()
        return l_contact, stats

