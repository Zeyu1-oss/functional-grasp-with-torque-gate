import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules




class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')

        # in_channels may exceed 3 (e.g. 4 = xyz+is_handle mask): only the first Linear widens,
        # the [64,128,256] trunk is untouched, so capacity stays identical to the xyz-only encoder
        # (+64 params). This is the "extra channel, same network" path -- distinct from
        # PointNetEncoderXYZRGB, which is a deeper/wider net ([64,128,256,512]) and would confound
        # a mask ablation with an encoder-capacity change.
        assert in_channels >= 3, cprint(
            f"PointNetEncoderXYZ needs at least 3 channels (xyz), but got {in_channels}", "red")
        if in_channels != 3:
            cprint(f"[PointNetEncoderXYZ] in_channels={in_channels} (xyz + "
                   f"{in_channels - 3} extra), same [64,128,256] trunk", "green")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

    


class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 state_split=None,
                 contact_gate=None,
                 state_obs_dim=None,
                 mask_dropout=0.0,
                 mask_modality_dropout=0.0,
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        # ---- mask-channel dropout (only meaningful when the point cloud has a 4th is_handle channel) ----
        # The mask arrives ALREADY NORMALIZED: the dataset's limits-normalizer maps the raw 0/1 label
        # through min=0/max=1, so "not handle" is -1 and "handle" is +1 (NOT 0/1). Dropping therefore
        # means writing -1, never 0 -- 0 would be an un-normalized 0.5, a value that never occurs in
        # the training data and would itself be out of distribution.
        #   mask_dropout          per-POINT: each point's label independently reset to "not handle".
        #                         Simulates the false negatives HandleSegNet will make at deploy time,
        #                         where the mask is predicted rather than read from simulator GT.
        #   mask_modality_dropout per-SAMPLE: the whole mask channel of a sample is blanked, so the
        #                         policy must stay usable when the mask is absent/broken entirely
        #                         (the modality-dropout idea behind force_dropout).
        # Both are active only under .train(); .eval()/deploy passes the mask through untouched.
        self.mask_dropout = float(mask_dropout)
        self.mask_modality_dropout = float(mask_modality_dropout)
        self._mask_off_value = -1.0

        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        # 点云通道数从 shape_meta 读(3=xyz, 4=xyz+mask, 6=xyz+rgb),不再写死。
        # use_pc_color=True 时保留全部通道(4 或 6),走通用 encoder;=False 时只用 xyz(3)。
        _pc_ch = int(self.point_cloud_shape[-1]) if use_pc_color else 3
        # slim_encoder: keep the small xyz trunk ([64,128,256]) even with >3 channels, so that an
        # added channel (e.g. the handle mask) is the ONLY difference from the xyz-only baseline.
        # Default False = legacy behaviour (>3 channels -> the deeper PointNetEncoderXYZRGB).
        _slim = bool(pointcloud_encoder_cfg.get("slim_encoder", False)) \
            if pointcloud_encoder_cfg is not None else False
        if pointnet_type == "pointnet":
            pointcloud_encoder_cfg.in_channels = _pc_ch
            if _pc_ch == 3 or _slim:
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
            else:
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            if _pc_ch != 3:
                _what = ('xyz+mask' if _pc_ch == 4 else 'xyz+rgb' if _pc_ch == 6 else 'xyz+extra')
                cprint(f"[DP3Encoder] point cloud in_channels={_pc_ch} ({_what}), "
                       f"encoder={'PointNetEncoderXYZ (slim)' if _slim else 'PointNetEncoderXYZRGB'}",
                       "green")
                if _pc_ch == 4 and (self.mask_dropout > 0 or self.mask_modality_dropout > 0):
                    cprint(f"[DP3Encoder] MASK dropout: per-point={self.mask_dropout} "
                           f"per-sample={self.mask_modality_dropout} (train only, off-value=-1 "
                           f"= normalized 'not handle')", "green")
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        # ---- state 编码 ----
        # 默认:单个 state_mlp 处理整个 agent_pos(旧行为,concat)。
        # 若 state_split 开启:把 agent_pos 切成 pos([:pos_dim]) 和 force([pos_dim:]),各过一个 MLP。
        #   pos / force 各自独立的 MLP + 可给 force 分支单独的输出维度和 dropout(减轻过度依赖 force)。
        #   agent_pos 维度不变,数据无需重采;deploy 也无需改。
        def _mlp(in_dim, size):
            arch = list(size[:-1]) if len(size) > 1 else []
            return nn.Sequential(*create_mlp(int(in_dim), int(size[-1]), arch, state_mlp_activation_fn))

        # ---- contact-gated torque branch --------------------------------------------------
        # Free-space joint torque is gravity/inertia, not contact. Feeding it in raw is the
        # "feature concatenation" baseline of arXiv:2604.01414, which measured identical to
        # vision-only (30%). The gate suppresses the torque of any finger that is not touching.
        #
        # Three deviations from that paper, each backed by a measurement on norobot.zarr:
        #   * per-finger gate, not one global scalar -- 44% of frames are PARTIAL contact
        #     (1-5 fingers). Per-finger contact is predictable from the matching joint's torque
        #     at AUC 0.93-0.98, versus 0.67-0.75 from a global torque magnitude (their proxy).
        #   * the gate sees ONLY finger signals. Arm joint angles alone predict finger contact
        #     at AUC 0.857 -- a pure task-phase shortcut ("the arm is posed like this, so we are
        #     probably grasping") carrying no contact information. Feeding them in would let the
        #     gate learn the script instead of the physics.
        #   * the arm's 7 torque dims do not enter the branch at all: their std is 10x the
        #     fingers' and is dominated by gravity, so passing them ungated would wave the
        #     noisiest channels straight through while only cleaning the quiet ones.
        # (1-phi)*tau_free is the paper's f*: free space gets its own LEARNED value instead of a
        # zero, so "not touching" stays distinguishable from "touching with zero net torque".
        _cg_on = contact_gate is not None and bool(contact_gate.get('enabled', False))

        # ---- state_obs_dim: torque as OBJECTIVE only -------------------------------------
        # The auxiliary torque target is sliced out of agent_pos (simple_dp3.compute_loss reads
        # nobs['agent_pos'][..., lo:hi]), i.e. the target and the observation are the same tensor.
        # So "torque supervises the trajectory but is NOT observed" -- the pi0+obj cell of the
        # TA-VLA ablation -- cannot be expressed by shrinking agent_pos: that would take the
        # target away too. state_obs_dim keeps the batch at its full width for the target and
        # narrows only what the encoder consumes.
        _obs_dim = int(state_obs_dim) if state_obs_dim else None
        _state_full = self.state_shape[0]
        if _obs_dim is not None:
            if not (0 < _obs_dim <= _state_full):
                raise ValueError(f"state_obs_dim={_obs_dim} must be in (0, {_state_full}]")
            if _obs_dim < _state_full:
                cprint(f"[DP3Encoder] state_obs_dim={_obs_dim}: encoder sees only "
                       f"agent_pos[..., :{_obs_dim}] of {_state_full}. The batch keeps all "
                       f"{_state_full} dims so the auxiliary target can still be sliced from it "
                       f"(torque as objective, not as observation).", "green")
                # nothing left to split once the force block is excluded from the input
                state_split = None
        self.state_obs_dim = _obs_dim if (_obs_dim is not None and _obs_dim < _state_full) else None

        state_dim = _obs_dim if _obs_dim is not None else _state_full
        self.state_split = None
        self.state_mlp = None
        self.contact_gate = None
        if state_split is not None and bool(state_split.get('enabled', True)):
            pos_dim = int(state_split.get('pos_dim', 13))
            force_dim = state_dim - pos_dim
            if force_dim <= 0:
                cprint(f"[DP3Encoder] state_split 开启但 force_dim={force_dim}<=0 "
                       f"(state_dim={state_dim}) -> 回退单 state_mlp", "red")
            else:
                pos_size = tuple(state_split.get('pos_mlp_size', state_mlp_size))
                force_size = tuple(state_split.get('force_mlp_size', state_mlp_size))
                fdrop = float(state_split.get('force_dropout', 0.0))
                # scope decides both how many gates there are and how much torque they carry:
                #   per_finger : one gate per finger joint; the force branch sees ONLY the finger
                #                torques (the arm's are dropped -- std 10x the fingers' and
                #                gravity-dominated, so passing them ungated would wave the
                #                noisiest channels through while cleaning only the quiet ones)
                #   global     : a single gate over ALL force_dim torques, arm included. Coarser
                #                -- 44% of contact frames are PARTIAL (some fingers on the drill,
                #                others still free) and one scalar cannot express that -- but it
                #                matches the mechanism of arXiv:2604.01414 and keeps the arm
                #                torque available whenever the hand is engaged.
                #   finger_single : the finger torque block of per_finger, but a SINGLE gate and the
                #                single global label. Exists so that gate_position=feature (which is
                #                forced to one gate, see below) has a control that differs from it in
                #                the gate POSITION alone -- per_finger would differ in the gate count
                #                as well, and global would differ in the torque block too.
                _scope = str(contact_gate.get('scope', 'per_finger')) if _cg_on else None
                if _cg_on and _scope not in ('per_finger', 'global', 'finger_single'):
                    raise ValueError(f"contact_gate.scope must be per_finger|global|finger_single, "
                                     f"got {_scope}")
                # torque block reaching force_mlp: global keeps the arm, the two finger scopes drop
                # it. The _cg_on guard is required, not cosmetic -- with the gate off contact_gate
                # is None (every checkpoint predating the gate), and _scope is None too, so the
                # condition must not fall through to contact_gate.get().
                n_arm = (int(contact_gate.get('arm_joints', 7))
                         if (_cg_on and _scope != 'global') else 0)
                n_fing = force_dim - n_arm
                if _cg_on and n_fing <= 0:
                    raise RuntimeError(f"contact_gate: force_dim={force_dim} <= arm_joints={n_arm}")
                _force_in = n_fing          # global -> n_arm=0 -> n_fing == force_dim
                # gate count: only per_finger with input gating has more than one
                n_gate = n_fing if _scope == 'per_finger' else 1
                self.pos_mlp = _mlp(pos_dim, pos_size)
                self.force_mlp = _mlp(_force_in if _cg_on else force_dim, force_size)
                self.force_dropout = nn.Dropout(fdrop) if fdrop > 0 else nn.Identity()
                self.state_split = (pos_dim, force_dim)
                self.n_output_channels += int(pos_size[-1]) + int(force_size[-1])
                cprint(f"[DP3Encoder] SPLIT state: pos_mlp({pos_dim}->{pos_size[-1]}) + "
                       f"force_mlp({_force_in if _cg_on else force_dim}->{force_size[-1]}, "
                       f"dropout={fdrop})", "green")
                # gate_position decides WHAT the gate multiplies, and that in turn fixes how many
                # gates there can be:
                #   input   (default, unchanged): phi multiplies the raw torque, tau_free is a
                #           learnable stand-in IN TORQUE SPACE, and the encoder only ever sees the
                #           already-gated signal. One gate per gated torque dim is possible.
                #   feature (arXiv:2604.01414 Eq.1): the encoder always sees the REAL torque and
                #           phi multiplies its OUTPUT --
                #               f_gated = phi * f_torque + (1 - phi) * f*
                #           with f* learnable IN FEATURE SPACE. A 64-d feature cannot be split
                #           across 6 finger gates, so this mode is necessarily a SINGLE global
                #           gate, and the BCE correspondingly gets one head ("is the hand touching
                #           at all", contact_gate.global_group) -- see SimpleDP3.__init__.
                # `scope` keeps its other job either way: which torque block reaches force_mlp
                # (per_finger drops the arm's 7 dims, global keeps all of them).
                _gpos = str(contact_gate.get('gate_position', 'input')) if _cg_on else 'input'
                if _cg_on and _gpos not in ('input', 'feature'):
                    raise ValueError(f"contact_gate.gate_position must be input|feature, got {_gpos}")
                if _gpos == 'feature':
                    n_gate = 1
                if _cg_on:
                    hid = int(contact_gate.get('hidden', 64))
                    if _gpos == 'feature':
                        # torque only, exactly the block the encoder sees. No joint angles: with a
                        # single global label the shortcut risk is at its worst -- finger angles
                        # alone "predict" contact at AUC 0.909 just by encoding "the hand is
                        # closed", which fails precisely on a closed but empty hand.
                        _gin = _force_in
                        _src = f"tau({_force_in}d) alone"
                    elif _scope == 'finger_single':
                        # same input as the feature-gated mode, so the two differ only in where
                        # phi is applied
                        _gin = _force_in
                        _src = f"tau({_force_in}d) alone"
                    elif _scope == 'global':
                        # torque alone, all force_dim of it, no joint angles. Arm TORQUE
                        # legitimately carries contact via J^T F_ext -- any force on the hand
                        # projects onto every upstream joint -- so it belongs in the decision.
                        # Joint ANGLES do not: arm angles alone "predict" finger contact at AUC
                        # 0.857 purely by correlating with task phase, and finger angles at 0.909
                        # by encoding "the hand is closed, so it is probably holding something".
                        # Both are shortcuts that would survive training and then fail exactly
                        # when it matters -- a closed but empty hand. Torque alone measures 0.947,
                        # within 0.008 of q+tau, so dropping them costs almost nothing.
                        _gin = force_dim
                        _src = f"tau({force_dim}d) alone"
                    else:
                        _gin = 2 * (force_dim - int(contact_gate.get('arm_joints', 7)))
                        _src = f"finger block of [q,tau] ({_gin}d)"
                    self.gate_mlp = nn.Sequential(nn.Linear(_gin, hid), nn.ReLU(),
                                                  nn.Linear(hid, n_gate))
                    if _gpos == 'feature':
                        # f* lives in feature space; tau_free is not created at all, so an
                        # input-gated checkpoint and a feature-gated one stay distinguishable
                        # by their state_dict rather than silently loading into each other.
                        self.f_star = nn.Parameter(torch.zeros(int(force_size[-1])))
                    else:
                        self.tau_free = nn.Parameter(torch.zeros(_force_in))
                    self.contact_gate = (pos_dim, n_arm, _force_in, n_gate, _scope, _gpos)
                    if _gpos == 'feature':
                        cprint(f"[DP3Encoder] CONTACT GATE [{_scope}/feature]: 1 gate from {_src} "
                               f"-> sigmoid; gating the force_mlp OUTPUT "
                               f"(f_gated = phi*f_torque + (1-phi)*f_star), "
                               f"f_star learnable({int(force_size[-1])})", "green")
                    else:
                        cprint(f"[DP3Encoder] CONTACT GATE [{_scope}]: {n_gate} gate(s) from "
                               f"{_src} -> sigmoid; gating {_force_in} torque dim(s); "
                               f"tau_free learnable({_force_in})", "green")
        elif _cg_on:
            raise RuntimeError("contact_gate requires state_split (it gates the force branch)")

        if self.state_split is None:
            if len(state_mlp_size) == 0:
                raise RuntimeError(f"State mlp size is empty")
            self.state_mlp = _mlp(state_dim, state_mlp_size)
            self.n_output_channels += int(state_mlp_size[-1])

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    def _apply_mask_dropout(self, points: torch.Tensor) -> torch.Tensor:
        """Blank part of the is_handle channel during training. See __init__ for why the 'off'
        value is -1 (normalized) rather than 0. No-op at eval and when the cloud has no 4th channel."""
        if not self.training or points.shape[-1] < 4:
            return points
        if self.mask_dropout <= 0.0 and self.mask_modality_dropout <= 0.0:
            return points
        m = points[..., 3:4]                                   # (B, N, 1) normalized mask
        off = torch.full_like(m, self._mask_off_value)
        if self.mask_dropout > 0.0:
            # per-point; applied to every point, so in effect it removes `mask_dropout` of the
            # POSITIVE labels (already-negative points are unchanged by writing -1 again)
            m = torch.where(torch.rand_like(m) < self.mask_dropout, off, m)
        if self.mask_modality_dropout > 0.0:
            # per-sample: blank the entire channel for a whole cloud
            drop_b = torch.rand(m.shape[0], 1, 1, device=m.device, dtype=m.dtype) < self.mask_modality_dropout
            m = torch.where(drop_b, off, m)
        return torch.cat([points[..., :3], m, points[..., 4:]], dim=-1)

    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        points = self._apply_mask_dropout(points)
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        pn_feat = self.extractor(points)    # B * out_channel
            
        state = observations[self.state_key]
        if self.state_obs_dim is not None:
            state = state[..., :self.state_obs_dim]
        # stashed rather than returned: the return signature is shared with dp3.py and every
        # other caller. compute_loss reads it right after calling the encoder.
        self.last_gate_logits = None
        if self.state_split is not None:
            pos_dim, _ = self.state_split
            pos_feat = self.pos_mlp(state[..., :pos_dim])
            tau = state[..., pos_dim:]
            force_feat = None      # set directly by the feature-gated path, which skips force_mlp below
            if self.contact_gate is not None:
                _, n_arm, force_in, n_gate, scope, gpos = self.contact_gate
                if gpos == 'feature':
                    # arXiv:2604.01414 Eq.1. The encoder sees the REAL torque -- it is never fed a
                    # fabricated reading -- and the gate decides whether its output or the learned
                    # free-space embedding reaches the conditioning. phi is (..., 1) and broadcasts
                    # over the whole feature.
                    tau_block = tau if scope == 'global' else tau[..., tau.shape[-1] - force_in:]
                    self.last_gate_logits = self.gate_mlp(tau_block)
                    phi = torch.sigmoid(self.last_gate_logits)
                    f_torque = self.force_mlp(tau_block)
                    force_feat = self.force_dropout(phi * f_torque + (1.0 - phi) * self.f_star)
                else:
                    if scope == 'global':
                        gate_in = tau                                  # torque only, all dims
                        tau_gated = tau
                    elif scope == 'finger_single':
                        tau_gated = tau[..., tau.shape[-1] - force_in:]  # finger torques
                        gate_in = tau_gated                             # torque only, one gate
                    else:
                        n_gin = self.gate_mlp[0].in_features // 2
                        tau_gated = tau[..., tau.shape[-1] - n_gin:]   # finger torques
                        gate_in = torch.cat(
                            [state[..., pos_dim - n_gin:pos_dim], tau_gated], dim=-1)
                    self.last_gate_logits = self.gate_mlp(gate_in)
                    phi = torch.sigmoid(self.last_gate_logits)         # (..., n_gate)
                    # phi -> 1 on contact (real torque passes), -> 0 in free space (the learned
                    # tau_free stands in for the gravity/inertia reading, which is not contact
                    # information and would otherwise be indistinguishable from a genuine zero).
                    # per_finger: n_gate == force_in, elementwise. global: n_gate == 1, broadcast
                    # across every torque dim, so one decision opens or closes the whole branch.
                    tau_in = phi * tau_gated + (1.0 - phi) * self.tau_free
            else:
                tau_in = tau
            if force_feat is None:
                force_feat = self.force_dropout(self.force_mlp(tau_in))
            final_feat = torch.cat([pn_feat, pos_feat, force_feat], dim=-1)
        else:
            state_feat = self.state_mlp(state)  # B * 64
            final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels