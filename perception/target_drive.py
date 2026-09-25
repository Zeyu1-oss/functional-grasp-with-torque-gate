"""Unified direct target drive -- shared by collect / deploy / play so dynamics are
byte-for-byte identical across all three.

env._apply_action natively only understands raw ([-1,1] absolute mapping + EMA + rate
limit + saturation). This module monkeypatches it to directly command joint target
angles q*, so that:
  - deploy: DP3 outputs q* directly -> commanded directly, no target->raw inverse mapping
    (the inverse mapping amplifies arm motion and tends to jitter).
  - collect/play: teacher/MLP output raw -> raw_to_target converts to q* -> commanded directly.

All three use the same code, so collect and deploy dynamics match exactly.

API:
  p = install_direct_drive(env_unwrapped)            # monkeypatch, returns DriveParams
  env_unwrapped._direct_target = q_star              # deploy: DP3's q*
  env_unwrapped._direct_target = raw_to_target(a, cur0, p)   # collect/play: raw->q*

Optional torque feedforward (deploy only, off unless set):
  env_unwrapped._direct_torque_ff = tau_ff           # (num_envs, 13) Nm, or None/unset = no-op
Applied via Articulation.set_joint_effort_target on the SAME controlled joints, every substep,
alongside the position target. For an ImplicitActuator this does not replace the PD term -- PhysX
computes `stiffness*pos_error + damping*vel_error + joint_effort_target` and clips to the effort
limit (isaaclab ActuatorBase._clip_effort; see actuator_pd.py's ImplicitActuator.compute) -- so a
predicted torque here is a feedforward ADDED on top of ordinary position tracking, not a
replacement for it. Left unset, behavior is byte-identical to before this attribute existed.
"""
import types

import torch


class DriveParams:
    """Read the drive constants from env once (same source as env._apply_action)."""

    def __init__(self, env_unwrapped):
        self.s = float(env_unwrapped.action_smoothing)
        self.n_sub = int(getattr(env_unwrapped.cfg, "decimation", 2))
        self.max_joint_delta = float(env_unwrapped.max_joint_delta)
        self.max_finger_delta = float(env_unwrapped.max_finger_delta)
        self.n_arm = int(env_unwrapped.num_arm_joints)
        self.jl = env_unwrapped.joint_lower_limits
        self.ju = env_unwrapped.joint_upper_limits
        self.rng_arm = (self.ju - self.jl)[:self.n_arm]

    def __repr__(self):
        return (f"DriveParams(s={self.s}, n_sub={self.n_sub}, "
                f"max_joint_delta={self.max_joint_delta}, "
                f"max_finger_delta={self.max_finger_delta}, n_arm={self.n_arm})")


def raw_to_target(a, cur0, p):
    """Replicate env._apply_action forward (EMA+rate-limit+saturation, n_sub substeps):
    raw action a -> joint target angles q*.

    a:    (B, 13) raw action [-1,1]; first n_arm dims are arm absolute position, rest are finger deltas.
    cur0: (B, 13) cur_targets before step.
    return: (B, 13) target angles q* this a would have driven cur_targets to.
    """
    na, s, mjd, mfd = p.n_arm, p.s, p.max_joint_delta, p.max_finger_delta
    cur = cur0.clone()
    arm_raw = p.jl[:na] + 0.5 * (a[:, :na] + 1.0) * p.rng_arm
    for _ in range(p.n_sub):
        smoothed = s * arm_raw + (1.0 - s) * cur[:, :na]
        if mjd > 0:
            d = torch.clamp(smoothed - cur[:, :na], -mjd, mjd)
            cur[:, :na] = cur[:, :na] + d
        else:
            cur[:, :na] = smoothed
        cur[:, na:] = cur[:, na:] + a[:, na:] * mfd
        cur = torch.max(torch.min(cur, p.ju.unsqueeze(0)), p.jl.unsqueeze(0))
    return cur


def install_direct_drive(env_unwrapped, rate_limit: bool = False):
    """monkeypatch env._apply_action so each substep directly commands env._direct_target
    (skipping raw interpretation / inverse mapping).

    After calling, set env_unwrapped._direct_target before step each control step:
        deploy:       env._direct_target = q_star (DP3 direct output)
        collect/play:  env._direct_target = raw_to_target(a, cur0, p)
    Returns DriveParams (for collect/play to compute q*; deploy does not need it).

    rate_limit: apply the SAME per-substep slew limit raw_to_target uses (+/-max_joint_delta on the
        arm, +/-max_finger_delta on the fingers) to _direct_target before commanding it.
        Off by default -- with it off this function behaves exactly as before.

        Why it exists: training labels are produced by raw_to_target, which clamps the increment
        once per substep, so a label sequence can move at most n_sub*max_delta per control step
        (measured: arm 0.02 rad, fingers 0.06, with 0.00% of training steps exceeding it). DP3
        outputs q* directly and never passes through raw_to_target, so nothing bounds its step size
        at deploy: measured 15-26% of deployed arm steps exceed 0.02, up to 0.327 rad (16x) in one
        1/60 s control step. Enabling this puts the executed target sequence back inside the
        dynamics the policy was trained on. It is a no-op for any target that already complies
        (collect / RL-teacher paths), so it only ever affects the DP3 path.
    """
    p = DriveParams(env_unwrapped)
    jl, ju = p.jl, p.ju

    # ---- rate limit (removable block: delete this + the `if rate_limit` line below + the arg) ----
    _cap = None
    if rate_limit:
        _n_all = int(jl.shape[0])
        _cap = torch.cat([
            torch.full((p.n_arm,), p.max_joint_delta, device=jl.device, dtype=jl.dtype),
            torch.full((_n_all - p.n_arm,), p.max_finger_delta, device=jl.device, dtype=jl.dtype),
        ]).unsqueeze(0)
        print(f"[RATE-LIMIT] direct-drive slew limit ON: arm +/-{p.max_joint_delta} "
              f"finger +/-{p.max_finger_delta} per substep (x{p.n_sub} substeps = "
              f"{p.max_joint_delta * p.n_sub:.3f}/{p.max_finger_delta * p.n_sub:.3f} per control step), "
              f"matching raw_to_target", flush=True)

    def _apply_action_direct(self):
        tgt = self._direct_target
        if _cap is not None:
            # same shape as raw_to_target: clamp the increment once per substep, so two substeps
            # accumulate to at most 2*cap -- identical to the training-label generation path
            tgt = torch.max(torch.min(tgt, self.cur_targets + _cap), self.cur_targets - _cap)
        tgt = torch.max(torch.min(tgt, ju.unsqueeze(0)), jl.unsqueeze(0))
        self.cur_targets = tgt
        self.franka.set_joint_position_target(tgt, joint_ids=self.controlled_joint_indices)
        # optional torque feedforward -- see module docstring. Unset (None, the default set
        # below) is a genuine no-op: this branch never runs and nothing about position-only
        # control changes relative to before this attribute existed.
        ff = getattr(self, "_direct_torque_ff", None)
        if ff is not None:
            self.franka.set_joint_effort_target(ff, joint_ids=self.controlled_joint_indices)

    env_unwrapped._apply_action = types.MethodType(_apply_action_direct, env_unwrapped)
    env_unwrapped._direct_target = env_unwrapped.cur_targets.clone()   # placeholder, legal value before first step
    env_unwrapped._direct_torque_ff = None   # opt-in: deploy sets this per-step to enable feedforward
    return p
