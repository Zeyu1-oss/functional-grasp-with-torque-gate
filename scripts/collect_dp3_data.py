import argparse
import os
import sys
import time
import shutil

import faulthandler

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import torch


def _safe_compile(model=None, *args, **kwargs):
    if callable(model):
        return model
    return lambda fn: fn
torch.compile = _safe_compile


from perception.dp3_pointcloud import (build_fused_camera_pc, build_plate_cam_pc,
                                       camera_crop_bounds, camera_pc, raise_z_floor,
                                       init_fps_kernel, wrist_cam_pose_w)
from perception.camera_setup import (load_perception_hp as _load_perception_hp,
                                     ensure_rgb_aov, get_cameras, detect_wrist_cam,
                                     apply_render_settings, setup_ground)


PC_NUM_POINTS = 2048


def parse_args():
    _PERC = _load_perception_hp()
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--chained', action='store_true')
    parser.add_argument('--stage1_only', action='store_true')
    parser.add_argument('--stage2_only', action='store_true')
    parser.add_argument('--success_dataset', type=str, default='collected_data/success_data_dp3.pkl',
                        help="pkl of successful stage1 grasp end-states (drill pose + joint pos), used "
                             "by --stage2_only to reset directly into an already-grasped configuration")
    parser.add_argument('--stage1_checkpoint', type=str, default='runs/inspire_hand_grasp_drill_26-06-13-21-10/inspire_hand_grasp_drill_26-06-13-21-10/nn/inspire_hand_grasp_drill_26-06-13-21-10.pth')
    parser.add_argument('--stage2_checkpoint', type=str, default='runs/stage2_26-07-11-18-36/stage2_26-07-11-18-36/nn/stage2_26-07-11-18-36.pth')
    parser.add_argument('--plate_pc_points', type=int, default=512)
    parser.add_argument('--success_hold_stop', type=int, default=20)
    parser.add_argument('--fixed_plate', action='store_true')
    parser.add_argument('--episode_length_s', type=float, default=10.0)
    parser.add_argument("--output", type=str, default="data/inspire_drill_dp3_target_forces_1200.zarr")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument('--episodes_per_variant', type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--drill_configs", type=str, default=None)
    parser.add_argument('--img_height', type=int, default=_PERC.img_height)
    parser.add_argument("--img_width", type=int, default=_PERC.img_width)
    parser.add_argument('--robot_pc_points', type=int, default=1280)
    parser.add_argument('--ground_points', type=int, default=0)
    parser.add_argument('--pc_dtype', type=str, default='float16', choices=['float16', 'float32'])
    parser.add_argument('--workspace', nargs=6, type=float, default=None)
    parser.add_argument("--state_abs_limit", type=float, default=20.0)
    parser.add_argument('--force_state', action='store_true')
    parser.add_argument('--save_privileged', action='store_true')
    parser.add_argument('--save_contact', action='store_true',
                        help="store the per-sensor hand contact-force magnitudes as data/contact "
                             "(T, n_sensors) float32. Read at the same instant as state and the "
                             "point cloud -- i.e. AFTER env.step() -- and written to the same row, "
                             "so contact[k] is the observation at the decision time of action[k], "
                             "which is the force produced by action[k-1]. Same convention as "
                             "state/point_cloud/privileged; nothing here is shifted.")
    parser.add_argument('--disable_cam2', action='store_true',
                        help="cam1 alone supplies the whole camera segment (all PC_NUM_POINTS); "
                             "cam2 (wrist cam) is not sampled into the point cloud")
    parser.add_argument('--robot_pc_per_link', type=int, default=0)
    parser.add_argument('--robot_pc_hand_only', action='store_true')
    parser.add_argument('--no_robot', action='store_true',
                        help="drop the robot FK segment entirely: point_cloud is the camera "
                             "segment alone (PC_NUM_POINTS, plus plate/ground only if those are "
                             "enabled). --robot_pc_points/--robot_pc_hand_only/--robot_pc_per_link "
                             "are then ignored. NOTE --robot_pc_points 0 does NOT do this -- 0 is "
                             "read as 'no downsample' and yields every point in the npz. Deploy "
                             "must be given the same flag.")
    parser.add_argument('--robot_pc_npz', type=str, default='assets/inspire_tac/robot_canonical_points.npz')
    parser.add_argument('--save_init_poses', action='store_true')
    args = parser.parse_args()

    if args.workspace is None:
        args.workspace = list(_PERC.chained_workspace if (args.chained or args.stage2_only) else _PERC.workspace)

    return args


PRIV_DIM = 26

def compute_privileged(env_unwrapped):
    d = env_unwrapped.drill.data
    drill_pos = d.root_pos_w - env_unwrapped.scene.env_origins
    drill_quat = d.root_quat_w
    drill_lin_vel = d.root_lin_vel_w
    drill_ang_vel = d.root_ang_vel_w
    contact_forces = env_unwrapped._get_contact_forces_obs()
    return torch.cat([drill_pos, drill_quat, drill_lin_vel, drill_ang_vel, contact_forces], dim=-1)


def build_sobol_override(env_unwrapped, pos_range, yaw_range, n_per_variant, seed=0):
    from torch.quasirandom import SobolEngine
    from isaaclab.utils.math import quat_mul, quat_from_euler_xyz
    VA = env_unwrapped._variant_attrs
    half = torch.tensor([pos_range[0], pos_range[1], yaw_range])
    rng = np.random.default_rng(seed)

    state = {}
    for vid, vdata in VA.items():
        pos_l = vdata["initial_pos_list"].cpu().float()
        rot_l = vdata["initial_rot_list"].cpu().float()
        goal_l = vdata["goal_rot_list"].cpu().float()
        nb, ng = pos_l.shape[0], goal_l.shape[0]
        per_base = [n_per_variant // nb + (1 if b < n_per_variant % nb else 0)
                    for b in range(nb)]
        P, Q, G, B = [], [], [], []
        for b in range(nb):
            m = per_base[b]
            u = SobolEngine(dimension=3, scramble=True,
                            seed=seed + int(vid) * 101 + b).draw(m)
            off = (u * 2.0 - 1.0) * half
            p = pos_l[b].repeat(m, 1)
            p[:, 0] += off[:, 0]; p[:, 1] += off[:, 1]
            zeros = torch.zeros(m)
            yq = quat_from_euler_xyz(zeros, zeros, off[:, 2])
            q = quat_mul(yq, rot_l[b].repeat(m, 1))
            g = goal_l[torch.arange(m) % ng]
            P.append(p); Q.append(q); G.append(g)
            B.append(torch.full((m,), b, dtype=torch.long))
        P, Q, G, B = torch.cat(P), torch.cat(Q), torch.cat(G), torch.cat(B)
        perm = torch.from_numpy(rng.permutation(len(P)))
        state[vid] = dict(pos=P[perm], quat=Q[perm], goal=G[perm], base=B[perm],
                          cur=0, n=len(P), attempts=0, success_count=0,
                          success=np.full(len(P), -1, dtype=np.int8))
        print(f"[SOBOL] variant {vid}: {len(P)} candidates (bases={nb} each {per_base}, "
              f"goals={ng}) range=±({pos_range[0]}m,{pos_range[1]}m,{yaw_range}rad) | "
              f"exactly one attempt each, failures dropped -> stored episodes = successes "
              f"(<= {len(P)})", flush=True)

    num_envs = int(env_unwrapped.num_envs)
    counted_next = np.zeros(num_envs, dtype=bool)
    idx_next = np.full(num_envs, -1, dtype=np.int64)
    inflight = [None] * num_envs

    def override(env_ids, variant_ids):
        n = len(env_ids)
        pos = torch.zeros(n, 3); quat = torch.zeros(n, 4); goal = torch.zeros(n, 4)
        try:
            success_now = env_unwrapped._cached_lenient_success
        except AttributeError:
            success_now = None
        for i in range(n):
            e = int(env_ids[i]); vid = int(variant_ids[i].item())
            st = state[vid]

            prev = inflight[e]
            if prev is not None and prev["vid"] == vid:
                ok = bool(success_now[e].item()) if success_now is not None else False
                st["success"][prev["idx"]] = 1 if ok else 0
                st["attempts"] += 1
                st["success_count"] += int(ok)

            if st["cur"] >= st["n"]:
                counted_next[e] = False; idx_next[e] = -1
                j = int(rng.integers(0, st["n"]))
                pos[i], quat[i], goal[i] = st["pos"][j], st["quat"][j], st["goal"][j]
                inflight[e] = None
                continue

            j = st["cur"]; st["cur"] += 1
            counted_next[e] = True; idx_next[e] = j
            pos[i], quat[i], goal[i] = st["pos"][j], st["quat"][j], st["goal"][j]
            inflight[e] = dict(vid=vid, idx=j)
        return pos, quat, goal

    override.state = state
    override.counted_next = counted_next
    override.idx_next = idx_next
    return override


def main():
    args = parse_args()
    if args.stage2_only:
        if args.chained or args.stage1_only:
            print("[ERROR] --stage2_only uses Stage2Env directly and is mutually exclusive with "
                  "--chained/--stage1_only (those use ChainedEnv)", flush=True)
            return
        print("[INFO] --stage2_only: Stage2Env, drill starts already grasped (from --success_dataset), "
              "single teacher (--stage2_checkpoint) drives alignment, camera segment = cam2+cam3", flush=True)
    elif args.stage1_only and not args.chained:
        args.chained = True
        _cam_desc = "cam1 alone" if args.disable_cam2 else "cam1+cam2 fused"
        print(f"[INFO] --stage1_only: enable ChainedEnv + wrist camera, but cam3 off (no plate segment). "
              f"camera segment={PC_NUM_POINTS} pts ({_cam_desc}) + robot (up to {args.robot_pc_points} pts); "
              f"teacher1 drives throughout, terminates on stable grasp, no stage2", flush=True)
    simulation_app = None

    faulthandler.enable()

    if args.headless:
        print("[INFO] Running headless collection with IsaacLab-managed TiledCamera sensors.", flush=True)

    if args.stage2_only:
        if not os.path.exists(args.stage2_checkpoint):
            print(f"[ERROR] Checkpoint not found: {args.stage2_checkpoint}")
            return
        if not os.path.exists(args.success_dataset):
            print(f"[ERROR] --success_dataset not found: {args.success_dataset} "
                  f"(generate with scripts/collect_success_data.py, or pass --success_dataset)")
            return
    elif args.chained:
        for _p in (args.stage1_checkpoint, args.stage2_checkpoint):
            if not os.path.exists(_p):
                print(f"[ERROR] Checkpoint not found: {_p}")
                return
    else:
        if not args.checkpoint or not os.path.exists(args.checkpoint):
            print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
            return

    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = app_launcher.app

    _exit_code = 0
    try:
        print("[IMP] math/yaml ...", flush=True)
        import math
        import yaml
        print("[IMP] carb/omni ...", flush=True)
        import carb
        import omni.timeline
        import omni.usd
        print("[IMP] rl_games.common (this triggers torch.compile -> torch._dynamo -> triton dlopen) ...", flush=True)
        from rl_games.common import env_configurations, vecenv
        print("[IMP] rl_games.torch_runner ...", flush=True)
        from rl_games.torch_runner import Runner
        print("[IMP] isaaclab_rl.rl_games ...", flush=True)
        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
        print("[IMP] tasks.grasp_drill_env ...", flush=True)
        from tasks.grasp_drill_env import GraspDrillEnv, create_grasp_drill_env_cfg
        print("[IMP] triton-dependent imports done", flush=True)

        print("[IMP] zarr ...", flush=True)
        import zarr
        print("[IMP] numcodecs ...", flush=True)
        import numcodecs
        print("[IMP] psutil ...", flush=True)
        import psutil
        print("[IMP] pytorch3d FPS kernel ...", flush=True)
        if init_fps_kernel("/home/zeyu/3D-Diffusion-Policy/third_party/pytorch3d_simplified"):
            print("[IMP] pytorch3d FPS available (fused CUDA kernel)", flush=True)
        else:
            print("[IMP] pytorch3d FPS NOT available -> python loop fallback", flush=True)
        print("[IMP] ALL imports done", flush=True)

        faulthandler.dump_traceback_later(60, repeat=True)

        omni.usd.get_context().new_stage()

        rl_config_path = os.path.join(project_root, "config/agents/rl_games_ppo_cfg.yaml")
        with open(rl_config_path, "r") as f:
            agent_cfg = yaml.safe_load(f)

        agent_cfg["params"]["config"]["num_actors"] = args.num_envs
        agent_cfg["params"]["config"]["device"] = args.device
        agent_cfg["params"]["config"]["device_name"] = args.device

        cfg = create_grasp_drill_env_cfg(
            num_envs=args.num_envs,
            device=args.device,
            headless=args.headless,
            debug=args.debug,
            drill_config_path=args.drill_configs,
            img_height=args.img_height,
            img_width=args.img_width,
            enable_cameras=True,
            include_plate=args.chained or args.stage2_only,
            include_plate_camera=args.chained or args.stage2_only,
            include_cam3=(args.chained and not args.stage1_only) or args.stage2_only,
        )
        cfg.seed = args.seed
        _log_ckpt = args.stage2_checkpoint if (args.chained or args.stage2_only) else args.checkpoint
        cfg.log_dir = os.path.dirname(os.path.dirname(_log_ckpt))
        if args.chained or args.stage2_only:
            cfg.episode_length_s = args.episode_length_s

        ensure_rgb_aov(cfg.scene)

        if args.stage2_only:
            from tasks.stage2_env import get_stage2_env_class, SuccessDataDataset
            Stage2Env = get_stage2_env_class()
            success_dataset = SuccessDataDataset(args.success_dataset)
            base_env = Stage2Env(cfg=cfg, debug=args.debug, success_dataset=success_dataset,
                                success_hold_stop=args.success_hold_stop)
            print(f"[STAGE2] Stage2Env: drill starts already grasped ({len(success_dataset)} pkl samples), "
                  f"single teacher (stage2) drives alignment, terminates early after "
                  f"{args.success_hold_stop} consecutive aligned steps (episode cap={args.episode_length_s}s)",
                  flush=True)
        elif args.chained:
            from tasks.chained_env import get_chained_env_class
            ChainedEnv = get_chained_env_class()
            base_env = ChainedEnv(cfg=cfg, debug=args.debug,
                                  success_hold_stop=args.success_hold_stop,
                                  stage1_only=args.stage1_only)
            _mode = "stage1_only (terminate on stable grasp, collect grasp segment only)" if args.stage1_only \
                else "stage1 reset + plate randomization + switch to stage2 on stable grasp"
            print(f"[CHAINED] ChainedEnv: {_mode}, "
                  f"success_hold_stop={args.success_hold_stop}, episode={args.episode_length_s}s",
                  flush=True)
        else:
            base_env = GraspDrillEnv(cfg=cfg, debug=args.debug)
        env_unwrapped = base_env.unwrapped

        if args.fixed_plate:
            _fp_pos = torch.tensor([0.0, 1.0, 0.5], device=args.device)
            _fp_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=args.device)

            def _plate_pose_override_fn(env_ids):
                n = len(env_ids)
                return _fp_pos.unsqueeze(0).expand(n, -1), _fp_quat.unsqueeze(0).expand(n, -1)

            env_unwrapped._plate_pose_override_fn = _plate_pose_override_fn
            print("[INFO] --fixed_plate: plate pinned at (0,1,0.5) default base, no random jitter each reset",
                  flush=True)

        from isaaclab.sim.utils import delete_prim
        try:
            delete_prim("/World/Template")
        except Exception as e:
            print(f"[WARN] failed to delete /World/Template: {e}", flush=True)

        print("[INFO] Resetting environment (before play)...", flush=True)
        _ = env_unwrapped.reset()

        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        env = RlGamesVecEnvWrapper(base_env, args.device, clip_obs, clip_actions)

        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw),
        )
        env_configurations.register(
            "rlgpu",
            {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env},
        )

        if args.chained:
            import importlib.util as _ilu
            _pc_spec = _ilu.spec_from_file_location(
                "_play_chained", os.path.join(project_root, "scripts", "play_chained.py"))
            _pc_mod = _ilu.module_from_spec(_pc_spec)
            _pc_spec.loader.exec_module(_pc_mod)
            d1 = _pc_mod._ckpt_obs_dim(args.stage1_checkpoint)
            d2 = _pc_mod._ckpt_obs_dim(args.stage2_checkpoint)
            print(f"[CHAINED] stage1 obs dim={d1}, stage2 obs dim={d2}", flush=True)
            if d2 != d1 + 7:
                print(f"[WARN] stage2 dim ({d2}) != stage1 ({d1})+7, confirm the checkpoints match", flush=True)
            act_dim = int(env_unwrapped.cfg.action_space)
            agent1 = _pc_mod._build_player(agent_cfg, args.stage1_checkpoint, d1, act_dim,
                                           args.num_envs, args.device)
            agent2 = _pc_mod._build_player(agent_cfg, args.stage2_checkpoint, d2, act_dim,
                                           args.num_envs, args.device)
            agent = agent1
            print("[CHAINED] dual teacher loaded (stage1 grasp / stage2 align)", flush=True)
        else:
            _single_ckpt = args.stage2_checkpoint if args.stage2_only else args.checkpoint
            agent_cfg["params"]["load_checkpoint"] = True
            agent_cfg["params"]["load_path"] = _single_ckpt
            agent_cfg["params"]["config"]["num_actors"] = env_unwrapped.num_envs

            runner = Runner()
            runner.load(agent_cfg)
            agent = runner.create_player()
            agent.restore(_single_ckpt)
            agent.reset()
            if args.stage2_only:
                print(f"[STAGE2] single teacher loaded from {_single_ckpt}", flush=True)

        dt = env_unwrapped.step_dt
        cam1, cam2, cam3 = get_cameras(env_unwrapped, chained=args.chained,
                                       need_cam3=(args.chained and not args.stage1_only) or args.stage2_only)
        _use_plate = args.chained and (cam3 is not None)
        _cam2_on_body = detect_wrist_cam(env_unwrapped, cam2)
        apply_render_settings(dt)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        sys.stdout.flush()
        if not simulation_app.is_running():
            print("[ERROR] simulation_app.is_running() is False before loop!")
            return

        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs.get("obs", obs)
        if args.chained:
            agent1.get_batch_size(obs[:, :d1], 1)
            agent1.reset()
            agent2.get_batch_size(obs, 1)
            agent2.reset()
        else:
            agent.get_batch_size(obs, 1)
            agent.reset()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(args.device)

        controlled_indices = env_unwrapped.controlled_joint_indices.cpu()
        workspace = tuple(args.workspace)

        from perception.target_drive import install_direct_drive, raw_to_target
        _drive_p = install_direct_drive(env_unwrapped)
        print(f"[TARGET_DRIVE] collect direct target drive: {_drive_p}", flush=True)

        robot_fk = None
        robot_pc_M = 0
        if args.no_robot:
            print("[INFO] --no_robot: the robot FK segment is not built; point_cloud is the camera "
                  "segment only (plus plate/ground if those are enabled)", flush=True)
        else:
            from perception.robot_pointcloud import RobotPointCloudFK, HAND_LINK_PREFIXES
            _npz = args.robot_pc_npz
            if not os.path.isabs(_npz):
                _npz = os.path.join(project_root, _npz)
            _rmax = args.robot_pc_points if (args.robot_pc_points and args.robot_pc_points > 0) else None
            robot_fk = RobotPointCloudFK(_npz, list(env_unwrapped.franka.body_names), args.device,
                                         max_points=_rmax,
                                         link_filter=(HAND_LINK_PREFIXES
                                                      if args.robot_pc_hand_only else None),
                                         per_link_points=(args.robot_pc_per_link or None))
            robot_pc_M = robot_fk.num_points
            print(f"[INFO] robot {robot_pc_M} points merged into point_cloud, npz={_npz}", flush=True)

        from perception.robot_pointcloud import jitter_ground_xy
        _PERC = _load_perception_hp()
        ground_batch, ground_M, _ground_xy_std = setup_ground(args.num_envs, workspace, args.device,
                                                              num_points=args.ground_points)
        _cam_follow = _PERC.camera_follow_drill
        _drill_crop_half = _PERC.drill_crop_half
        if _cam_follow:
            print(f"[INFO] camera crop follows the drill: center=GT pose, {2*_drill_crop_half*100:.0f}cm cube,"
                  f"box bottom z>={workspace[4]} (no table pickup, ground handled by synthetic cloud), oracle sim-only", flush=True)

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        blosc_compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=1)
        zarr_root = zarr.group(args.output, zarr_format=2)
        zarr_data = zarr_root.create_group("data")
        zarr_meta = zarr_root.create_group("meta")
        zarr_root.attrs["action_timing"] = "obs_before_action"
        print("[TIMING] collection timing = official convention: row=(obs at decision time, this-step action)."
              "ckpt trained on this data must deploy with --lag_comp 0; cannot mix with old-gen (action-shifted) zarr",
              flush=True)

        if args.disable_cam2:
            print(f"[INFO] --disable_cam2: cam1 alone supplies the camera segment "
                  f"({PC_NUM_POINTS} points); cam2 (wrist cam) not sampled", flush=True)
        if args.stage2_only:
            print(f"[INFO] --stage2_only: camera segment = cam2+cam3 ({PC_NUM_POINTS} points, "
                  f"{PC_NUM_POINTS//2} each); cam1 not sampled, no separate plate segment", flush=True)
        plate_M = args.plate_pc_points if _use_plate else 0
        total_pc = (PC_NUM_POINTS
                    + plate_M
                    + robot_pc_M
                    + ground_M)
        print(f"[INFO] chained={args.chained} -> total_pc={total_pc} "
              f"(camera={PC_NUM_POINTS} + plate={plate_M} "
              f"+ robot={robot_pc_M} "
              f"+ ground={ground_M}) "
              f"| training config point_cloud.shape must be set to [{total_pc}, 3]", flush=True)
        _CHUNK_FRAMES = 16
        from perception.student_obs import build_agent_pos, agent_pos_dim, POS_DIM
        _state_dim = agent_pos_dim(args.force_state)
        print(f"[STATE] force_state={args.force_state} -> state dim={_state_dim} "
              f"({'pos13+torque13' if args.force_state else 'pos13'})", flush=True)
        zarr_data.create_dataset("state", shape=(0, _state_dim), dtype="float32",
                                 compressor=blosc_compressor, chunks=(_CHUNK_FRAMES, _state_dim), overwrite=True)
        zarr_data.create_dataset("action", shape=(0, 13), dtype="float32",
                                 compressor=blosc_compressor, chunks=(_CHUNK_FRAMES, 13), overwrite=True)
        zarr_data.create_dataset("point_cloud", shape=(0, total_pc, 3), dtype=args.pc_dtype,
                                 compressor=blosc_compressor, chunks=(_CHUNK_FRAMES, total_pc, 3), overwrite=True)
        print(f"[INFO] point_cloud storage precision: {args.pc_dtype}"
              f"{'(fp16, training side astype(float32) converts back)' if args.pc_dtype == 'float16' else ''}", flush=True)
        if args.save_privileged:
            zarr_data.create_dataset("privileged", shape=(0, PRIV_DIM), dtype="float32",
                                     compressor=blosc_compressor, chunks=(_CHUNK_FRAMES, PRIV_DIM), overwrite=True)
            print(f"[INFO] save_privileged=ON -> data/privileged dim={PRIV_DIM}", flush=True)
        _contact_dim = 0
        if args.save_contact:
            # probed from the env rather than hard-coded: the sensor list lives in
            # GraspDrillEnv._get_contact_forces_obs and a hard-coded width here would silently
            # mis-shape the dataset the moment a sensor is added or removed there.
            _contact_dim = int(env_unwrapped._get_contact_forces_obs().shape[1])
            zarr_data.create_dataset("contact", shape=(0, _contact_dim), dtype="float32",
                                     compressor=blosc_compressor,
                                     chunks=(_CHUNK_FRAMES, _contact_dim), overwrite=True)
            print(f"[INFO] save_contact=ON -> data/contact dim={_contact_dim} "
                  f"(per-sensor force magnitude, clamped to +/-50N upstream); read after "
                  f"env.step() and stored on the same row as state -> contact[k] is the force "
                  f"felt at the decision time of action[k], produced by action[k-1]", flush=True)
        zarr_meta.create_dataset("episode_ends", shape=(0,), dtype="int64",
                                 compressor=blosc_compressor, chunks=(100,), overwrite=True)

        episode_ends = []

        active_vids = sorted(v.variant_index for v in env_unwrapped.drill_variants)
        collected_per_variant = {vid: 0 for vid in active_vids}
        if args.episodes_per_variant is not None:
            args.num_episodes = args.episodes_per_variant * len(active_vids)
            print(f"[INFO] Per-variant quota: {args.episodes_per_variant} x "
                  f"{len(active_vids)} variants {active_vids} = {args.num_episodes} attempts",
                  flush=True)

        _rcfg = env_unwrapped.cfg.randomization
        _pr = tuple(_rcfg.drill_pos_random_range[:2])
        _yr = _rcfg.drill_rot_random_range[2]
        _npv = args.episodes_per_variant or max(1, args.num_episodes // len(active_vids))
        _sobol_fn = build_sobol_override(env_unwrapped, _pr, _yr, _npv, seed=args.seed)
        env_unwrapped._drill_pose_override_fn = _sobol_fn
        print(f"[SOBOL] pos_range=±{_pr} yaw_range=±{_yr} (from cfg.randomization) | "
              f"{_npv} attempts per variant, one per candidate -- a failed candidate is dropped, "
              f"not retried, so the stored count per variant is the teacher's success count on "
              f"the exam (expect ~{_npv} x pass_rate)", flush=True)

        buffers = [[] for _ in range(args.num_envs)]
        env_started = np.zeros(args.num_envs, dtype=bool)
        cur_counted = np.zeros(args.num_envs, dtype=bool)
        cur_qidx = np.full(args.num_envs, -1, dtype=np.int64)
        attempts_per_variant = {v: 0 for v in active_vids}
        prev_state_np = None
        prev_pc_np = None
        prev_priv_np = None
        prev_contact_np = None
        prev_bad_np = np.ones(args.num_envs, dtype=bool)
        total_collected = 0
        total_steps = 0
        total_zero_pc = 0
        total_zero_pc_c1 = 0
        total_zero_pc_c2 = 0
        total_zero_pc_c3 = 0
        n_dropped_abnormal = 0
        print_interval = 100
        start_time = time.time()

        flushed_init_poses = []
        _pose_dir = args.output.rstrip("/") + "_init_poses"
        _pose_file = os.path.join(_pose_dir, "init_poses.npz")

        def _save_init_poses_now():
            if not flushed_init_poses:
                return
            os.makedirs(_pose_dir, exist_ok=True)
            _vid = np.array([p[0] for p in flushed_init_poses], dtype=np.int64)
            _pos = np.stack([p[1] for p in flushed_init_poses]).astype(np.float32)
            _quat = np.stack([p[2] for p in flushed_init_poses]).astype(np.float32)
            _goal = np.stack([p[3] for p in flushed_init_poses]).astype(np.float32)
            np.savez(_pose_file, variant=_vid, pos_local=_pos, quat=_quat, goal_quat=_goal)

        def read_init_pose(e):
            eo = env_unwrapped.scene.env_origins[e]
            pos_local = (env_unwrapped.initial_drill_pos[e] - eo).detach().cpu().numpy().astype(np.float32)
            quat = env_unwrapped.drill_initial_rot_tensor[e].detach().cpu().numpy().astype(np.float32)
            goal = env_unwrapped._drill_goal_quat_per_env[e].detach().cpu().numpy().astype(np.float32)
            vid = int(env_unwrapped._drill_variant_indices[e].item())
            return (vid, pos_local, quat, goal)
        buf_init_pose = [read_init_pose(e) for e in range(args.num_envs)] if args.save_init_poses else None

        def flush_episode(states, actions, pcs, privs, vid, init_pose=None,
                          contacts=None):
            nonlocal total_collected, n_dropped_abnormal
            n = len(states)
            if n == 0:
                return
            if (args.episodes_per_variant is not None
                    and collected_per_variant[vid] >= args.episodes_per_variant):
                return
            states_arr = np.stack(states).astype(np.float32)
            ep_max_abs = float(np.abs(states_arr[:, :POS_DIM]).max())
            if ep_max_abs > args.state_abs_limit:
                n_dropped_abnormal += 1
                print(f"  [DROP] episode (len={n}): state max|val|={ep_max_abs:.1f} "
                      f"> {args.state_abs_limit}, discarded", flush=True)
                return
            actions_arr = np.stack(actions).astype(np.float32)
            pcs_arr = np.stack(pcs).astype(np.float32)
            idx = zarr_data["state"].shape[0]
            zarr_data["state"].resize((idx + n, _state_dim))
            zarr_data["action"].resize((idx + n, 13))
            zarr_data["point_cloud"].resize((idx + n, total_pc, 3))
            zarr_data["state"][idx:idx + n] = states_arr
            zarr_data["action"][idx:idx + n] = actions_arr
            zarr_data["point_cloud"][idx:idx + n] = pcs_arr
            if args.save_privileged:
                privs_arr = np.stack(privs).astype(np.float32)
                zarr_data["privileged"].resize((idx + n, PRIV_DIM))
                zarr_data["privileged"][idx:idx + n] = privs_arr
            if args.save_contact and contacts is not None:
                contacts_arr = np.stack(contacts).astype(np.float32)
                zarr_data["contact"].resize((idx + n, _contact_dim))
                zarr_data["contact"][idx:idx + n] = contacts_arr
            episode_ends.append(idx + n)
            ends = np.array(episode_ends, dtype=np.int64)
            zarr_meta["episode_ends"].resize((len(ends),))
            zarr_meta["episode_ends"][:] = ends
            collected_per_variant[vid] += 1
            total_collected += 1
            if args.save_init_poses and init_pose is not None:
                flushed_init_poses.append(init_pose)
                _save_init_poses_now()

        faulthandler.cancel_dump_traceback_later()
        print(f"Running a {args.num_episodes}-attempt Sobol exam "
              f"(num_envs={args.num_envs}, drive=direct-target)...", flush=True)

        def _work_remaining():
            if any(s["cur"] < s["n"] for s in _sobol_fn.state.values()):
                return True
            return bool(cur_counted.any())

        try:
            with torch.inference_mode():
                while _work_remaining() and simulation_app.is_running():
                    if args.chained:
                        _a1 = agent1.get_action(agent1.obs_to_torch(obs[:, :d1]), is_deterministic=True)
                        _a2 = agent2.get_action(agent2.obs_to_torch(obs), is_deterministic=True)
                        teacher_raw = torch.where(
                            (env_unwrapped.phase == 1).unsqueeze(1), _a2, _a1)
                    else:
                        obs_tensor = agent.obs_to_torch(obs)
                        teacher_raw = agent.get_action(obs_tensor, is_deterministic=True)
                    if not isinstance(teacher_raw, torch.Tensor):
                        teacher_raw = torch.from_numpy(np.array(teacher_raw)).float().to(args.device)
                    actions = teacher_raw

                    cur0 = env_unwrapped.cur_targets.clone()
                    env_unwrapped._direct_target = raw_to_target(actions, cur0, _drive_p)

                    label_np = env_unwrapped._direct_target.detach().cpu().numpy()
                    if prev_state_np is not None:
                        for env_id in range(args.num_envs):
                            if not env_started[env_id] or prev_bad_np[env_id]:
                                continue
                            buffers[env_id].append(
                                (prev_state_np[env_id], label_np[env_id], prev_pc_np[env_id],
                                 None if prev_priv_np is None else prev_priv_np[env_id],
                                 None if prev_contact_np is None else prev_contact_np[env_id]))

                    obs_dict, rewards, terminated, truncated, extras = env_unwrapped.step(actions)
                    total_steps += 1
                    cam1.update(dt)
                    cam2.update(dt)
                    if cam3 is not None:
                        cam3.update(dt)

                    obs = env._process_obs(obs_dict)

                    state_vec = build_agent_pos(env_unwrapped, args.force_state)

                    _cam_budget = PC_NUM_POINTS
                    if args.stage2_only:
                        _half = _cam_budget // 2
                        _ws = camera_crop_bounds(env_unwrapped.drill.data.root_pos_w,
                                                 env_unwrapped.scene.env_origins, _PERC, workspace)
                        _cam2_pose = (wrist_cam_pose_w(env_unwrapped.franka, cam2)
                                      if _cam2_on_body else None)
                        _ws_cam2 = raise_z_floor(_ws, getattr(_PERC, "wrist_cam_z_floor", None))
                        pc2, zmask2, _ = camera_pc(cam2, _ws_cam2, _half, env_unwrapped.scene.env_origins,
                                                   pose_w=_cam2_pose)
                        pc3, zmask3_cam, _ = camera_pc(cam3, _ws, _half, env_unwrapped.scene.env_origins)
                        pc_fused = torch.zeros(args.num_envs, _cam_budget, 3,
                                               device=args.device, dtype=torch.float32)
                        pc_fused[:, :_half] = pc2
                        pc_fused[:, _half:_half * 2] = pc3
                        zmask1 = torch.zeros_like(zmask2)
                    elif args.disable_cam2:
                        _crop1 = camera_crop_bounds(env_unwrapped.drill.data.root_pos_w,
                                                    env_unwrapped.scene.env_origins, _PERC, workspace)
                        pc_fused, zmask1, _ = camera_pc(cam1, _crop1, _cam_budget,
                                                        env_unwrapped.scene.env_origins)
                        zmask2 = torch.zeros_like(zmask1)
                    else:
                        _cam2_pose = (wrist_cam_pose_w(env_unwrapped.franka, cam2)
                                      if _cam2_on_body else None)
                        pc_fused, zmask1, zmask2, _, _ = build_fused_camera_pc(
                            cam1, cam2, env_unwrapped.drill.data.root_pos_w,
                            env_unwrapped.scene.env_origins, _PERC, _cam_budget, workspace,
                            cam2_pose_w=_cam2_pose)

                    _z1 = int(zmask1.sum().item()); _z2 = int(zmask2.sum().item())
                    total_zero_pc += _z1 + _z2
                    total_zero_pc_c1 += _z1
                    total_zero_pc_c2 += _z2

                    zmask3 = zmask3_cam if args.stage2_only else None
                    if args.stage2_only:
                        _z3 = int(zmask3.sum().item())
                        total_zero_pc += _z3
                        total_zero_pc_c3 += _z3
                    elif _use_plate:
                        pc_plate, zmask3, _ = build_plate_cam_pc(
                            cam3, env_unwrapped.plate.data.root_pos_w,
                            env_unwrapped.scene.env_origins, plate_M)
                        pc_fused = torch.cat([pc_fused, pc_plate], dim=1)
                        _z3 = int(zmask3.sum().item())
                        total_zero_pc += _z3
                        total_zero_pc_c3 += _z3

                    if not args.disable_cam2 and not args.stage2_only and _cam2_on_body and (total_steps <= 2 or total_steps % 200 == 0):
                        _fk_p = _cam2_pose[0][0]
                        _st_p = cam2.data.pos_w[0]
                        _half = PC_NUM_POINTS // 2
                        _p2 = pc_fused[0, _half:PC_NUM_POINTS]
                        _ref = torch.cat([pc_fused[0, :_half],
                                          pc_fused[0, PC_NUM_POINTS:
                                                   PC_NUM_POINTS + plate_M]])
                        _p2 = _p2[_p2.abs().sum(1) > 1e-6]; _ref = _ref[_ref.abs().sum(1) > 1e-6]
                        if len(_ref) > 0 and len(_p2) > 0:
                            _nn = torch.cdist(_p2, _ref).min(dim=1).values
                            _nn_med = _nn.median().item()
                        else:
                            _nn_med = float("nan")

                    if robot_fk is not None:
                        robot_pc = robot_fk(env_unwrapped.franka.data.body_pos_w,
                                            env_unwrapped.franka.data.body_quat_w,
                                            env_unwrapped.scene.env_origins)
                        pc_fused = torch.cat([pc_fused, robot_pc], dim=1)

                    _ground_jit = jitter_ground_xy(ground_batch, _ground_xy_std,
                                                   workspace[0], workspace[1], workspace[2], workspace[3])
                    pc_fused = torch.cat([pc_fused, _ground_jit], dim=1)

                    state_np = state_vec.detach().cpu().numpy()
                    pc_np = pc_fused.detach().cpu().numpy()
                    priv_np = (compute_privileged(env_unwrapped).detach().cpu().numpy()
                               if args.save_privileged else None)
                    # same instant as state_vec / pc_fused above (all read after env.step()), so it
                    # lands on the same row and carries the same "observation before the action"
                    # meaning. Do NOT move this read to before the step.
                    contact_np = (env_unwrapped._get_contact_forces_obs().detach().cpu().numpy()
                                  if args.save_contact else None)

                    if total_steps == 1 and robot_pc_M > 0:
                        _base = env_unwrapped.franka.data.body_pos_w[0, 0].detach().cpu().numpy()
                        _rs = PC_NUM_POINTS + plate_M
                        _r = pc_np[0, _rs:_rs + robot_pc_M]
                        _c = pc_np[0, :PC_NUM_POINTS]
                        _cc = _c[(np.abs(_c).sum(1) > 1e-6)]
                        if len(_cc) > 0:
                            _d = np.linalg.norm(_r[:, None, :] - _cc[None, :, :], axis=-1).min(1)
                            print(f"[FK-CHECK] base(world)={np.round(_base,3)} | "
                                  f"robot segment env-local bbox "
                                  f"x[{_r[:,0].min():.2f},{_r[:,0].max():.2f}] "
                                  f"y[{_r[:,1].min():.2f},{_r[:,1].max():.2f}] "
                                  f"z[{_r[:,2].min():.2f},{_r[:,2].max():.2f}] | "
                                  f"NN->camera p10={np.percentile(_d,10)*100:.1f}cm "
                                  f"median={np.median(_d)*100:.1f}cm "
                                  f"(p10 should be cm-scale, else the base/frame is wrong)", flush=True)

                    is_done = terminated.bool() | truncated.bool()
                    try:
                        lenient_success = env_unwrapped._cached_lenient_success
                    except AttributeError:
                        lenient_success = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
                    is_done_np = is_done.detach().cpu().numpy()
                    success_np = lenient_success.detach().cpu().numpy().astype(bool)
                    if args.stage2_only:
                        _bad = zmask2 | zmask3
                    elif _use_plate:
                        _bad = zmask3
                    else:
                        _bad = zmask1 | zmask2
                    bad_pc_np = _bad.detach().cpu().numpy()
                    prev_state_np = state_np
                    prev_pc_np = pc_np
                    prev_priv_np = priv_np
                    prev_contact_np = contact_np
                    prev_bad_np = bad_pc_np

                    any_done = False
                    for env_id in range(args.num_envs):
                        if is_done_np[env_id]:
                            any_done = True
                            vid = int(env_unwrapped._drill_variant_indices[env_id].item())
                            _record = bool(cur_counted[env_id])
                            if _record:
                                attempts_per_variant[vid] += 1
                            if _record and env_started[env_id] and buffers[env_id] and success_np[env_id]:
                                states, acts, pcs, privs, cts = zip(*buffers[env_id])
                                flush_episode(list(states), list(acts), list(pcs), list(privs), vid,
                                              init_pose=(buf_init_pose[env_id] if args.save_init_poses else None),
                                              contacts=(list(cts) if args.save_contact else None))
                            buffers[env_id] = []
                            env_started[env_id] = True
                            cur_counted[env_id] = bool(_sobol_fn.counted_next[env_id])
                            cur_qidx[env_id] = int(_sobol_fn.idx_next[env_id])
                            if args.save_init_poses:
                                buf_init_pose[env_id] = read_init_pose(env_id)
                        elif not env_started[env_id]:
                            buffers[env_id] = []
                            env_started[env_id] = True

                    if any_done:
                        agent.reset()
                        if args.chained:
                            agent2.reset()

                    if total_steps % print_interval == 0:
                        elapsed = max(time.time() - start_time, 1e-6)
                        _done_n = sum(attempts_per_variant.values())
                        _tot_n = sum(s["n"] for s in _sobol_fn.state.values())
                        eta_sec = (elapsed / _done_n * (_tot_n - _done_n) if _done_n > 0 else 0)
                        m, s = divmod(int(eta_sec), 60)
                        h, m = divmod(m, 60)
                        filled = int(30 * min(_done_n / max(_tot_n, 1), 1.0))
                        bar = "#" * filled + "-" * (30 - filled)
                        proc = psutil.Process()
                        cpu_mem_gb = proc.memory_info().rss / 1e9
                        gpu_mem_gb = (torch.cuda.max_memory_allocated(args.device) / 1e9
                                      if torch.cuda.is_available() else 0)
                        num_pending = sum(len(b) for b in buffers)
                        per_var = " ".join(f"v{vid}={n}/{attempts_per_variant[vid]}"
                                           for vid, n in collected_per_variant.items())
                        _s2 = (f" | s2={int((env_unwrapped.phase == 1).sum().item())}/{args.num_envs}"
                               if args.chained else "")
                        _head = (f"attempts {_done_n}/{_tot_n} stored {total_collected} "
                                 f"(pass {total_collected / max(_done_n, 1):.0%})")
                        print(f"  [{bar}] {_head} ({per_var}) | {total_steps}stp | "
                              f"{total_steps / elapsed:.0f}/s | ETA={h}h{m}m{s}s | "
                              f"RAM={cpu_mem_gb:.1f}GB GPU={gpu_mem_gb:.1f}GB | "
                              f"pending={num_pending} | zero_pc={total_zero_pc}"
                              f"(cam1={total_zero_pc_c1} cam2={total_zero_pc_c2} "
                              f"cam3={total_zero_pc_c3}){_s2}", flush=True)

                    if isinstance(obs, dict):
                        obs = obs["obs"]

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user, saving collected data...", flush=True)
        except Exception as e:
            import traceback
            print(f"\n[ERROR] Exception in main loop: {e}")
            traceback.print_exc()
            print(f"[DEBUG] Last step count: {total_steps}, collected: {total_collected}")

        elapsed = max(time.time() - start_time, 1e-6)
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        gpu_mem_gb = (torch.cuda.max_memory_allocated(args.device) / 1e9
                      if torch.cuda.is_available() else 0)
        per_var = " ".join(f"v{vid}={n}" for vid, n in collected_per_variant.items())
        print(f"\nDone: collected={total_collected} episodes ({per_var}), steps={total_steps}, "
              f"dropped_abnormal={n_dropped_abnormal}, "
              f"zero_pc={total_zero_pc}(cam1={total_zero_pc_c1} cam2={total_zero_pc_c2} "
              f"cam3={total_zero_pc_c3}), peak_gpu={gpu_mem_gb:.1f}GB, time={h}h{m}m{s}s", flush=True)
        if total_zero_pc_c1 > 0:
            print(f"[NOTE] cam1 (fixed camera) had {total_zero_pc_c1} empty frames -- the fixed camera sees the table and should not be empty,"
                  f"check whether cam1's pose/crop workspace crops the table out of view (suspicious)."
                  f"cam2 (wrist camera) empty {total_zero_pc_c2} times is mostly normal (hand far from drill / beyond far_clip)", flush=True)
        else:
            print(f"[NOTE] zero_pc all from cam2 (wrist camera={total_zero_pc_c2}), cam1=0 --"
                  f"these are normal empty frames from far_clip / the wrist camera not seeing near objects, not a stats bug;"
                  f"these frames are already filtered by _bad and do not enter the dataset", flush=True)
        print(f"Output: {args.output}", flush=True)

        if args.save_init_poses and len(flushed_init_poses) > 0:
            _save_init_poses_now()
            print(f"[INFO] Saved {len(flushed_init_poses)} episode init-poses -> {_pose_file}", flush=True)

        _pose_dir = args.output.rstrip("/") + "_init_poses"
        os.makedirs(_pose_dir, exist_ok=True)
        _ev, _ep, _eq, _eg, _es = [], [], [], [], []
        for _v, _st in _sobol_fn.state.items():
            _n = _st["n"]
            _ev.append(np.full(_n, int(_v), dtype=np.int64))
            _ep.append(_st["pos"].numpy().astype(np.float32))
            _eq.append(_st["quat"].numpy().astype(np.float32))
            _eg.append(_st["goal"].numpy().astype(np.float32))
            _es.append(_st["success"])
        _exam_file = os.path.join(_pose_dir, "exam_proposals.npz")
        np.savez(_exam_file,
                 variant=np.concatenate(_ev), pos_local=np.concatenate(_ep),
                 quat=np.concatenate(_eq), goal_quat=np.concatenate(_eg),
                 success=np.concatenate(_es))
        _sc = np.concatenate(_es)
        _run = int((_sc >= 0).sum())
        print(f"[EXAM] exam saved -> {_exam_file} | {len(_sc)} items total: "
              f"success {(_sc == 1).sum()} fail {(_sc == 0).sum()} not-run {(_sc == -1).sum()} "
              f"| pass rate {int((_sc == 1).sum()) / max(_run, 1):.1%}", flush=True)

        env.close()
    except Exception:
        _exit_code = 1
        import traceback
        print("\n[FATAL] the real exception is here:", flush=True)
        traceback.print_exc()
    finally:
        faulthandler.cancel_dump_traceback_later()
        os._exit(_exit_code)


if __name__ == "__main__":
    main()
