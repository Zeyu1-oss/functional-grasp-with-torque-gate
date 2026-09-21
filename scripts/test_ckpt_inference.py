#!/usr/bin/env python3
"""
Test DP3 checkpoint inference with dummy inputs.

This verifies that the checkpoint can be loaded and run without hanging.
Run with:
    cd /home/zeyu/3D-Diffusion-Policy/3D-Diffusion-Policy
    python /home/zeyu/inspire_drill/test_ckpt_inference.py

Or:
    cd /home/zeyu/inspire_drill
    PYTHONPATH="/home/zeyu/3D-Diffusion-Policy:/home/zeyu/3D-Diffusion-Policy/3D-Diffusion-Policy:$PYTHONPATH" \
        python test_ckpt_inference.py
"""

import sys
import os
import time

# DP3 root
DP3_ROOT = "/home/zeyu/3D-Diffusion-Policy/3D-Diffusion-Policy"
sys.path.insert(0, DP3_ROOT)

import torch
import dill
import zarr
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)


def compute_normalizer_from_zarr(data_path: str, device: str):
    """Compute normalizer from zarr dataset (matches training-time normalizer)."""
    from diffusion_policy_3d.model.common.normalizer import LinearNormalizer

    z = zarr.open_group(data_path)
    state_data = z['data']['state'][:]
    action_data = z['data']['action'][:]
    pc_data = z['data']['point_cloud'][:]

    data = {
        'action': action_data,
        'agent_pos': state_data,
        'point_cloud': pc_data,
    }

    normalizer = LinearNormalizer()
    normalizer.fit(data=data, last_n_dims=1, mode='limits')
    print(f"    [Normalizer] Computed from zarr: {data_path}")
    print(f"      action  range: {action_data.min():.3f} ~ {action_data.max():.3f}")
    print(f"      agent_pos range: {state_data.min():.3f} ~ {state_data.max():.3f}")
    print(f"      point_cloud range: {pc_data.min():.3f} ~ {pc_data.max():.3f}")
    return normalizer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
        default="/home/zeyu/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/inspire_drill-simple_dp3-simple_dp3_seed0/checkpoints/latest.ckpt")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print("=" * 60)
    print("DP3 Checkpoint Inference Test")
    print("=" * 60)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Device: {args.device}")
    print()

    # ---- 1. Load checkpoint ----
    print("[1] Loading checkpoint...")
    t0 = time.time()
    # Load onto CPU first; map_location will be re-applied during weight loading
    payload = torch.load(args.ckpt, pickle_module=dill, map_location="cpu")
    print(f"    Loaded in {time.time()-t0:.1f}s")

    cfg = payload["cfg"]
    print(f"    horizon={cfg.horizon}, n_obs_steps={cfg.n_obs_steps}, "
          f"n_action_steps={cfg.n_action_steps}")

    # ---- 2. Inspect shape_meta ----
    sm = OmegaConf.to_container(cfg.shape_meta, resolve=True)
    print(f"    shape_meta:")
    for k, v in sm["obs"].items():
        print(f"      {k}: {v['shape']} ({v['type']})")
    print(f"    action: {sm['action']['shape']}")

    pc_shape = sm["obs"]["point_cloud"]["shape"]   # e.g. [2048, 3]
    pos_shape = sm["obs"]["agent_pos"]["shape"]    # e.g. [26]
    B = 2
    n_obs = cfg.n_obs_steps                       # e.g. 2
    num_pc = pc_shape[0]                          # e.g. 2048

    # ---- 3. Instantiate policy ----
    print(f"\n[2] Instantiating policy (B={B}, n_obs={n_obs}, num_pc={num_pc})...")
    t0 = time.time()
    dp3_policy: torch.nn.Module = hydra_instantiate(cfg.policy)
    print(f"    Policy created in {time.time()-t0:.1f}s")

    # ---- 3. Load weights (prefer EMA) ----
    print(f"\n[3] Loading model weights...")
    use_ema = "ema_model" in payload["state_dicts"]
    sd_key = "ema_model" if use_ema else "model"
    state_dict = payload["state_dicts"][sd_key]

    # Separate normalizer from model keys (both were flattened into one dict)
    normalizer_sd = {}
    model_sd_clean = {}
    for k, v in state_dict.items():
        if k.startswith("normalizer."):
            normalizer_sd[k[len("normalizer."):]] = v
        else:
            model_sd_clean[k] = v

    # checkpoint was saved from CPU; load weights first, then move entire model to GPU
    dp3_policy.eval()
    dp3_policy.load_state_dict(model_sd_clean, strict=False)
    dp3_policy.to(args.device)

    if use_ema:
        print(f"    Loaded EMA model weights ({len(model_sd_clean)} keys, all on {args.device})")
    else:
        print(f"    Loaded base model weights ({len(model_sd_clean)} keys, all on {args.device})")

    # ---- 5. Set normalizer ----
    print(f"\n[4] Setting normalizer...")
    if normalizer_sd:
        from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
        tmp_normalizer = LinearNormalizer()
        tmp_normalizer.load_state_dict(normalizer_sd)
        # Move normalizer tensors to GPU (load_state_dict doesn't do this)
        _device = torch.device(args.device)
        for key in tmp_normalizer.params_dict:
            for subkey in tmp_normalizer.params_dict[key]:
                v = tmp_normalizer.params_dict[key][subkey]
                if isinstance(v, torch.Tensor) and v.device != _device:
                    tmp_normalizer.params_dict[key][subkey] = v.to(_device)
        dp3_policy.set_normalizer(tmp_normalizer)
        print(f"    Loaded normalizer from state_dict ({len(normalizer_sd)} keys, moved to {args.device})")
    else:
        # Fallback: compute from dataset
        print("    No normalizer in checkpoint, computing from dataset...")
        normalizer = compute_normalizer_from_zarr(
            "/home/zeyu/inspire_drill/data/inspire_drill_dp3.zarr", args.device)
        dp3_policy.set_normalizer(normalizer)
        print("    Computed normalizer from dataset")

    # ---- 5. Load real data from zarr ----
    print(f"\n[5] Loading real data from zarr...")
    zarr_path = "/home/zeyu/inspire_drill/data/inspire_drill_dp3.zarr"
    z = zarr.open_group(zarr_path, mode='r')

    episode_ends = z['meta/episode_ends'][:]   # (n_episodes,)
    starts = [0] + list(episode_ends[:-1])
    episode_lengths = episode_ends - starts
    n_episodes = len(episode_ends)
    n_total_frames = episode_ends[-1]

    # Load ALL frames into memory (298k frames is manageable)
    print(f"    Loading {n_total_frames} frames from {n_episodes} episodes...")
    pc_data = z['data/point_cloud'][:n_total_frames]       # (N, 2048, 3)
    pos_data = z['data/state'][:n_total_frames]             # (N, 26)
    act_data = z['data/action'][:n_total_frames]            # (N, 13)
    print(f"    point_cloud shape: {pc_data.shape}")
    print(f"    agent_pos shape:   {pos_data.shape}")
    print(f"    action shape:      {act_data.shape}")
    print(f"    pc range:          [{pc_data.min():.3f}, {pc_data.max():.3f}]")
    print(f"    pos range:         [{pos_data.min():.3f}, {pos_data.max():.3f}]")
    print(f"    act range:         [{act_data.min():.3f}, {act_data.max():.3f}]")
    print(f"    Episode lengths: min={episode_lengths.min()}, max={episode_lengths.max()}, mean={episode_lengths.mean():.1f}")

    # Pick the first episode
    ep_idx = 0
    ep_start = starts[ep_idx]
    ep_end   = episode_ends[ep_idx]
    ep_len   = episode_lengths[ep_idx]
    print(f"\n    Using episode {ep_idx}: frames {ep_start}-{ep_end} (length={ep_len})")

    # ---- 6. Prepare initial observation window ----
    # shape: (B=1, n_obs, num_points, 3) and (B=1, n_obs, 26)
    B = 1
    window_pc  = torch.from_numpy(pc_data[ep_start : ep_start + n_obs]).float().to(args.device)  # (n_obs, 2048, 3)
    window_pos = torch.from_numpy(pos_data[ep_start : ep_start + n_obs]).float().to(args.device)  # (n_obs, 26)
    window_pc  = window_pc.unsqueeze(0)    # (1, n_obs, 2048, 3)
    window_pos = window_pos.unsqueeze(0)   # (1, n_obs, 26)

    print(f"    Initial window: pc={window_pc.shape}, pos={window_pos.shape}")

    # ---- 7. Run inference on all frames of episode ----
    print(f"\n[6] Running inference on episode {ep_idx}...")
    print(f"    n_obs={n_obs}, n_action_steps={dp3_policy.n_action_steps}, horizon={dp3_policy.horizon}")
    print(f"    num_inference_steps: {dp3_policy.num_inference_steps}")

    pred_actions = []
    gt_actions = []
    total_steps = ep_len - n_obs
    global_step = 0   # frame index relative to episode start

    t_start = time.time()
    with torch.inference_mode():
        for step in range(total_steps):
            obs = {
                "point_cloud": window_pc,
                "agent_pos": window_pos,
            }
            result = dp3_policy.predict_action(obs)
            action = result["action"].cpu().numpy()[0]  # (n_action_steps, 13)

            # Ground truth action aligned with the last observation frame
            gt_a = act_data[ep_start + global_step + n_obs - 1]  # (13,)

            pred_actions.append(action[0])
            gt_actions.append(gt_a)

            if step % 20 == 0 or step == total_steps - 1:
                mse = ((action[0] - gt_a) ** 2).mean()
                print(f"    step {step:3d}/{total_steps-1} (frame {global_step:3d}): "
                      f"pred=[{action[0].min():.3f}, {action[0].max():.3f}] "
                      f"GT=[{gt_a.min():.3f}, {gt_a.max():.3f}] MSE={mse:.4f}")

            # Advance observation window by n_action_steps frames
            for _ in range(dp3_policy.n_action_steps):
                global_step += 1
                next_frame = ep_start + global_step + n_obs - 1
                if next_frame >= ep_end:
                    break
                new_pc  = torch.from_numpy(pc_data[next_frame]).float().to(args.device).unsqueeze(0).unsqueeze(0)  # (1, 1, 2048, 3)
                new_pos = torch.from_numpy(pos_data[next_frame]).float().to(args.device).unsqueeze(0).unsqueeze(0)  # (1, 1, 26)
                window_pc  = torch.cat([window_pc[:, 1:],  new_pc],  dim=1)
                window_pos = torch.cat([window_pos[:, 1:], new_pos], dim=1)

    elapsed = time.time() - t_start
    print(f"\n    Rollout completed in {elapsed:.2f}s ({elapsed/total_steps*1000:.1f}ms/step)")

    # Compute overall statistics
    pred_arr = np.stack(pred_actions)
    gt_arr = np.stack(gt_actions)
    mse = ((pred_arr - gt_arr) ** 2).mean()
    mae = np.abs(pred_arr - gt_arr).mean()
    print(f"\n    Overall MSE: {mse:.4f}")
    print(f"    Overall MAE: {mae:.4f}")
    print(f"    Pred range: [{pred_arr.min():.4f}, {pred_arr.max():.4f}]")
    print(f"    GT   range: [{gt_arr.min():.4f}, {gt_arr.max():.4f}]")

    print("\n" + "=" * 60)
    print("ROLL-OUT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import numpy as np
    main()
