# 3D Diffusion Policy — Contact-Gated Torque for Functional Grasping

Fork of [**3D Diffusion Policy**](https://github.com/YanjieZe/3D-Diffusion-Policy) (DP3, RSS 2024), adapted to
train the visuomotor **student** policy for the `inspire_drill` functional-grasping project: a Franka arm +
Inspire five-finger hand, distilled from a privileged PPO teacher, studying how applied joint torque should
enter the student when it is only informative during hand–tool contact.

This repo is not meant to be read standalone. Task configs assume zarr data produced by
`inspire_drill/scripts/collect_dp3_data.py`, and checkpoints trained here are deployed back into Isaac Lab via
`inspire_drill/scripts/deploy_dp3_sim.py`. **See the `inspire_drill` repo's README for the full pipeline** —
this document only covers what changed on the DP3 side.

---

## Installation

Unchanged from upstream — follow [INSTALL.md](INSTALL.md) (or the original
[3D Diffusion Policy README](https://github.com/YanjieZe/3D-Diffusion-Policy)). Developed against Python 3.8,
torch 2.4.1 (cu124), diffusers 0.36, zarr 2.16, hydra 1.3.2.

---

## What's different from upstream DP3

### Task

- Adroit / DexArt / MetaWorld benchmark envs removed (`env/__init__.py`); replaced with a single
  `InspireDrillEnv` under `env/inspire_drill/`.
- `dataset/inspire_drill_dataset.py` (`InspireDrillDataset`, extends `RealDexDataset`): loads the zarr layout
  `collect_dp3_data.py` writes — `state` (13), `action` (13), `point_cloud` (fused camera + robot-FK + ground),
  and optionally `contact` (13-d per-sensor force) and a torque-augmented `state` (26).
- `env_runner/` is a `DummyRunner` — this project doesn't run closed-loop eval inside the training process;
  evaluation happens externally in Isaac Lab (`deploy_dp3_sim.py`), so the runner is a no-op stub.
- `config/task/`: one yaml per data/ablation variant (`inspire_drill_grasp*.yaml`, `drill.yaml`,
  `inspire_drill_chained.yaml`, `inspire_drill_stage2_force.yaml`, ...) — differ mainly in point-cloud
  composition (camera-only vs +robot FK), `agent_pos` width (13 vs 26 with torque), and whether `data/contact`
  is loaded.

### Architecture: contact-gated torque branch

The actual contribution, in `model/vision/pointnet_extractor.py` and `policy/simple_dp3.py` — **off by
default**: with every new config flag at its default, the policy is byte-identical to vanilla SimpleDP3.

- **`state_split`**: splits `agent_pos` into a position branch (13-d, `pos_mlp_size`) and an optional force
  branch (torque dims, `force_mlp_size`), instead of one flat MLP over the whole vector.
- **`contact_gate`** (the Eq.1 mechanism from arXiv:2604.01414, adapted): a learned gate
  `phi = sigmoid(psi(tau))` modulates the torque branch, supervised by a BCE loss against ground-truth
  hand–tool contact (`data/contact`, requires the zarr to have been collected with `--save_contact`).
  - `scope`: `per_finger` (6 independent gates, arm torque dropped from the force branch entirely) vs
    `global` (1 gate over all 13 torque dims — the literal Eq.1 baseline).
  - `gate_position`: `input` (gates the raw torque before the encoder sees it) vs `feature` (encoder always
    sees the real torque, gate multiplies its output feature instead — necessarily `scope=global`).
  - `soft_label`: grade the BCE target by contact force (log-scaled) instead of a hard 0/1 threshold.
- **Auxiliary torque prediction** (`AUX=on`): the diffusion UNet's trajectory channel widens 13→26
  (`[action ; torque]`), jointly denoising the action chunk and a torque target shifted one step to match
  `applied_torque`'s actual timing (the TA-VLA-style objective, `L = L_action + beta * L_torque`). Inference
  is unaffected — `predict_action` only reads back the first 13 channels, so a deploy command is identical
  whether or not AUX was used at train time.
- Two small robustness fixes: `checkpoint_util.TopKCheckpointManager` no longer crashes on an epoch with no
  `val_loss` (skipped-validation epochs); `train.py` gained `resume_from_checkpoint` /
  `_resume_ckpt_candidates` for resuming a crashed run.

### Ablation grid (`scripts/train_policy_inspire_drill_grasp_norobot_*.sh`)

One script per cell. Every variant `exec`s the same base script (`_eq1.sh` / `_baseline.sh`) rather than
copy-pasting it, so the data pipeline, validation, and checkpoint-naming logic can't silently drift between
cells — only the ablated switch differs.

| Script | obs | aux target | gate |
|---|:--:|:--:|:--:|
| `..._baseline.sh` | 13 | off | off |
| `..._torqueobs.sh` | 26 | off | off |
| `..._torqueobj.sh` | 13 | on | off |
| `..._torqueboth.sh` | 26 | on | off |
| `..._contactgate.sh` | 26 | on | on |
| `..._eq1.sh` / `..._eq1_auxtorque.sh` | 26 | off / on | on (input-gated) |
| `..._gatehard_pnorm.sh` / `..._gatesoft_pnorm.sh` | 26 | on | on (hard vs soft BCE label) |
| `..._gateglobal.sh` / `..._gatefeat_pnorm.sh` | 26 | on | on (`scope=global` / `gate_position=feature`) |
| `..._maskgate_pnorm.sh` | 26 | on | on (unsupervised gate ablation) |
| `..._handonly.sh` / `..._handpc1280.sh` / `..._cam1_2048_force*.sh` | — | — | point-cloud composition variants (hand-only robot segment, denser robot cloud, camera-only) |

```bash
bash scripts/train_policy_inspire_drill_grasp_norobot_eq1_auxtorque.sh <zarr_path> [seed] [gpu_id] [epochs] [gamma]
AUX_BETA=0.05 bash scripts/train_policy_inspire_drill_grasp_norobot_eq1_auxtorque.sh   # override the aux loss weight
```

Must be run from the repo root (the base scripts guard on it).

### Other scripts

- `scripts/viz.py` — zarr point-cloud viewer (matplotlib scatter, one or several frames). Needs only
  `zarr`+`numpy`+`matplotlib`, no IsaacLab/torch, so it runs in a plain Python env. Replaces the deleted
  `visualizer/` (Flask+Plotly, unused).
- `scripts/test_ckpt_inference.py` — smoke-test: loads a checkpoint and runs one dummy forward pass, to check
  it doesn't hang/crash before committing to a full eval.

### Removed

- `visualizer/` — the Flask+Plotly point-cloud web visualizer. Deleted; nothing in this project referenced it.

---

## License / Citation

Unchanged — see [LICENSE](LICENSE). If you use DP3 itself, cite the original work:

```
@inproceedings{Ze2024DP3,
	title={3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations},
	author={Yanjie Ze and Gu Zhang and Kangning Zhang and Chenyuan Hu and Muhan Wang and Huazhe Xu},
	booktitle={Proceedings of Robotics: Science and Systems (RSS)},
	year={2024}
}
```
