# 3D Diffusion Policy — Contact-Gated Torque for Functional Grasping

Fork of [**3D Diffusion Policy**](https://github.com/YanjieZe/3D-Diffusion-Policy) (DP3, RSS 2024), adapted to
train the visuomotor **student** policy for the `inspire_drill` functional-grasping project: a Franka arm +
Inspire five-finger hand, distilled from a privileged PPO teacher, studying how applied joint torque should
enter the student when it is only informative during hand–tool contact.

This branch is not meant to be read standalone — it is **one step** of a larger pipeline, and the rest of it
lives on the [**`main` branch**](https://github.com/Zeyu1-oss/functional-grasp-with-torque-gate/tree/main) of
this same repository (the Isaac Lab side). Training data comes from there via
`scripts/collect_dp3_data.py`, and checkpoints trained here are deployed and graded back there via
`scripts/deploy_dp3_sim.py`. **Read the `main` branch's README first** — it covers the full pipeline, and
this document only records what changed on the DP3 side.

The two branches are checked out into separate directories, since they need separate Python environments:

```
<workspace>/
├── functional-grasp-with-torque-gate/   the main branch — Isaac Lab (Python 3.11)
└── 3D-Diffusion-Policy/                 this branch     — DP3 student training (Python 3.8)
```

---

## Installation

Dependencies are unchanged from upstream — follow [INSTALL.md](INSTALL.md) (or the original
[3D Diffusion Policy README](https://github.com/YanjieZe/3D-Diffusion-Policy)). Developed against Python 3.8,
torch 2.4.1 (cu124), diffusers 0.36, zarr 2.16, hydra 1.3.2.

Two things the training scripts currently hardcode, and that you will have to adjust:

- **The interpreter.** `scripts/train_policy_inspire_drill_grasp_norobot_eq1.sh` invokes
  `/home/zeyu/anaconda3/envs/dp3/bin/python` and exports a `PYTHONPATH` rooted at `$HOME/3D-Diffusion-Policy`.
  Either clone this repo to `~/3D-Diffusion-Policy` and edit that one interpreter path to point at your own
  conda env, or edit both.
- **Repository nesting.** The Python package lives one level down, in `3D-Diffusion-Policy/diffusion_policy_3d/`
  (inherited from upstream's layout). Every source path below is relative to that inner directory; the
  `scripts/` and `train.py` entry points are found from the repo root, which is where the scripts must be run
  from. Checkpoints land in `3D-Diffusion-Policy/data/outputs/<run>/checkpoints/`.

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

### Training

One script produces the reported checkpoint — contact gate on, auxiliary torque objective on:

```bash
# from the repo root (the script guards on it)
bash scripts/train_policy_inspire_drill_grasp_norobot_eq1_auxtorque.sh <zarr_path> [seed] [gpu_id] [epochs] [gamma]
```

`<zarr_path>` is the dataset `collect_dp3_data.py` wrote on the Isaac Lab side; it must have been collected
with `--save_contact`, since the gate is supervised from it. Everything else defaults to the reported
configuration — `SCOPE=global`, `GATE_POS=feature`, `AUX_BETA=0.1`, 1000 epochs — and each is an environment
variable you can override:

```bash
AUX_BETA=0.05 bash scripts/train_policy_inspire_drill_grasp_norobot_eq1_auxtorque.sh <zarr_path>
```

The run directory encodes the configuration, so ablation cells never collide:
`data/outputs/..._eq1_global_feature_auxon_seed0/checkpoints/`. The reported result is epoch 180.

The other `train_policy_inspire_drill_grasp_norobot_*.sh` scripts are the ablation cells (baseline, torque as
observation or objective only, gate scope/position, hard vs. soft label, …). Each `exec`s the same base
script with one switch flipped rather than copying it, so the data pipeline and validation logic cannot drift
between cells — they are not separate code paths, and they are not part of the pipeline above.

### Visualizing data

Before training on a freshly collected zarr, it is worth looking at it — a mis-specified collect flag shows
up immediately as a missing or misplaced point-cloud segment.

```bash
python 3D-Diffusion-Policy/diffusion_policy_3d/env/inspire_drill/viz.py --data <zarr_path> --episode 0
python 3D-Diffusion-Policy/diffusion_policy_3d/env/inspire_drill/viz.py --data <zarr_path> --only cam1 --frame -1
```

This is the tool actually used to inspect collected data: an interactive Plotly viewer that colours points by
segment (`cam1`/`cam2`/`plate`/`robot`/`drill`/`ground`, each toggleable in the legend), or by the
ground-truth handle mask when the zarr carries one. Without `--frame` it animates the whole episode; with it
you get a single frame (negative indices count from the end). It also runs a wrist-camera extrinsics check,
which catches a frozen-calibration bug present in some older datasets.

> It writes its HTML output to a **hardcoded** `/home/zeyu/inspire_drill/data/` path — edit `html_path` near
> the bottom of the file before running it elsewhere.

`scripts/viz.py` is a lighter static alternative: matplotlib scatter → PNG, no Plotly or browser needed, and
it takes an explicit `--out`. Between them they replace upstream's `visualizer/` (a Flask+Plotly web app),
which is deleted here — nothing in this project referenced it.

### Checking a checkpoint

`scripts/test_ckpt_inference.py` loads a checkpoint and runs one dummy forward pass. Worth doing before
handing a checkpoint to `deploy_dp3_sim.py`, which spins up Isaac Sim and takes considerably longer to tell
you the same thing.

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
