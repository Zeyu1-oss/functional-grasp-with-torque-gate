# 3D Diffusion Policy — Contact-Gated Torque for Functional Grasping

Fork of [**3D Diffusion Policy**](https://github.com/YanjieZe/3D-Diffusion-Policy) (DP3, RSS 2024), adapted to
train the visuomotor **student** for the functional-grasping project: a Franka arm + Inspire five-finger hand,
distilled from a privileged PPO teacher, studying how applied joint torque should enter the student when it
is only informative during hand–tool contact.

This branch is one step of a larger pipeline. The rest is the
[**`main` branch**](https://github.com/Zeyu1-oss/functional-grasp-with-torque-gate/tree/main) of this same
repository (the Isaac Lab side): training data comes from its `collect_dp3_data.py`, and checkpoints trained
here are deployed and graded by its `deploy_dp3_sim.py`. **Read that README first** — it covers the full
pipeline; this one only records what changed on the DP3 side.

---

## Installation

Dependencies are unchanged from upstream — follow [INSTALL.md](INSTALL.md). Developed against Python 3.8,
torch 2.4.1 (cu124), diffusers 0.36, zarr 2.16, hydra 1.3.2.

Two things the training scripts hardcode and that you will have to adjust:

- **The interpreter.** `scripts/train_policy_inspire_drill_grasp_norobot_eq1.sh` invokes
  `/home/zeyu/anaconda3/envs/dp3/bin/python` and exports a `PYTHONPATH` rooted at `$HOME/3D-Diffusion-Policy`.
  Clone to `~/3D-Diffusion-Policy` and edit the interpreter path, or edit both.
- **Nesting.** The Python package is one level down, in `3D-Diffusion-Policy/diffusion_policy_3d/` (upstream's
  layout). Source paths below are relative to it; scripts are run from the repo root.

---

## Training

One script produces the reported checkpoint — contact gate on, auxiliary torque objective on:

```bash
# from the repo root (the script guards on it)
bash scripts/train_policy_inspire_drill_grasp_norobot_eq1_auxtorque.sh <zarr_path> [seed] [gpu_id] [epochs] [gamma]
```

`<zarr_path>` is the dataset `collect_dp3_data.py` wrote; it must have been collected with `--save_contact`,
since the gate is supervised from it. Everything else defaults to the reported configuration —
`SCOPE=global`, `GATE_POS=feature`, `AUX_BETA=0.1` — each overridable by environment variable
(`AUX_BETA=0.05 bash scripts/...`). Checkpoints land in
`data/outputs/..._eq1_global_feature_auxon_seed0/checkpoints/`; the reported result is epoch 180.

The other `train_policy_inspire_drill_grasp_norobot_*.sh` scripts are the ablation cells. Each `exec`s the
same base script with one switch flipped rather than copying it, so the pipeline cannot drift between cells.

---

## What's different from upstream DP3

Benchmark envs (Adroit / DexArt / MetaWorld) are replaced by a single `InspireDrillEnv`;
`dataset/inspire_drill_dataset.py` reads the zarr `collect_dp3_data.py` writes; `env_runner/` is a stub,
since evaluation happens externally in Isaac Lab.

The contribution itself is in `model/vision/pointnet_extractor.py` and `policy/simple_dp3.py`, and is **off
by default** — at default flags the policy is byte-identical to vanilla SimpleDP3.

- **`state_split`** splits `agent_pos` into a position branch (13-d) and a force branch (13-d torque),
  instead of one flat MLP over the whole vector.
- **`contact_gate`** modulates the force branch by a learned `phi = sigmoid(psi(tau))`, supervised by BCE
  against ground-truth hand–tool contact. `scope` = `global` (one gate) or `per_finger` (6 gates, arm torque
  dropped); `gate_position` = `input` (gates raw torque) or `feature` (gates the encoder's output, which
  requires `scope=global`); `soft_label` grades the target by contact force instead of a 0/1 threshold.
- **Auxiliary torque prediction** (`AUX=on`) widens the UNet trajectory channel 13→26 (`[action ; torque]`),
  jointly denoising the action chunk and a torque target shifted one step to match `applied_torque`'s timing
  (`L = L_action + beta * L_torque`). Inference is unaffected — `predict_action` reads back only the first 13
  channels, so deploy commands are identical either way.

Plus two robustness fixes: `TopKCheckpointManager` no longer crashes on epochs without `val_loss`, and
`train.py` gained `resume_from_checkpoint`.

---

## Inspecting data and checkpoints

```bash
python 3D-Diffusion-Policy/diffusion_policy_3d/env/inspire_drill/viz.py --data <zarr_path> --episode 0
```

Interactive Plotly viewer for a collected zarr, colouring points by segment (`cam1`/`cam2`/`plate`/`robot`/
`drill`/`ground`, toggleable in the legend) or by the handle mask. Worth running before training: a
mis-specified collect flag shows up immediately as a missing segment. `--frame` gives a single frame instead
of an animation, `--only cam1` isolates a segment.

> It writes HTML to a **hardcoded** `/home/zeyu/inspire_drill/data/` path — edit `html_path` near the bottom
> of the file before running it elsewhere.

`scripts/viz.py` is a static alternative (matplotlib → PNG, explicit `--out`).
`scripts/test_ckpt_inference.py` runs one dummy forward pass on a checkpoint — faster than learning the same
thing from `deploy_dp3_sim.py`, which has to start Isaac Sim first.

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
