# Learning-based Functional Grasping for Dexterous Hands

**Master's thesis · Technical University of Munich (TUM)**<br>
Supervisors: Qian Feng, Zitao Zhang

Isaac Lab environments, data pipeline and training/evaluation code for functional grasping of a
power drill with a 7-DoF Franka arm and a 6-DoF Inspire five-finger hand.

<p align="center">
  <img src="docs/img/setup.png" width="49%" alt="Isaac Lab scene: Franka arm with Inspire hand and a power drill">
  <img src="docs/img/functional_grasp.png" width="35%" alt="A functional grasp: handle enclosed, index finger at the trigger">
</p>

A *functional* grasp does not merely immobilise the tool, it acquires it in the configuration from
which the tool can be operated: the handle enclosed, the index finger on the trigger. A privileged
PPO teacher is trained on simulator state, then distilled into a
[3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) student that sees only a
depth-camera point cloud and joint proprioception. The methodological question the project studies
is how joint torque should enter that student — and the answer is that it has to be **gated on
contact** before it helps at all.

The gate follows [FoAR](https://arxiv.org/abs/2411.15753); what differs here is its supervision.
A simulator reports contact on each of the 13 hand links separately, so the label is exact and its
threshold is task-independent — where a gate supervised from a measured force/torque trace needs
that threshold retuned per task. The labels enter the loss only and are never a network input, so
the deployed policy still consumes exactly what a real robot can measure.

---

## Installation

This repo covers the full pipeline **up to student training** in a single **Isaac Lab**
environment: both PPO teacher stages, interactive play/visualization, DP3 data collection, and —
once a DP3 checkpoint exists — deployment. Training the DP3 student itself is **not** done here:
it happens on the separate [`dp3` branch](https://github.com/Zeyu1-oss/functional-grasp-with-torque-gate/tree/dp3)
of the forked 3D-Diffusion-Policy repo, in its own environment (its dependencies conflict with
Isaac Sim's) — see [DP3 student training](#dp3-student-training-separate-branch-separate-env)
below.

The two repos are expected to sit side by side — the commands in Usage below assume this layout:

```
<workspace>/
├── drill_sim2real/        this repo — Isaac Lab side (teachers, data collection, deploy)
└── 3D-Diffusion-Policy/   the DP3 fork, `dp3` branch — student training only
```

1. Isaac Sim + Isaac Lab (developed against Python 3.11 / isaaclab 0.53.1): follow the official
   [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

```bash
# 2. This repo, plus the extra packages in that same env — covers everything here, deploy included
git clone https://github.com/Zeyu1-oss/drill_sim2real.git && cd drill_sim2real
pip install rl_games==1.6.1 zarr numcodecs dill omegaconf trimesh
```

USD assets go in `assets/` (gitignored). Download and unpack them:

```bash
pip install gdown
gdown --fuzzy 'https://drive.google.com/file/d/1PdrOZZjNwIF0OrMTwvL6x9sda6tw_FAN/view?usp=drive_link' -O assets.zip
unzip assets.zip -d assets/
```

Then build the canonical robot cloud once:

```bash
python tools/build_robot_pointcloud.py     # -> assets/inspire_tac/robot_canonical_points.npz
```

`data/`, `collected_data/`, `runs/` and `output/` are gitignored.

### DP3 student training (separate branch, separate env)

Only needed for step 4 of the pipeline below (training the diffusion-policy student) — nothing
else in this repo depends on it. Clone the fork's `dp3` branch *next to* this repo and follow its
own install:

```bash
cd ..    # back to <workspace>, so the two repos end up siblings
git clone -b dp3 https://github.com/Zeyu1-oss/functional-grasp-with-torque-gate.git 3D-Diffusion-Policy
#    -> see that branch's README for what's added on top of upstream DP3, INSTALL.md for setup.
#    Developed against: Python 3.8 · torch 2.4.1 (cu124) · diffusers 0.36 · zarr 2.16 · hydra 1.3.2
```

---

## Repository Layout

```
tasks/          Isaac Lab environments: envs, MDP reward/termination terms, hyperparameters
scripts/        Entry points — train / collect / deploy (the pipeline below)
perception/     Point-cloud & observation code shared by collect and deploy
tools/          Asset prep, result analysis, plotting, slide decks — outside the repro path
config/         Scene, drill-variant, and RL-games/DP3 agent YAMLs
results/        Plots and tables assembled from runs/, for the thesis and paper
assets/ data/ collected_data/ runs/ output/   Generated — gitignored, not checked in
```

<details>
<summary><b>tasks/</b> — environments</summary>

| File | Role |
|---|---|
| `grasp_drill_env.py` | Stage-1 core env (`GraspDrillEnv`). Scene/robot/drill spawn, 75-d observation, action processing, contact sensors. Everything else in `tasks/` builds on this. |
| `stage2_env.py` | `Stage2Env(GraspDrillEnv)` — alignment stage; resets from banked stage-1 success end-states, adds the align reward/termination. Used by `train2.py`, `collect_dp3_data.py`, `deploy_dp3_sim.py`, `play_stage2.py`. |
| `chained_env.py` | `ChainedEnv(Stage2Env)` — runs stage 1 → stage 2 back-to-back in one episode, no pkl dependency. Used for full-pipeline collection/deploy (`play_chained.py`). |
| `stage2_grasp_drill_env.py` | An earlier stage-2 env variant, superseded by `stage2_env.py`; nothing currently imports it. |
| `config/config.py` | Central hyperparameter dataclasses (`DEFAULT_HYPERPARAMETERS`); kept import-safe standalone since `perception/camera_setup.py` loads it before Isaac Lab is up. |
| `mdp/rewards.py`, `mdp/terminations.py` | Reward terms (approach, lift, trigger, penalties) and success/termination checks used by the env configs. |

</details>

<details>
<summary><b>perception/</b> — shared observation code</summary>

The observation is built by the *same* code on the collect and deploy sides (see the warning in Usage below) — this is where it lives.

| File | Role |
|---|---|
| `dp3_pointcloud.py` | Farthest-point-sampling kernel for the DP3 point cloud. |
| `robot_pointcloud.py` | FK-based robot point cloud from canonical link points (built once by `tools/build_robot_pointcloud.py`). |
| `groundtruth_mask.py` | Ground-truth drill-handle point mask (sim-only), the single shared implementation after a past collect/deploy threshold mismatch. |
| `handle_mask.py` | Per-point handle labels for training a segmentation net. |
| `handle_seg_net.py` | PointNet-style handle segmentation net intended to replace `groundtruth_mask.py` at real-robot deploy time; its trainer script isn't in the repo yet, so it's currently unused. |
| `student_obs.py` | Builds the DP3 student's `agent_pos` vector (13-d joint positions, or 26-d with joint torque). |
| `target_drive.py` | Shared joint-target drive (`_apply_action` patch) so collect/deploy/play dynamics match exactly. |
| `camera_setup.py` | Loads the perception hyperparameter block standalone, before Isaac Lab is imported. |

</details>

<details>
<summary><b>tools/</b> — everything outside the reproduction path</summary>

- **Asset prep**: `add_physics_apis.py`, `convert_usdz.py`, `normalize_asset.py`, `build_robot_pointcloud.py`
- **Eval poses**: `make_sobol_init_poses.py`
- **Gate / torque analysis** (consume `deploy_dp3_sim.py --dump_gate`/`--dump_torque`): `analyze_gate.py`, `analyze_gate_phases.py`, `analyze_light_contact.py`, `recompute_gate_stats.py`, `plot_gate_calibration.py`, `plot_torque_pred.py`, `plot_contact_torque_delta.py`
- **Training/eval curves**: `plot_training_curves.py`, `plot_success_curves.py`, `plot_ablation_curves.py`, `plot_beta_ablation.py`, `plot_sobol_coverage.py`
- **Result pipeline**: `sweep_eval.py` (runs `deploy_dp3_sim.py` over checkpoints → CSV), `sync_results_from_notes.py`
- **Defense slide decks**: `make_deck.py`, `make_deck_full.py`, `make_deck_defense.py`
- **Env shim**: `isaac_python.sh` — joins the conda env with Isaac Sim's kit bindings

</details>

---

## Usage

The teacher (PPO, privileged state) rolls out under Isaac Lab and produces the demonstrations the
student is distilled from offline, through the contact-gated encoder and diffusion head:

<p align="center"><img src="docs/img/pipeline.png" width="88%" alt="Teacher/student pipeline: PPO teacher rollout producing demonstrations (a), and the student's point-cloud/joint/torque encoders, contact gate, and diffusion head consuming them (b)"></p>

PPO teacher training is two stages — grasp, then align. Everything below runs in this repo's
Isaac Lab env, **except step 4**, which switches to the `dp3` branch's own env:

```bash
# 1. Stage-1 teacher: grasp (PPO)
python scripts/train_with_rl_games.py --headless --num_envs 4096

# optional: watch the trained stage-1 policy (play_stage2.py / play_chained.py are the
# stage-2 / full-pipeline equivalents, see the tasks/ table above)
python scripts/play_drill.py --checkpoint <path_to_checkpoint>

# 2. Bank successful grasp end-states, then train the stage-2 teacher: align (PPO)
python scripts/collect_success_data.py --headless --num_envs 4096 \
    --checkpoint <path_to_checkpoint> --output collected_data/success_data.pkl
python scripts/train2.py --headless --num_envs 4096 --dataset collected_data/success_data.pkl

# 3. Collect DP3 demonstrations (only successful episodes are written) — this repo's full
#    contribution to the DP3 side; everything needed to produce training data lives here
python scripts/collect_dp3_data.py --stage1_only --headless \
    --num_envs 256 --episodes_per_variant 1000 \
    --disable_cam2 --no_robot --force_state --save_contact \
    --stage1_checkpoint <path_to_checkpoint> --output data/norobot.zarr

# 4. Train the DP3 student — switch to the sibling repo / its own env (see Installation above)
cd ../3D-Diffusion-Policy && bash scripts/train_policy_inspire_drill_grasp_norobot_eq1_auxtorque.sh \
    ../drill_sim2real/data/norobot.zarr
cd ../drill_sim2real

# 5. Back in this repo's env: draw a fixed, policy-independent evaluation pose set, then deploy
#    the checkpoint step 4 produced and grade it
python tools/make_sobol_init_poses.py -n 100 -o data/eval_sobol_100.npz   # -n is per drill variant -> 300
python scripts/deploy_dp3_sim.py --stage1_only --headless --num_envs 70 \
    --disable_cam2 --no_robot --init_pose_file data/eval_sobol_100.npz \
    --dp3_ckpt ../3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/<run>/checkpoints/epoch_0180.ckpt
```

`--init_pose_file` replays a fixed pose set, each pose exactly once, so different checkpoints are
graded on identical initial conditions — and the poses are drawn independently of any policy
rather than reused from the teacher's successes, which would be a curated easy subset. Other
deploy modes: `--stage2_rl` (DP3 grasps, the privileged teacher aligns), `--stage2_dp3_ckpt` (both
stages distilled), `--dump_gate` (log the gate against ground-truth contact), `--policy rl` (the
teacher inside the student's harness).

> **Collect and deploy must agree.** The observation is built by the same `perception/` code on
> both sides, but its composition is chosen by flags (`--disable_cam2`, `--no_robot`,
> `--force_state`, …). A mismatch is silent — only the point-cloud *size* is checked against the
> checkpoint at startup.

---

## Results

All in simulation, on a fixed set of 300 unseen initial poses shared by every checkpoint. Success
requires envelopment *and* trigger reachability, held for 20 of the last 50 control steps.

| Policy | Observation | Success |
|---|---|---|
| Stage-1 teacher (grasp) | privileged state (75-d) | 93.0 % |
| Stage-2 teacher (align) | privileged state + plate pose | 92.0 % |
| Grasp **student** (ours) | point cloud + proprioception | **79.7 %** |

### Torque ablation

| Configuration | Obs. | Target | Gate | Success |
|---|:--:|:--:|:--:|--:|
| Baseline — point cloud + joint positions | | | | 70.3 % |
| Torque as observation | ✓ | | | 74.3 % |
| Torque as objective only | | ✓ | | 69.7 % |
| Observation + objective | ✓ | ✓ | | 73.3 % |
| Observation + **gate** | ✓ | | ✓ | 78.3 % |
| **Observation + gate + objective (ours)** | ✓ | ✓ | ✓ | **79.7 %** |

<p align="center"><img src="docs/img/ablation.png" width="86%" alt="Deployed grasp success rate versus DP3 training step, one curve per ablation condition"></p>

Gating is what carries the result: +9.3 points over the baseline (n = 300, single seed, p < 0.01),
of which the gate alone accounts for +8.0. The two smaller increments — ungated torque, and the
auxiliary objective on top of the gate — are within the resolution of a 300-episode evaluation.

### Generalisation to unseen drills

On drill geometries held out from training the hand still encloses the handle, but the *functional*
condition often fails: trigger offsets are specified per variant in the object frame, so they do
not transfer. Regressing them from the point cloud is the obvious next step.

<p align="center">
  <img src="docs/img/unseen_failure.png" width="38%" alt="Unseen drill: handle enclosed but the index finger misses the trigger">
  <img src="docs/img/stage2_alignment.png" width="52%" alt="Stage 2: the grasped drill aligned against the target plate">
</p>

---

## References

- Ze et al., **3D Diffusion Policy**, RSS 2024 — the student architecture
- He et al., **FoAR: Force-Aware Reactive Policy**, RA-L 2025 ([arXiv:2411.15753](https://arxiv.org/abs/2411.15753)) — the contact-gated branch adopted here
- Lei et al., **Learning When to See and When to Feel**, [arXiv:2604.01414](https://arxiv.org/abs/2604.01414) — suppressing torque in free motion carries most of the benefit
- Zhang et al., **TA-VLA**, [arXiv:2509.07962](https://arxiv.org/abs/2509.07962) — torque as observation vs. as auxiliary target

<!-- ## Citation

```
@mastersthesis{xu2026functional,
  title  = {Learning-based Functional Grasping for Dexterous Hands},
  author = {Xu, Zeyu},
  school = {Technical University of Munich},
  year   = {2026}
}
``` -->
