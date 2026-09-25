# Learning-based Functional Grasping for Dexterous Hands

**Master's thesis · Technical University of Munich (TUM)**<br>
Supervisors: Qian Feng, Zitao Zhang

Isaac Lab environments, data pipeline and training/evaluation code for functional grasping of a
power drill with a 7-DoF Franka arm and a 6-DoF Inspire five-finger hand.

<p align="center">
  <img src="docs/img/setup.png" width="49%" alt="Isaac Lab scene: Franka arm with Inspire hand and a power drill">
  <img src="docs/img/functional_grasp.png" width="35%" alt="A functional grasp: handle enclosed, index finger at the trigger">
</p>

A *functional* grasp acquires the tool in the configuration from which it can be operated: handle
enclosed, index finger on the trigger. A privileged PPO teacher is trained on simulator state, then
distilled into a [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) student that
sees only a depth-camera point cloud and joint proprioception. The question studied is how joint
torque should enter that student — and the answer is that it has to be **gated on contact** before
it helps at all. The gate follows [FoAR](https://arxiv.org/abs/2411.15753), but is supervised from
per-link simulator contact, so its label is exact and its threshold task-independent. The labels
enter the loss only, never the network input, so the policy still consumes what a real robot can
measure.

---

## Installation

This branch (`main`) is the Isaac Lab side: both PPO teacher stages, data collection, and deploy.
Student training lives on the [**`dp3` branch**](https://github.com/Zeyu1-oss/functional-grasp-with-torque-gate/tree/dp3)
of this same repo and needs its own Python env, so the two are checked out side by side:

```
<workspace>/
├── functional-grasp-with-torque-gate/   this branch (main) — Python 3.11 + Isaac Lab
└── 3D-Diffusion-Policy/                 the dp3 branch     — Python 3.8, student training only
```

1. Isaac Sim + Isaac Lab (developed against Python 3.11 / isaaclab 0.53.1): follow the official
   [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

```bash
# 2. This branch + extra packages, in that same env
git clone https://github.com/Zeyu1-oss/functional-grasp-with-torque-gate.git
cd functional-grasp-with-torque-gate
pip install rl_games==1.6.1 zarr numcodecs dill omegaconf trimesh

# 3. USD assets (gitignored), then build the canonical robot cloud once
pip install gdown
gdown --fuzzy 'https://drive.google.com/file/d/1PdrOZZjNwIF0OrMTwvL6x9sda6tw_FAN/view?usp=drive_link' -O assets.zip
unzip assets.zip -d assets/
python tools/build_robot_pointcloud.py     # -> assets/inspire_tac/robot_canonical_points.npz

# 4. The trained DP3 student, if you only want to run it — this alone is enough for step 5 below,
#    and lets you skip teacher training, data collection and student training entirely
gdown --fuzzy 'https://drive.google.com/file/d/19svkGp74ZOuaX3Tyom0twH36VG3Tywde/view?usp=drive_link' -O dp3_student.ckpt

# 5. The dp3 branch, in its own directory — only needed to TRAIN a student (step 4 of the pipeline
#    below). Skip it if you downloaded the checkpoint above.
cd .. && git clone -b dp3 https://github.com/Zeyu1-oss/functional-grasp-with-torque-gate.git 3D-Diffusion-Policy
```

---

## Repository Layout

```
tasks/          Isaac Lab envs (GraspDrillEnv -> Stage2Env -> ChainedEnv), reward/termination terms
scripts/        Entry points — train / play / collect / deploy (the pipeline below)
perception/     Point-cloud & observation code, shared by collect and deploy
config/         Scene, drill-variant, and RL-games/DP3 agent YAMLs
tools/          Asset prep, eval poses, analysis and plotting — outside the repro path
results/        Plots and tables assembled from runs/
assets/ data/ collected_data/ runs/ output/   Generated — gitignored
```

To grasp a tool other than the three drills shipped here, see
**[docs/ADDING_OBJECTS.md](docs/ADDING_OBJECTS.md)** — asset preparation and the variant fields
that have to be annotated per object.

---

## Usage

<p align="center"><img src="docs/img/pipelinev1.png" width="88%" alt="Teacher/student pipeline: (a) VLM-assisted functional targets and a 74-d privileged state drive a PPO teacher whose Isaac Lab rollouts are recorded as demonstrations; (b) the student encodes a two-frame point cloud, joint positions and torques, mixes the torque feature through a contact-supervised gate, and denoises actions plus an auxiliary torque sequence"></p>

**(a)** The teacher is a single PPO stage — no curriculum — trained on a 74-d privileged state
against functional targets annotated in the object frame; one episode reorients the tool, grasps
it, and reaches the trigger. A second, separate teacher (`train2.py`) learns screw-hole alignment,
warm-started from grasp end-states rather than continuing the first. **(b)** The student is
distilled offline and sees only point cloud, joint positions and joint torque; the annotations and
contact labels are training signal and never enter its input.

Everything runs in this branch's Isaac Lab env **except step 4**, which switches to the `dp3`
checkout and its own env. **To just run the trained policy, download the checkpoint (Installation
step 4) and go straight to step 5** — steps 1–4 reproduce it from scratch.

```bash
# 1. Stage-1 teacher: grasp (PPO)
python scripts/train_with_rl_games.py --headless --num_envs 4096

# optional: watch it (play_stage2.py / play_chained.py for stage 2 / both stages)
python scripts/play_drill.py --checkpoint <ckpt>

# 2. Bank successful grasp end-states, then train the stage-2 teacher: align (PPO)
python scripts/collect_success_data.py --headless --num_envs 4096 \
    --checkpoint <ckpt> --output collected_data/success_data.pkl
python scripts/train2.py --headless --num_envs 4096 --dataset collected_data/success_data.pkl

# 3. Collect DP3 demonstrations (only successful episodes are written)
python scripts/collect_dp3_data.py --stage1_only --headless \
    --num_envs 256 --episodes_per_variant 1000 \
    --disable_cam2 --no_robot --force_state --save_contact \
    --stage1_checkpoint <ckpt> --output data/norobot.zarr

# 4. Train the DP3 student — the dp3 checkout, its own env
cd ../3D-Diffusion-Policy && bash scripts/train_policy_inspire_drill_grasp_norobot_eq1_auxtorque.sh \
    ../functional-grasp-with-torque-gate/data/norobot.zarr
cd ../functional-grasp-with-torque-gate

# 5. Deploy and grade a student. The downloaded checkpoint (Installation step 4) runs as-is —
#    it was trained with exactly these flags, so steps 1-4 above are not needed to run it.
python scripts/deploy_dp3_sim.py --stage1_only --headless --num_envs 70 --disable_cam2 --no_robot \
    --dp3_ckpt dp3_student.ckpt

# ...or a checkpoint you trained yourself in step 4, same flags as the collection in step 3
python scripts/deploy_dp3_sim.py --stage1_only --headless --num_envs 70 --disable_cam2 --no_robot \
    --dp3_ckpt ../3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/<run>/checkpoints/epoch_0180.ckpt
```

To compare checkpoints, add `--init_pose_file`: it replays a fixed pose set, each pose once, so every
checkpoint is graded on identical initial conditions. Generate one with
`tools/make_sobol_init_poses.py -n 100` (per variant, so 300 total) — drawn independently of any
policy, rather than reused from the teacher's successes. Other deploy modes: `--stage2_rl` (DP3
grasps, teacher aligns), `--stage2_dp3_ckpt` (both stages distilled), `--dump_gate` (log the gate
against ground-truth contact), `--policy rl`.

> **Collect and deploy must agree.** Both build the observation from the same `perception/` code,
> but its composition is chosen by flags (`--disable_cam2`, `--no_robot`, `--force_state`, …). A
> mismatch is silent — only the point-cloud *size* is checked against the checkpoint at startup.

The tables and figures below are produced from these runs by `tools/`: `sweep_eval.py` grades every
checkpoint of a run into a CSV, `analyze_gate.py` and `plot_*.py` turn `--dump_gate` / `--dump_torque`
dumps into the plots. `tools/isaac_python.sh` joins the conda env with Isaac Sim's kit bindings, if
your shell doesn't already.

---

## Results

In simulation, on 300 unseen initial poses shared by every checkpoint. Success requires envelopment
*and* trigger reachability, held for 20 of the last 50 control steps.

| Policy | Observation | Success |
|---|---|---|
| Grasp teacher | privileged state (74-d) | 93.0 % |
| Alignment teacher | privileged state + plate pose | 92.0 % |
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

Gating carries the result: +9.3 points over the baseline (n = 300, single seed, p < 0.01), of which
the gate alone accounts for +8.0. The two smaller increments are within the resolution of a
300-episode evaluation.

On drill geometries held out from training the hand still encloses the handle, but the *functional*
condition often fails — trigger offsets are specified per variant in the object frame and do not
transfer. Regressing them from the point cloud is the obvious next step.

<p align="center">
  <img src="docs/img/stage2_alignment.png" width="62%" alt="Stage 2: the grasped drill aligned against the target plate">
</p>

---

## References

- Ze et al., **3D Diffusion Policy**, RSS 2024 — the student architecture
- He et al., **FoAR: Force-Aware Reactive Policy**, RA-L 2025 ([arXiv:2411.15753](https://arxiv.org/abs/2411.15753)) — the contact-gated branch adopted here
- Lei et al., **Learning When to See and When to Feel**, [arXiv:2604.01414](https://arxiv.org/abs/2604.01414) — suppressing torque in free motion carries most of the benefit
- Zhang et al., **TA-VLA**, [arXiv:2509.07962](https://arxiv.org/abs/2509.07962) — torque as observation vs. as auxiliary target
