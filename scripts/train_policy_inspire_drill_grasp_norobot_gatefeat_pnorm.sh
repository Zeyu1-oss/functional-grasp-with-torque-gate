#!/bin/bash
# FEATURE-SPACE contact gate (arXiv:2604.01414 Eq.1) + percentile torque normalization.
#
# The gate mechanism of that paper is
#       f_torque-gated = phi * f_torque + (1 - phi) * f*
# where f_torque is the OUTPUT of the torque encoder and f* is a learnable parameter standing for
# the encoded torque feature during free-space movement. Our existing runs do NOT implement this:
# train_..._norobot_{contactgate,gatehard_pnorm}.sh gate the RAW torque going INTO the encoder,
# with the learnable stand-in living in torque space (6 numbers) rather than feature space (64).
# Those are two different mechanisms, and this script is the faithful one.
#
# What is gated where:
#     tau(13) -> finger 6 -> force_mlp(6->64->64) = f_torque      encoder sees the REAL torque
#                         -> gate_mlp(6->64->1)   = phi           torque only, no joint angles
#                         -> phi*f_torque + (1-phi)*f_star        f_star = Parameter(64), init 0
# The point cloud branch and pos_mlp are untouched, as is the torque block fed to the encoder
# (still the 6 finger dims; the arm's 7 are dropped for the same reason as before -- their std is
# 10x the fingers' and they are dominated by inertia, not contact).
#
# Why exactly ONE gate. A 64-d feature cannot be split across 6 finger gates without inventing an
# arbitrary blocking, so feature-space gating is necessarily a single scalar phi. The BCE therefore
# gets ONE head, supervised by contact_gate.global_group ("is the hand touching the drill at all")
# instead of the six per-finger labels. That is also what the paper does.
#
# Why the gate sees torque only. With a single global label the shortcut risk is at its worst:
# finger joint angles alone "predict" contact at AUC 0.909 purely by encoding "the hand is closed,
# so it is probably holding something" -- which fails exactly on a closed but empty hand. Torque
# alone measures 0.947. So gate_in is the same 6 finger torques the encoder sees, nothing else.
#
# Both arms run scope=finger_single: the finger-6 torque block, ONE gate, and the single global
# label. That scope exists precisely so the control below is matched -- scope=per_finger would give
# the control six gates and six labels, and scope=global would give it a 13-dim torque block, so
# neither would isolate the gate position.
#     GATE_POS=feature  (default)  gate the encoder OUTPUT   -- the paper's Eq.1
#     GATE_POS=input               gate the raw torque       -- the matched control
# Do NOT compare the feature run against gatehard_pnorm.sh directly: that one is per_finger with
# SIX gates and six labels, so it differs in both the gate position and the gate count.
#
# For the percentile torque normalization and why clamp_agent_pos must be on with it, see the
# header of train_policy_inspire_drill_grasp_norobot_gatehard_pnorm.sh.
#
# Cell map (all: new normalization, aux torque on, hard label):
#   per_finger + input   (6 gates)  train_..._norobot_gatehard_pnorm.sh
#   global-1   + input   (1 gate)   THIS, with GATE_POS=input
#   global-1   + feature (1 gate)   THIS, default            <- paper Eq.1
#
# Usage:
#   bash scripts/train_policy_inspire_drill_grasp_norobot_gatefeat_pnorm.sh [data_path] [seed] [gpu_id] [epochs] [gamma]
#   GATE_POS=input bash scripts/train_policy_inspire_drill_grasp_norobot_gatefeat_pnorm.sh

DEBUG=False
save_ckpt=True

task_name=inspire_drill_grasp_norobot_contact
config_name=simple_dp3
data_path=${1:-/home/zeyu/inspire_drill/data/norobot.zarr}
seed=${2:-0}
gpu_id=${3:-0}
num_epochs=${4:-1000}
gamma=${5:-0.1}
aux_beta=${AUX_BETA:-0.1}
torque_pct=${TORQUE_PCT:-1.0}
gate_pos=${GATE_POS:-feature}          # feature | input
zarr_backend=${ZARR_BACKEND:-auto}

EXPECTED_PC=${EXPECTED_PC:-2048}

case "${gate_pos}" in
    feature|input) ;;
    *) echo -e "\033[31m[ERROR] GATE_POS 只能是 feature 或 input,收到 '${gate_pos}'\033[0m"; exit 1 ;;
esac
addition_info=norobot_gatefeat_pnorm_${gate_pos}

IFS=',' read -ra _zarr_paths <<< "${data_path}"
for _p in "${_zarr_paths[@]}"; do
    read -r pc_points state_dim has_contact <<< $(/home/zeyu/anaconda3/envs/dp3/bin/python -c "
import zarr; z = zarr.open('${_p}', mode='r')
print(z['data']['point_cloud'].shape[1], z['data']['state'].shape[1],
      1 if 'contact' in z['data'] else 0)")
    if [ "${pc_points}" != "${EXPECTED_PC}" ]; then
        echo -e "\033[31m[ERROR] ${_p} 点云=${pc_points} != ${EXPECTED_PC}。本脚本要的是 --no_robot 采的纯相机点云\033[0m"; exit 1
    fi
    if [ "${state_dim}" != "26" ]; then
        echo -e "\033[31m[ERROR] ${_p} state=${state_dim} != 26,采集时必须加 --force_state\033[0m"; exit 1
    fi
    if [ "${has_contact}" != "1" ]; then
        echo -e "\033[31m[ERROR] ${_p} 没有 data/contact。门控的 BCE 监督取自它,采集时必须加 --save_contact\033[0m"; exit 1
    fi
done

exp_name="${task_name}-${config_name}-${addition_info}"
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m[Train NoRobot + AUX-TORQUE + GATE(${gate_pos}, 单门/全局标签) + p${torque_pct} 力矩归一化] zarr=${data_path} pc=${pc_points} state=${state_dim}(pos13+torque13) 力矩支路输入=手指6维 beta=${aux_beta} gamma=${gamma} epochs=${num_epochs} gpu=${gpu_id}\033[0m"

cd 3D-Diffusion-Policy

export PYTHONPATH="${HOME}/3D-Diffusion-Policy:${HOME}/3D-Diffusion-Policy/3D-Diffusion-Policy:${HOME}/3D-Diffusion-Policy/third_party/pytorch3d_simplified"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

/home/zeyu/anaconda3/envs/dp3/bin/python train.py --config-name=${config_name}.yaml \
    task=${task_name} \
    hydra.run.dir=${run_dir} \
    training.debug=$DEBUG \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    logging.mode=online \
    checkpoint.save_ckpt=${save_ckpt} \
    training.checkpoint_every=10 \
    +training.save_ckpt_every_n_epochs=10 \
    dataloader.num_workers=8 \
    dataloader.pin_memory=False \
    dataloader.persistent_workers=False \
    val_dataloader.num_workers=8 \
    val_dataloader.pin_memory=False \
    "task.dataset.zarr_path=[${data_path}]" \
    +task.dataset.zarr_backend=${zarr_backend} \
    +task.dataset.torque_start=13 \
    +task.dataset.torque_percentile=${torque_pct} \
    policy.clamp_agent_pos=true \
    +policy.aux_torque.enabled=true \
    +policy.aux_torque.start=13 \
    +policy.aux_torque.end=26 \
    +policy.aux_torque.beta=${aux_beta} \
    policy.contact_gate.enabled=true \
    policy.contact_gate.scope=finger_single \
    policy.contact_gate.gate_position=${gate_pos} \
    policy.contact_gate.beta=${gamma} \
    policy.contact_gate.soft_label=false \
    training.num_epochs=${num_epochs}
