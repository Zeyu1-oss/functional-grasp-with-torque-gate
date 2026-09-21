#!/bin/bash
# CONTROL RUN: percentile torque normalization + the existing HARD 0/1 contact gate.
#
# Identical to train_..._norobot_contactgate.sh except for the normalization of the torque half
# of agent_pos. It exists so that "soft gate vs hard gate" (against
# train_..._norobot_gatesoft_pnorm.sh) is a single-variable comparison: both runs share the new
# normalization, so any difference is the label, not the scaling.
#
# Why the normalization changed (measured on norobot.zarr, 435781 frames):
#   agent_pos[13:26] is joint torque, and its min/max are the actuator effort limits
#   (+-87 / +-40 / +-10 Nm) which only <=0.8% of frames ever touch. Fitting the limits
#   normalizer on them squeezed 98% of the torque into 12% of [-1,1]:
#       normalized std   action 0.522   pos13 0.394   torque 0.106
#   In diffusion each channel's SNR_t = alpha_bar_t * sigma^2 / (1 - alpha_bar_t), so a 5x
#   smaller sigma is a 25x lower SNR: with squaredcos_cap_v2 over 100 steps the torque channels
#   fall below SNR=1 at t=5 while the action channels hold to t=29. For ~95% of uniformly
#   sampled timesteps the torque half of the trajectory was indistinguishable from noise, and
#   L_action + beta*L_torque weighted the torque term about 24x below its nominal beta.
#   p1/p99 scaling brings sigma to ~0.41, level with pos13 and action.
#
#   The price is ~2% of frames (the saturation frames) landing outside [-1,1], up to +-8.9,
#   where MSE (squared) hands those 2% about 11% of the torque gradient -- and clip_sample=True
#   means the sampler could never emit such a value anyway. policy.clamp_agent_pos=true clamps
#   them back. The two flags MUST move together; SimpleDP3 prints a red warning if only the
#   dataset one is set.
#
# Ablation cells (all obs=26, aux=on unless noted):
#   (1) baseline      obs=13 aux=off gate=off  old norm   train_..._norobot_baseline.sh
#   (4) no gate                     gate=off   old norm   train_..._norobot_torqueboth.sh
#   (5) hard gate                   gate=on    old norm   train_..._norobot_contactgate.sh
#   (6) THIS ONE                    gate=on    NEW norm
#   (7) soft gate                   gate=on    NEW norm   train_..._norobot_gatesoft_pnorm.sh
# (6) vs (5) isolates the normalization; (7) vs (6) isolates the label. val_loss is not
# comparable across any of these -- use bc_loss, val_action_mse_error, gate/separation.
#
# Usage:
#   bash scripts/train_policy_inspire_drill_grasp_norobot_gatehard_pnorm.sh [data_path] [seed] [gpu_id] [epochs] [gamma]

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
torque_pct=${TORQUE_PCT:-1.0}     # p1/p99; TORQUE_PCT=0.5 for the more conservative p0.5/p99.5
addition_info=norobot_gatehard_pnorm
zarr_backend=${ZARR_BACKEND:-auto}

EXPECTED_PC=${EXPECTED_PC:-2048}   # 2048 = cam1 alone (--disable_cam2 --no_robot)

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
    # 没有 contact 就没有门的监督信号,gate 会退化成自由变量 —— 与其静默跑歪不如直接拒绝
    if [ "${has_contact}" != "1" ]; then
        echo -e "\033[31m[ERROR] ${_p} 没有 data/contact。门控的 BCE 监督取自它,采集时必须加 --save_contact\033[0m"; exit 1
    fi
done

exp_name="${task_name}-${config_name}-${addition_info}"
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m[Train NoRobot + AUX-TORQUE + HARD GATE + p${torque_pct} 力矩归一化] zarr=${data_path} pc=${pc_points}(cam1 only) state=${state_dim}(pos13+torque13) contact=13 beta=${aux_beta} gamma=${gamma} epochs=${num_epochs} gpu=${gpu_id}\033[0m"

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
    policy.contact_gate.beta=${gamma} \
    policy.contact_gate.soft_label=false \
    training.num_epochs=${num_epochs}
