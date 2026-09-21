#!/bin/bash
# Torque as OBSERVATION only -- the pi0+obs cell of the ablation.
#
# Conditioning : point cloud (2048, cam1 alone) + [joint position 13 | joint torque 13]
#                encoded by a split state branch: pos_mlp(13->64) + force_mlp(13->64, dropout 0.1)
# Prediction   : action chunk (13) only
# Loss         : L = L_action
#
# The split branch (rather than one wider MLP over the 26-dim vector) and the dropout on the
# torque half keep the policy from collapsing onto the torque channel, which is far
# lower-dimensional and far less noisy than the point cloud.
#
# Torque is ungated: every joint's torque reaches the conditioning whether or not the hand is in
# contact. Against (5) this isolates the gating; against (1) it isolates the observation itself.
#
# Ablation cell:      obs=26   aux=off  gate=off      cond/step 192,  UNet 13 ch
#   (1) baseline      obs=13   aux=off  gate=off      train_..._norobot_baseline.sh
#   (2) THIS ONE      obs=26   aux=off  gate=off
#   (3) target only   obs=13   aux=on   gate=off      train_..._norobot_torqueobj.sh
#   (4) obs + target  obs=26   aux=on   gate=off      train_..._norobot_torqueboth.sh
#   (5) + gating      obs=26   aux=on   gate=on       train_..._norobot_contactgate.sh
#
# Usage:
#   bash scripts/train_policy_inspire_drill_grasp_norobot_torqueobs.sh [data_path] [seed] [gpu_id] [epochs]

DEBUG=False
save_ckpt=True

task_name=inspire_drill_grasp_norobot_contact
config_name=simple_dp3
data_path=${1:-/home/zeyu/inspire_drill/data/norobot.zarr}
seed=${2:-0}
gpu_id=${3:-0}
num_epochs=${4:-1000}
addition_info=norobot_torqueobs
zarr_backend=${ZARR_BACKEND:-auto}

EXPECTED_PC=${EXPECTED_PC:-2048}

IFS=',' read -ra _zarr_paths <<< "${data_path}"
for _p in "${_zarr_paths[@]}"; do
    read -r pc_points state_dim <<< $(/home/zeyu/anaconda3/envs/dp3/bin/python -c "
import zarr; z = zarr.open('${_p}', mode='r')
print(z['data']['point_cloud'].shape[1], z['data']['state'].shape[1])")
    if [ "${pc_points}" != "${EXPECTED_PC}" ]; then
        echo -e "\033[31m[ERROR] ${_p} 点云=${pc_points} != ${EXPECTED_PC}(--no_robot 采的纯相机点云)\033[0m"; exit 1
    fi
    # 26 维是硬要求:力矩要进观测,13 维数据下 state_split 会因 force_dim<=0 静默回退成单 MLP
    if [ "${state_dim}" != "26" ]; then
        echo -e "\033[31m[ERROR] ${_p} state=${state_dim} != 26,力矩要当观测,采集时必须加 --force_state\033[0m"; exit 1
    fi
done

exp_name="${task_name}-${config_name}-${addition_info}"
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m[Train NoRobot + TORQUE AS OBSERVATION] zarr=${data_path} pc=${pc_points} state=${state_dim}(pos13+torque13, 均进观测) | 只预测 action13 | epochs=${num_epochs} gpu=${gpu_id}\033[0m"

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
    task.dataset.load_contact=false \
    policy.contact_gate.enabled=false \
    training.num_epochs=${num_epochs}
