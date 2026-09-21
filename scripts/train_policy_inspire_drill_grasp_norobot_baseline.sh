#!/bin/bash
# Baseline: no torque anywhere -- the pi0 cell of the ablation.
#
# Conditioning : point cloud (2048, cam1 alone) + joint positions (13), single state MLP
# Prediction   : action chunk (13) only
# Loss         : L = L_action
#
# Uses its own task config with agent_pos declared as 13 rather than narrowing a 26-dim state
# with policy.state_obs_dim. The difference shows up at DEPLOY time: deploy_dp3_sim.py reads the
# agent_pos width out of the checkpoint to decide whether to assemble the torque half at all, so
# a 26-dim declaration would make a policy that never looks at torque still demand it from the
# robot. The cells that use torque as an auxiliary TARGET cannot do this -- the target is sliced
# from agent_pos itself -- and keep 26 with state_obs_dim instead.
#
# Ablation cell:      obs=13   aux=off  gate=off      cond/step 128,  UNet 13 ch
#   (1) THIS ONE      obs=13   aux=off  gate=off
#   (2) obs only      obs=26   aux=off  gate=off      train_..._norobot_torqueobs.sh
#   (3) target only   obs=13   aux=on   gate=off      train_..._norobot_torqueobj.sh
#   (4) obs + target  obs=26   aux=on   gate=off      train_..._norobot_torqueboth.sh
#   (5) + gating      obs=26   aux=on   gate=on       train_..._norobot_contactgate.sh
#
# Usage:
#   bash scripts/train_policy_inspire_drill_grasp_norobot_baseline.sh [data_path] [seed] [gpu_id] [epochs]

DEBUG=False
save_ckpt=True

task_name=inspire_drill_grasp_norobot_pos13
config_name=simple_dp3
data_path=${1:-/home/zeyu/inspire_drill/data/norobot.zarr}
seed=${2:-0}
gpu_id=${3:-0}
num_epochs=${4:-1000}
addition_info=norobot_baseline
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
    # zarr 存 26 维(--force_state 采的),dataset.agent_pos_dim=13 在加载时切掉力矩那半
    if [ "${state_dim}" != "26" ]; then
        echo -e "\033[31m[ERROR] ${_p} state=${state_dim} != 26。消融各组必须同源;本组在 dataset 侧切成 13 维\033[0m"; exit 1
    fi
done

exp_name="${task_name}-${config_name}-${addition_info}"
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m[Train NoRobot BASELINE] zarr=${data_path} pc=${pc_points} 观测只用前 13 维(关节位置) | 只预测 action13 | 无力矩 | epochs=${num_epochs} gpu=${gpu_id}\033[0m"

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
    policy.contact_gate.enabled=false \
    training.num_epochs=${num_epochs}
