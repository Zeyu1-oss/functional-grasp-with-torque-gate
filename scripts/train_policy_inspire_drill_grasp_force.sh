#!/bin/bash
# Train DP3 on the grasp-only InspireDrill task WITH contact-force observation (state split).
#
# 与 train_policy_inspire_drill_grasp.sh 的唯一区别:task=inspire_drill_grasp_force
#   -> agent_pos 用满 26 维 [关节位置13 | 接触力13],触发 simple_dp3.yaml 的 state_split
#      (pos13 一个 MLP、force13 另一个带 dropout 的 MLP,再和点云特征 concat)。
# 数据同一份(1184s1.zarr 的 state 本就存了 26 维),无需重采。
#
# 点云 [1184,3] = cam1 512 | cam2 512 | robot 160。EXPECTED_PC=1184。
#
# Usage:
#   bash scripts/train_policy_inspire_drill_grasp_force.sh [data_path] [config_name] [seed] [gpu_id] [num_epochs]

DEBUG=False
save_ckpt=True

task_name=inspire_drill_grasp_force
data_path=${1:-/home/zeyu/inspire_drill/data/inspire_drill_dp3_1184s1.zarr}
config_name=${2:-simple_dp3}
seed=${3:-0}
gpu_id=${4:-0}
num_epochs=${5:-1000}
addition_info=grasp_force
case "${data_path}" in *,*) addition_info=grasp_force_mix;; esac

EXPECTED_PC=${EXPECTED_PC:-1184}   # 1184=cam1 512|cam2 512|robot 160(纯抓取,无 plate/drill/ground)
first_path=${data_path%%,*}
pc_points=$(/home/zeyu/anaconda3/envs/dp3/bin/python -c "
import zarr; z = zarr.open('${first_path}', mode='r')
print(z['data']['point_cloud'].shape[1])")
if [ -z "${pc_points}" ]; then
    echo -e "\033[31m[ERROR] 读不到 ${first_path} 的点云形状,检查路径\033[0m"; exit 1
fi
if [ "${pc_points}" != "${EXPECTED_PC}" ]; then
    echo -e "\033[31m[ERROR] ${first_path} 点云=${pc_points} != 固定布局 ${EXPECTED_PC},拒绝训练\033[0m"; exit 1
fi

# state 必须是 26 维(含接触力),否则 state_split 会因 force_dim<=0 回退成单 MLP(等于没用 force)
state_dim=$(/home/zeyu/anaconda3/envs/dp3/bin/python -c "
import zarr; z = zarr.open('${first_path}', mode='r')
print(z['data']['state'].shape[1])")
if [ "${state_dim}" != "26" ]; then
    echo -e "\033[31m[ERROR] ${first_path} state=${state_dim} 维 != 26(需含接触力)。"
    echo -e "force 模式要求 collect 时加 --force_state。用 13 维数据跑本脚本 state_split 会静默失效\033[0m"; exit 1
fi

exp_name="${task_name}-${config_name}-${addition_info}"
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m[Train InspireDrill-Grasp-Force] task=${task_name} data=${data_path} pc=${pc_points} state=${state_dim}(pos13+force13) epochs=${num_epochs} exp=${exp_name} gpu=${gpu_id}\033[0m"

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
    training.num_epochs=${num_epochs}
