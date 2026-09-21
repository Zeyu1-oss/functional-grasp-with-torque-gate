#!/bin/bash
# Train DP3 on grasp-only InspireDrill, cam1-only point cloud (cam2/wrist cam disabled) + robot FK
# points, WITH joint-torque observation (state split), no mask/rgb channel.
#
# 与 train_policy_inspire_drill_grasp_force.sh 的区别:点云换成 cam1 2048 + robot 512(cam2 完全禁用),
#   agent_pos 仍是满 26 维 [关节位置13 | 关节力矩13]（perception/student_obs.py 已从"接触力"
#   改成"力矩" applied_torque，同样触发 simple_dp3.yaml 的 state_split）。
#
# 数据由 collect_dp3_data.py --stage1_only --disable_cam2 --pc_num_points 2048 \
#   --robot_pc_points 512 --force_state ... 采集。
#
# 点云 [2560,3] = cam1 2048 | robot 512（无 mask/rgb）。EXPECTED_PC=2560。
#
# data_path 可传逗号分隔的多个 zarr 混训(InspireDrillDataset 原生支持,各 zarr 独立划 val)。
# 注意:这台机器只有 15GB 内存,finalstage1.zarr(6.6GB)+ finalstagecoversobol.zarr(13.78GB 且还在
# 长)加起来 ~20GB,超过可用内存——2026-07-27 最早直接用 copy_from_path 整包进 RAM 时把训练进程
# OOM 杀了(dmesg: Out of memory, anon-rss:12466260kB)。之后给 InspireDrillDataset 加了
# zarr_backend='auto':按可用内存自动判断,装不下就整批退化成 ReplayBuffer.create_from_path
# (zarr 磁盘直读 + chunk 级解压缓存,不会 OOM,只是比整包进 RAM 慢一点,实测 7.75 it/s 还可以),
# 所以现在两个 zarr 一起混训是安全的,默认改回都加上。
#
# zarr_backend: 默认 auto(按可用内存自动选,见上面的说明),装不下会自动退化成磁盘直读。
# 确定数据集当前肯定能整包塞进内存(比如只跑一个小 zarr)时,可以用环境变量强制走内存版更快:
#   ZARR_BACKEND=numpy bash scripts/train_policy_inspire_drill_grasp_cam1_2048_force.sh ...
#
# Usage:
#   bash scripts/train_policy_inspire_drill_grasp_cam1_2048_force.sh [data_path] [config_name] [seed] [gpu_id] [num_epochs]
#   data_path 单个: /path/a.zarr   多个: /path/a.zarr,/path/b.zarr

DEBUG=False
save_ckpt=True

task_name=inspire_drill_grasp_cam1_2048_force
data_path=${1:-/home/zeyu/inspire_drill/data/1.zarr}
config_name=${2:-simple_dp3}
seed=${3:-0}
gpu_id=${4:-0}
num_epochs=${5:-1000}
addition_info=cam1_2048_force
zarr_backend=${ZARR_BACKEND:-auto}   # auto(默认,按内存自动判断) / numpy(强制整包进RAM) / zarr(强制磁盘直读)

# data_path 支持逗号分隔的多个 zarr 混训(InspireDrillDataset 原生支持,见该文件注释);
# 这里对每一个都校验一遍点云/state 形状,任何一个不匹配就在拉起 Isaac/torch 之前直接拒绝,
# 避免混进不同代数据导致 state_split/point_cloud 维度不一致的隐蔽错误。
EXPECTED_PC=${EXPECTED_PC:-2560}   # 2560=cam1 2048(cam2 disabled)|robot 512(纯抓取,无 plate/drill/ground)
IFS=',' read -ra _zarr_paths <<< "${data_path}"
for _p in "${_zarr_paths[@]}"; do
    pc_points=$(/home/zeyu/anaconda3/envs/dp3/bin/python -c "
import zarr; z = zarr.open('${_p}', mode='r')
print(z['data']['point_cloud'].shape[1])")
    if [ -z "${pc_points}" ]; then
        echo -e "\033[31m[ERROR] 读不到 ${_p} 的点云形状,检查路径\033[0m"; exit 1
    fi
    if [ "${pc_points}" != "${EXPECTED_PC}" ]; then
        echo -e "\033[31m[ERROR] ${_p} 点云=${pc_points} != 固定布局 ${EXPECTED_PC},拒绝训练\033[0m"; exit 1
    fi

    # state 必须是 26 维(含关节力矩),否则 state_split 会因 force_dim<=0 回退成单 MLP(等于没用力矩)
    state_dim=$(/home/zeyu/anaconda3/envs/dp3/bin/python -c "
import zarr; z = zarr.open('${_p}', mode='r')
print(z['data']['state'].shape[1])")
    if [ "${state_dim}" != "26" ]; then
        echo -e "\033[31m[ERROR] ${_p} state=${state_dim} 维 != 26(需含关节力矩)。"
        echo -e "force 模式要求 collect 时加 --force_state。用 13 维数据跑本脚本 state_split 会静默失效\033[0m"; exit 1
    fi
done

exp_name="${task_name}-${config_name}-${addition_info}"
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m[Train InspireDrill-Grasp-Cam1-2048-Force] task=${task_name} zarr(${#_zarr_paths[@]})=${data_path} pc=${pc_points}(cam1 2048+robot 512) state=${state_dim}(pos13+torque13) zarr_backend=${zarr_backend} epochs=${num_epochs} exp=${exp_name} gpu=${gpu_id}\033[0m"

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
    training.num_epochs=${num_epochs}
