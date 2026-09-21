#!/bin/bash
# Train DP3 on the full chained InspireDrill task (grasp + trigger + alignment).
#
# 数据由 collect_dp3_data.py --chained(全量)/ --stage1_only(抓取加强)--force_state 采集:
#   point_cloud 固定布局,不自动适配,点数不符直接拒训(EXPECTED_PC 校验,见下)。当前默认
#   [1696,3] = cam1 512|cam2 512|plate 512|robot 160(--ground_points 0,禁用地面段排查用)。
#   历史布局:1792=+ground 96(默认地面段);2304=+drill(mesh定点,oracle)512。
#   不同布局点数不同,zarr 不可混训,训练前用 EXPECTED_PC 覆盖确认点数匹配。
#   agent_pos   [26]     ([关节位置13|接触力13],训练只取前 13,见 task yaml agent_pos_dim)
#   action      [13]
# task 配置见 diffusion_policy_3d/config/task/inspire_drill_chained.yaml
#
# Usage:
#   bash scripts/train_policy_inspire_drill_chained.sh [data_path] [config_name] [seed] [gpu_id] [num_epochs]
#   data_path 支持逗号分隔多个 zarr(混训,如 串联全量+stage1 抓取加强;布局必须同代):
# 例:
#   bash scripts/train_policy_inspire_drill_chained.sh \
#       /home/zeyu/inspire_drill/data/inspire_drill_dp3_chained_eg.zarr,/home/zeyu/inspire_drill/data/inspire_drill_dp3_s1.zarr \
#       simple_dp3 0 0


DEBUG=False
save_ckpt=True

task_name=inspire_drill_chained
data_path=${1:-/home/zeyu/inspire_drill/data/inspire_drill_dp3_chained_v5.zarr}
config_name=${2:-simple_dp3}   # simple_dp3=本任务定制(默认);dp3=原版(大 encoder/batch128)
seed=${3:-0}
gpu_id=${4:-0}
num_epochs=${5:-1000}
addition_info=chained
case "${data_path}" in *,*) addition_info=chained_mix;; esac   # 多 zarr 混训时区分 run 目录


EXPECTED_PC=${EXPECTED_PC:-1696}   # 1696=camera+plate+robot(无drill oracle段、无ground段,--ground_points 0 采集);
                                    # 1792=+ground(96点);2304=+drill;需要时 EXPECTED_PC=1792/2304 覆盖
first_path=${data_path%%,*}
pc_points=$(/home/zeyu/anaconda3/envs/dp3/bin/python -c "
import zarr; z = zarr.open('${first_path}', mode='r')
print(z['data']['point_cloud'].shape[1])")
if [ -z "${pc_points}" ]; then
    echo -e "\033[31m[ERROR] 读不到 ${first_path} 的点云形状,检查路径\033[0m"; exit 1
fi
if [ "${pc_points}" != "${EXPECTED_PC}" ]; then
    echo -e "\033[31m[ERROR] ${first_path} 点云=${pc_points} != 固定布局 ${EXPECTED_PC},拒绝训练"
    echo -e "(可能布局:1696=cam1 512|cam2 512|plate 512|robot 160(--ground_points 0,无地面段);"
    echo -e " 1792=+ground 96(默认地面段);2304=+drill(mesh定点,oracle)512。"
    echo -e " 用 EXPECTED_PC=1792 或 EXPECTED_PC=2304 覆盖默认的 1696,别拿错代 zarr)\033[0m"; exit 1
fi

exp_name="${task_name}-${config_name}-${addition_info}"
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m[Train InspireDrill-Chained] task=${task_name} data=${data_path} pc=${pc_points} epochs=${num_epochs} exp=${exp_name} gpu=${gpu_id}\033[0m"

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
