DEBUG=False
save_ckpt=True

task_name=inspire_drill_grasp
data_path=${1:-/home/zeyu/inspire_drill/data/inspire_drill_dp3_1184s1.zarr}
config_name=${2:-simple_dp3}
seed=${3:-0}
gpu_id=${4:-0}
num_epochs=${5:-1000}
addition_info=grasp
case "${data_path}" in *,*) addition_info=grasp_mix;; esac   # 多 zarr 混训时区分 run 目录

EXPECTED_PC=${EXPECTED_PC:-1184}   # 1184=cam1 512|cam2 512|robot 160(纯抓取,无 plate/drill/ground)
first_path=${data_path%%,*}
pc_points=$(/home/zeyu/anaconda3/envs/dp3/bin/python -c "
import zarr; z = zarr.open('${first_path}', mode='r')
print(z['data']['point_cloud'].shape[1])")
if [ -z "${pc_points}" ]; then
    echo -e "\033[31m[ERROR] 读不到 ${first_path} 的点云形状,检查路径\033[0m"; exit 1
fi
if [ "${pc_points}" != "${EXPECTED_PC}" ]; then
    echo -e "\033[31m[ERROR] ${first_path} 点云=${pc_points} != 固定布局 ${EXPECTED_PC},拒绝训练"
    echo -e "(纯抓取布局 1184=cam1 512|cam2 512|robot 160;若拿的是串联 zarr(1696 含 plate),"
    echo -e " 用对应的 chained 训练脚本,别混布局)\033[0m"; exit 1
fi

exp_name="${task_name}-${config_name}-${addition_info}"
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m[Train InspireDrill-Grasp] task=${task_name} data=${data_path} pc=${pc_points} epochs=${num_epochs} exp=${exp_name} gpu=${gpu_id}\033[0m"

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
    training.checkpoint_every=5 \
    +training.save_ckpt_every_n_epochs=5 \
    dataloader.num_workers=8 \
    dataloader.pin_memory=False \
    dataloader.persistent_workers=False \
    val_dataloader.num_workers=8 \
    val_dataloader.pin_memory=False \
    "task.dataset.zarr_path=[${data_path}]" \
    training.num_epochs=${num_epochs}
