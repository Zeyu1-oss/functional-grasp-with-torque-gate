
DEBUG=False
save_ckpt=True

task_name=inspire_drill_grasp_handpc1280
data_path=${1:-/home/zeyu/inspire_drill/data/handpc1280.zarr}
config_name=${2:-simple_dp3}
seed=${3:-0}
gpu_id=${4:-0}
num_epochs=${5:-1000}
addition_info=handpc1280
zarr_backend=${ZARR_BACKEND:-auto}

# 前置校验:形状不对就在拉起 torch 之前拒绝,避免混进不同代数据导致 state_split /
# point_cloud 维度不一致的隐蔽错误。
EXPECTED_PC=${EXPECTED_PC:-3328}   # 3328 = cam1 2048(cam2 disabled) | robot 1280(hand-only)
IFS=',' read -ra _zarr_paths <<< "${data_path}"
for _p in "${_zarr_paths[@]}"; do
    pc_points=$(/home/zeyu/anaconda3/envs/dp3/bin/python -c "
import zarr; z = zarr.open('${_p}', mode='r')
print(z['data']['point_cloud'].shape[1])")
    if [ -z "${pc_points}" ]; then
        echo -e "\033[31m[ERROR] 读不到 ${_p} 的点云形状,检查路径\033[0m"; exit 1
    fi
    if [ "${pc_points}" != "${EXPECTED_PC}" ]; then
        echo -e "\033[31m[ERROR] ${_p} 点云=${pc_points} != 固定布局 ${EXPECTED_PC}\033[0m"
        echo -e "\033[31m        2560 是旧的 robot 512(全身)布局,要用 cam1_2048_force 那个脚本\033[0m"; exit 1
    fi

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

echo -e "\033[33m[Train InspireDrill-Grasp-HandPC1280] task=${task_name} zarr(${#_zarr_paths[@]})=${data_path} pc=${pc_points}(cam1 2048+hand 1280) state=${state_dim}(pos13+torque13) zarr_backend=${zarr_backend} epochs=${num_epochs} exp=${exp_name} gpu=${gpu_id}\033[0m"

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
