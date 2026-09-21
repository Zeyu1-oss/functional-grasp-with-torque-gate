
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
addition_info=norobot_contactgate
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

echo -e "\033[33m[Train NoRobot + AUX-TORQUE + CONTACT-GATE] zarr=${data_path} pc=${pc_points}(cam1 only) state=${state_dim}(pos13+torque13) contact=13 beta=${aux_beta} gamma=${gamma} epochs=${num_epochs} gpu=${gpu_id}\033[0m"

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
    +policy.aux_torque.enabled=true \
    +policy.aux_torque.start=13 \
    +policy.aux_torque.end=26 \
    +policy.aux_torque.beta=${aux_beta} \
    policy.contact_gate.enabled=true \
    policy.contact_gate.beta=${gamma} \
    training.num_epochs=${num_epochs}
