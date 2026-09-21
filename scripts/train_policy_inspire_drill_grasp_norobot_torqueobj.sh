#!/bin/bash
# Torque as OBJECTIVE only -- the pi0+obj cell of the TA-VLA ablation (arXiv:2509.07962 Table 5).
#
# Conditioning : point cloud (2048, cam1 alone) + joint POSITIONS (13)
# Prediction   : action chunk (13) AND future joint torque chunk (13), denoised jointly by the
#                same UNet through the same widened in/out conv
# Loss         : L = L_action + beta * L_torque      (beta = 0.1)
#
# The policy never OBSERVES torque here; it only has to predict it. That is the whole point of
# this cell: it isolates how much the auxiliary objective alone buys, separately from the gain of
# feeding torque in as an extra input.
#
# Why policy.state_obs_dim=13 rather than task.dataset.agent_pos_dim=13
# --------------------------------------------------------------------
# The auxiliary target is sliced out of agent_pos itself (simple_dp3.compute_loss reads
# nobs['agent_pos'][..., 13:26]) -- target and observation are the same tensor. Shrinking
# agent_pos to 13 would therefore delete the target as well and the run would crash on a
# zero-width slice. state_obs_dim keeps the batch at 26 for the target and narrows only what the
# encoder consumes, so the observation really is positions-only while the torque chunk is still
# supervised. It also disables state_split (there is no force block left in the input to split).
#
# Conditioning width drops to 128 (64 point cloud + 64 state) versus 192 when torque is observed
# -- that is expected and is exactly the variable this ablation isolates.
#
# Comparable runs (same data, same everything else):
#   train_policy_inspire_drill_grasp_norobot_contactgate.sh   obs=26 + aux + contact gate
#   this script with AUX_BETA=0 (or without the aux flags)    plain baseline, obs=13
#
# Reading the logs
#   bc_loss      action term only -> directly comparable across all cells of the ablation
#   torque_loss  auxiliary term (unweighted)
#   val_loss     bc_loss + 0.1*torque_loss -> NOT comparable with a no-aux run. Compare cells by
#                val_action_mse_error and by deployed success rate, never by val_loss.
#
# Usage:
#   bash scripts/train_policy_inspire_drill_grasp_norobot_torqueobj.sh [data_path] [seed] [gpu_id] [epochs] [beta]

DEBUG=False
save_ckpt=True

task_name=inspire_drill_grasp_norobot_contact
config_name=simple_dp3
data_path=${1:-/home/zeyu/inspire_drill/data/norobot.zarr}
seed=${2:-0}
gpu_id=${3:-0}
num_epochs=${4:-1000}
aux_beta=${5:-0.1}
addition_info=norobot_torqueobj
zarr_backend=${ZARR_BACKEND:-auto}

EXPECTED_PC=${EXPECTED_PC:-2048}   # cam1 alone (--disable_cam2 --no_robot)

IFS=',' read -ra _zarr_paths <<< "${data_path}"
for _p in "${_zarr_paths[@]}"; do
    read -r pc_points state_dim <<< $(/home/zeyu/anaconda3/envs/dp3/bin/python -c "
import zarr; z = zarr.open('${_p}', mode='r')
print(z['data']['point_cloud'].shape[1], z['data']['state'].shape[1])")
    if [ "${pc_points}" != "${EXPECTED_PC}" ]; then
        echo -e "\033[31m[ERROR] ${_p} 点云=${pc_points} != ${EXPECTED_PC}(--no_robot 采的纯相机点云)\033[0m"; exit 1
    fi
    # 26 维是硬要求:力矩目标取自 state[13:26],13 维数据下这个目标不存在
    if [ "${state_dim}" != "26" ]; then
        echo -e "\033[31m[ERROR] ${_p} state=${state_dim} != 26。力矩预测目标取自 state[13:26],"
        echo -e "采集时必须加 --force_state。注意:观测只用前 13 维是由 policy.state_obs_dim 控制的,"
        echo -e "不能靠缩小 state 维度来实现,那会把目标一起删掉\033[0m"; exit 1
    fi
done

exp_name="${task_name}-${config_name}-${addition_info}"
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m[Train NoRobot + TORQUE-AS-OBJECTIVE] zarr=${data_path} pc=${pc_points}(cam1 only) state=${state_dim} -> 观测只用前 13 维(关节位置) | 预测 action13 + torque13 | beta=${aux_beta} epochs=${num_epochs} gpu=${gpu_id}\033[0m"

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
    policy.state_obs_dim=13 \
    policy.contact_gate.enabled=false \
    +policy.aux_torque.enabled=true \
    +policy.aux_torque.start=13 \
    +policy.aux_torque.end=26 \
    +policy.aux_torque.beta=${aux_beta} \
    training.num_epochs=${num_epochs}
