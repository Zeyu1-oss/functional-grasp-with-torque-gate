#!/bin/bash
# Torque as OBSERVATION (all 13 joints) + auxiliary TARGET restricted to the 6 HAND joints only.
#
# Same as train_policy_inspire_drill_grasp_norobot_torqueboth.sh (obs=26, aux=on, gate=off)
# except the auxiliary future-torque objective supervises agent_pos[20:26] (hand torque only)
# instead of agent_pos[13:26] (all 13, arm+hand). Observation is untouched -- the policy still
# SEES all 13 joints' torque; only what it is asked to PREDICT as an auxiliary task shrinks.
#
# Why: the contact-onset Delta-tau audit (tools/plot_contact_torque_delta.py) found the arm
# torque channel's signal-to-noise at contact is ~3.6-3.8x its own empty-hand baseline, against
# ~22-248x for the hand channel -- the arm torque trace is dominated by configuration/controller
# effort, not contact. The paper's "replication that failed" paragraph (Sec. IV-C) already
# floats "a torque trace dominated by free motion supplying a target that is largely controller
# effort" as a candidate explanation for why the auxiliary objective under-performs; this run
# tests that candidate directly by removing the noisy 7 arm dims from the target and keeping
# everything else identical to torqueboth (one variable apart, same ablation discipline as the
# rest of this family).
#
# Ablation cell:  obs=26  aux=on (hand-only, 6d)  gate=off
#   pair against: train_..._norobot_torqueboth.sh        (obs=26  aux=on all-13  gate=off, 73.3%)
#   pair against: train_..._norobot_torqueobj.sh          (obs=13  aux=on all-13 gate=off, 69.7%)
#
# Usage:
#   bash scripts/train_policy_inspire_drill_grasp_norobot_torqueboth_handaux.sh [data_path] [seed] [gpu_id] [epochs] [aux_beta]

DEBUG=False
save_ckpt=True

task_name=inspire_drill_grasp_norobot_contact
config_name=simple_dp3
data_path=${1:-/home/zeyu/inspire_drill/data/norobot.zarr}
seed=${2:-0}
gpu_id=${3:-0}
num_epochs=${4:-1000}
aux_beta=${5:-0.1}
addition_info=norobot_torqueboth_handaux
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
    if [ "${state_dim}" != "26" ]; then
        echo -e "\033[31m[ERROR] ${_p} state=${state_dim} != 26,采集时必须加 --force_state\033[0m"; exit 1
    fi
done

exp_name="${task_name}-${config_name}-${addition_info}"
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m[Train NoRobot + TORQUE OBS(13) + HAND-ONLY OBJ(6)] zarr=${data_path} pc=${pc_points} state=${state_dim}(pos13+torque13, 均进观测) | 预测 action13 + hand_torque6(agent_pos[20:26]) | beta=${aux_beta} epochs=${num_epochs} gpu=${gpu_id}\033[0m"

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
    +policy.aux_torque.enabled=true \
    +policy.aux_torque.start=20 \
    +policy.aux_torque.end=26 \
    +policy.aux_torque.beta=${aux_beta} \
    training.num_epochs=${num_epochs}
