#!/bin/bash
# Global contact gate: ONE gate over all 13 joint torques, supervised by the contact sensors.
#
# Conditioning : point cloud (2048, cam1 alone) + [joint position 13 | gated joint torque 13]
# Gate         : a single scalar phi = sigmoid(gate_mlp([q_finger(6), tau_finger(6)]))
#                    tau_tilde = phi * tau_13 + (1 - phi) * tau_free_13
#                no contact anywhere -> all 13 torque dims are replaced by a learned free-space
#                value; any contact  -> all 13 pass through unchanged
# Label        : union of every live contact sensor ("is the hand touching the drill at all").
#                Sensor 6 (thumb_proximal_base) never fires and is excluded from the union.
# Prediction   : action chunk (13) AND future joint torque chunk (13), same UNet
# Loss         : L = L_action + beta * L_torque + gamma * L_contact
#
# This is the mechanism of arXiv:2604.01414 (global gate, learned free-space vector f*), with the
# threshold heuristic replaced by supervision from real contact sensors. Unlike the per-finger
# variant it keeps the ARM's 7 torque dims in the branch, so the load carried once the drill is
# grasped stays observable; the cost is that the 44% of contact frames that are only PARTIAL
# (some fingers on the drill, others still free) open the gate for every joint at once.
#
# Contact sensors are training-only. Inference reads the gate off proprioception alone, so
# deploy_dp3_sim.py needs no tactile sensing and no change.
#
# Compare against: train_..._norobot_contactgate.sh (identical except scope=per_finger)
#
# On gamma
# --------
# Default 0.001 here. Measured on this dataset, the gradient the BCE puts on the gate logits is
# 247x the gradient the diffusion loss puts there, so gamma ~0.004 already balances the two.
# At 0.001 the BCE is roughly 4x WEAKER than the main loss on the gate: the gate is then shaped
# mostly by what helps action prediction, with contact acting as a weak prior rather than a
# command. That is a legitimate choice, but it is also the regime in which arXiv:2604.01414
# reports their learned router collapsing to a near-constant. Watch gate/separation: if it drifts
# towards 0 while contact_loss stalls near ln2 = 0.693, the gate has stopped tracking contact and
# gamma needs raising.
#
# Reading the logs
#   contact_loss     gate BCE (unweighted). 0.693 = learned nothing.
#   gate/separation  mean(phi | contact) - mean(phi | free). THE number. Near 0 = collapsed.
#   gate/std         spread of phi. -> 0 means it became a constant.
#   gate/sep_0       only one head in this scope, so sep_0 == separation.
#
# Usage:
#   bash scripts/train_policy_inspire_drill_grasp_norobot_gateglobal.sh [data_path] [seed] [gpu_id] [epochs] [gamma]

DEBUG=False
save_ckpt=True

task_name=inspire_drill_grasp_norobot_contact
config_name=simple_dp3
data_path=${1:-/home/zeyu/inspire_drill/data/norobot.zarr}
seed=${2:-0}
gpu_id=${3:-0}
num_epochs=${4:-1000}
gamma=${5:-0.001}
aux_beta=${AUX_BETA:-0.1}
addition_info=norobot_gateglobal
zarr_backend=${ZARR_BACKEND:-auto}

EXPECTED_PC=${EXPECTED_PC:-2048}

IFS=',' read -ra _zarr_paths <<< "${data_path}"
for _p in "${_zarr_paths[@]}"; do
    read -r pc_points state_dim has_contact <<< $(/home/zeyu/anaconda3/envs/dp3/bin/python -c "
import zarr; z = zarr.open('${_p}', mode='r')
print(z['data']['point_cloud'].shape[1], z['data']['state'].shape[1],
      1 if 'contact' in z['data'] else 0)")
    if [ "${pc_points}" != "${EXPECTED_PC}" ]; then
        echo -e "\033[31m[ERROR] ${_p} 点云=${pc_points} != ${EXPECTED_PC}(--no_robot 采的纯相机点云)\033[0m"; exit 1
    fi
    if [ "${state_dim}" != "26" ]; then
        echo -e "\033[31m[ERROR] ${_p} state=${state_dim} != 26,采集时必须加 --force_state\033[0m"; exit 1
    fi
    # 没有 contact 就没有门的监督信号,门会退化成自由变量 —— 与其静默跑歪不如直接拒绝
    if [ "${has_contact}" != "1" ]; then
        echo -e "\033[31m[ERROR] ${_p} 没有 data/contact,采集时必须加 --save_contact\033[0m"; exit 1
    fi
done

exp_name="${task_name}-${config_name}-${addition_info}"
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m[Train NoRobot + GLOBAL CONTACT GATE] zarr=${data_path} pc=${pc_points} state=${state_dim} | 1 个全局门管全部 13 维力矩 | beta=${aux_beta} gamma=${gamma} epochs=${num_epochs} gpu=${gpu_id}\033[0m"
if [ "$(printf '%s\n' "${gamma}" | awk '{print ($1 < 0.004) ? 1 : 0}')" = "1" ]; then
    echo -e "\033[33m[NOTE] gamma=${gamma} < 0.004:BCE 在门上的梯度弱于主 loss,门主要由动作目标塑造。\033[0m"
    echo -e "\033[33m       盯住 gate/separation,若趋近 0 且 contact_loss 停在 0.693,说明门已塌成常数。\033[0m"
fi

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
    policy.contact_gate.enabled=true \
    policy.contact_gate.scope=global \
    policy.contact_gate.beta=${gamma} \
    +policy.aux_torque.enabled=true \
    +policy.aux_torque.start=13 \
    +policy.aux_torque.end=26 \
    +policy.aux_torque.beta=${aux_beta} \
    training.num_epochs=${num_epochs}
