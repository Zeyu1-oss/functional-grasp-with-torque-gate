#!/bin/bash
# GRADED contact gate + percentile torque normalization.
#
# Differs from train_..._norobot_gatehard_pnorm.sh in exactly one flag:
# policy.contact_gate.soft_label=true. Run both and the pair is a clean single-variable test of
# the label; everything else (normalization, aux torque, scope, beta/gamma) is identical.
#
# What "graded" changes. The gate's forward pass was ALREADY continuous --
#     phi = sigmoid(gate_mlp(...));  tau_in = phi*tau + (1-phi)*tau_free
# -- so the network could always express a partial gate. What pinned it to 0/1 was the
# supervision: BCE against (contact_force > 0.01) has its optimum at a saturated sigmoid.
# Measured on norobot.zarr, in-contact force spans more than a decade per finger:
#       group    pos_rate   p10      p50      p90     max
#       index      0.420    4.23    22.59    59.73   86.60
#       middle     0.483    2.59    19.74    45.50   86.60
#       pinky      0.369    4.49    13.86    36.48   86.60
#       ring       0.399    8.28    21.91    39.69   86.60
#       thumb      0.514    4.13    28.79    67.13   86.60
# The hard label maps all of that onto 1, so a fingertip barely brushing the drill and a finger
# bearing the whole weight open the gate equally. The graded target is
#     clip(log1p(|F|) / log1p(c_ref), 0, 1),  c_ref = that group's p90 above
# so no contact -> 0, p90-strength contact -> 1, and everything between is graded. log rather
# than linear compression: linearly, a ~3 N touch would sit at 0.05, indistinguishable from zero.
#
# Reading the metrics:
#   contact_loss  NOT comparable with the hard run. BCE against a soft target floors at the
#                 target's entropy, not at 0, so a higher number here means nothing by itself.
#   gate/separation, gate/sep_*, gate/pos_rate  computed against the HARD threshold in both
#                 runs on purpose, so these ARE comparable.
#   gate/in_contact_std  new, soft mode only. Spread of phi within contact frames. Near 0 means
#                 the gate re-collapsed to binary despite the graded target -- the thing this
#                 run exists to detect.
#   bc_loss, val_action_mse_error  the action-side numbers; comparable across all cells.
#   val_loss      never comparable across cells (it carries beta*torque + gamma*contact).
#
# For the normalization change and why clamp_agent_pos must be on with it, see the header of
# train_policy_inspire_drill_grasp_norobot_gatehard_pnorm.sh.
#
# Usage:
#   bash scripts/train_policy_inspire_drill_grasp_norobot_gatesoft_pnorm.sh [data_path] [seed] [gpu_id] [epochs] [gamma]

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
torque_pct=${TORQUE_PCT:-1.0}     # keep identical to the hard-gate control run
addition_info=norobot_gatesoft_pnorm
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
    if [ "${has_contact}" != "1" ]; then
        echo -e "\033[31m[ERROR] ${_p} 没有 data/contact。软标签的力幅值取自它,采集时必须加 --save_contact\033[0m"; exit 1
    fi
done

exp_name="${task_name}-${config_name}-${addition_info}"
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m[Train NoRobot + AUX-TORQUE + SOFT GATE + p${torque_pct} 力矩归一化] zarr=${data_path} pc=${pc_points}(cam1 only) state=${state_dim}(pos13+torque13) contact=13 beta=${aux_beta} gamma=${gamma} epochs=${num_epochs} gpu=${gpu_id}\033[0m"

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
    +task.dataset.torque_start=13 \
    +task.dataset.torque_percentile=${torque_pct} \
    policy.clamp_agent_pos=true \
    +policy.aux_torque.enabled=true \
    +policy.aux_torque.start=13 \
    +policy.aux_torque.end=26 \
    +policy.aux_torque.beta=${aux_beta} \
    policy.contact_gate.enabled=true \
    policy.contact_gate.beta=${gamma} \
    policy.contact_gate.soft_label=true \
    training.num_epochs=${num_epochs}
