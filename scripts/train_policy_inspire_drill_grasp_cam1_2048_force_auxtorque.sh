#!/bin/bash
# Train DP3 with the TA-VLA auxiliary torque objective (arXiv:2509.07962, CoRL 2025, Sec. 5).
#
# Same data, same observation layout, same network as
# train_policy_inspire_drill_grasp_cam1_2048_force.sh -- the ONLY difference is the training
# objective, so the two runs form a clean single-variable ablation.
#
# What the auxiliary objective does
# ---------------------------------
# The diffusion trajectory becomes Z_t = [A_t ; T_t]: the 13-d action chunk concatenated with the
# 13-d joint-torque chunk over the same 16-step horizon, denoised jointly by the SAME UNet through
# the SAME widened in/out conv. TA-VLA is explicit that this must not be a second network or a
# second projection head ("we use a single linear layer instead that outputs concatenated action
# and torque predictions together, then split them back for their respective losses") -- the point
# of the auxiliary task is to shape the SHARED trunk, and a separate network would keep the torque
# gradient in its own parameters where the action branch never sees it.
#
#   L = L_action + beta * L_torque,   beta = 0.1
#
# beta=0.1 FOLLOWS the paper. TA-VLA ablates beta in appendix A.9 (Table 9) on Button Pushing
# over {0.01, 0.1, 0.2, 0.5, 1}, and picks a different value per configuration:
#   pi0+obj       6/20   8/20  10/20   9/20  11/20   -> plateaus above 0.2, they use beta=1
#   pi0+obs+obj  14/20  18/20  18/20  15/20  12/20   -> peaks low, falls high, they use beta=0.1
# Ours is the obs+obj configuration, so 0.1 is the value their sweep selects for it.
# (An earlier version of this comment claimed Table 9 does not exist. It does -- it is in the
# appendix, past the 8 main sections and Tables 1-6, which is what that check had looked at.)
# What the paper DOES show (Table 5, 5 contact-rich tasks x 20 trials) is that torque as an
# observation and torque as an objective are complementary rather than interchangeable:
#   pi0            23/100      pi0+obs   78/100
#   pi0+obj        64/100      pi0+obs+obj  86/100
# i.e. the objective alone is worse than the observation alone, but stacking it on top of the
# observation -- which is our setting -- is what buys the last 78 -> 86.
#
# Cost: zero new data. The dataset already returns the full `horizon` of observations (only the
# first n_obs_steps are consumed for conditioning), so the torque target comes out of
# agent_pos[..., 13:26] of the batch already in memory, normalised by the same LinearNormalizer
# that maps the action to [-1,1]. Nothing about collect_dp3_data.py or deploy_dp3_sim.py changes:
# at inference predict_action keeps returning nsample[..., :13] and the torque channels are
# discarded, so deployment is bit-for-bit the current path.
#
# Torque timestamp (IMPORTANT -- the target is SHIFTED, not index-aligned)
# -----------------------------------------------------------------------
# A zarr row is (obs before action, action): collect_dp3_data.py reads applied_torque AFTER
# env.step() and stores it as the NEXT row's state, so agent_pos[i] holds the torque produced by
# action[i-1]. Verified two ways -- the collect loop appends (prev_state, this-step action), and
# on 200 episodes of finalstage1.zarr the torque correlates better with the PREVIOUS command's
# tracking error, a[k-1]-pos[k], than with a[k]-pos[k] (11/13 joints, |r| 0.648 vs 0.618).
#
# So simple_dp3.compute_loss shifts the torque chunk one step left before concatenating it, i.e.
# slot i is supervised with agent_pos[i+1], the response to action[i]. Without the shift the
# auxiliary task degenerates towards "reproduce the force you already felt" -- and its first
# n_obs_steps frames are verbatim in the conditioning, so they are free to copy.
#
# Supervision therefore covers steps 0..horizon-2 (15 of 16): the last slot's target would be
# agent_pos[horizon], outside the sampled window, so it is padded and dropped from the torque
# loss. One slot (i=0, target agent_pos[1]) is still inside the conditioning window and remains
# copyable; the shift removes the other one.
#
# Reading the logs
# ----------------
#   bc_loss                 action term ONLY -> directly comparable with the no-aux run
#   torque_loss             auxiliary term (unweighted)
#   train/val_action_mse_error  full-sampling action error, 13-d -> comparable with the no-aux run
#   val_loss                bc_loss + 0.1*torque_loss -> NOT comparable with the no-aux run, and it
#                           is what val_best-epoch=*.ckpt selects on. Compare runs by
#                           val_action_mse_error, not by val_loss.
#
# The widened first/last conv means checkpoints are NOT interchangeable with no-aux runs
# (7.7765M -> 7.7882M UNet params). Nothing else in the architecture moves.
#
# Usage:
#   bash scripts/train_policy_inspire_drill_grasp_cam1_2048_force_auxtorque.sh [data_path] [config_name] [seed] [gpu_id] [num_epochs] [beta]
#   data_path 单个: /path/a.zarr   多个: /path/a.zarr,/path/b.zarr

DEBUG=False
save_ckpt=True

task_name=inspire_drill_grasp_cam1_2048_force
data_path=${1:-/home/zeyu/inspire_drill/data/finalstage1.zarr}
config_name=${2:-simple_dp3}
seed=${3:-0}
gpu_id=${4:-0}
num_epochs=${5:-1000}
aux_beta=${6:-0.1}
addition_info=cam1_2048_force_auxtorque
zarr_backend=${ZARR_BACKEND:-auto}   # auto(默认,按内存自动判断) / numpy(强制整包进RAM) / zarr(强制磁盘直读)

# agent_pos 里力矩所在的切片 [start, end)。student_obs.build_agent_pos 的布局是
# [关节位置 13 | 关节力矩 13],所以力矩是 [13, 26)。改 collect 的 state 布局时这里要跟着改。
aux_start=${AUX_START:-13}
aux_end=${AUX_END:-26}

# 与 cam1_2048_force 脚本同样的前置校验:任何一个 zarr 形状不对就在拉起 torch 之前拒绝,
# 避免混进不同代数据导致 state_split / point_cloud 维度不一致的隐蔽错误。
EXPECTED_PC=${EXPECTED_PC:-2560}   # 2560=cam1 2048(cam2 disabled)|robot 512
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

    # 辅助目标就是 state 的 [13,26) 段,所以 state 必须是 26 维:13 维数据下这个目标根本不存在,
    # 而且 state_split 也会因 force_dim<=0 静默回退成单 MLP。
    state_dim=$(/home/zeyu/anaconda3/envs/dp3/bin/python -c "
import zarr; z = zarr.open('${_p}', mode='r')
print(z['data']['state'].shape[1])")
    if [ "${state_dim}" != "26" ]; then
        echo -e "\033[31m[ERROR] ${_p} state=${state_dim} 维 != 26(需含关节力矩)。"
        echo -e "辅助力矩目标取自 state[${aux_start}:${aux_end}],采集时必须加 --force_state\033[0m"; exit 1
    fi
done

exp_name="${task_name}-${config_name}-${addition_info}"
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m[Train InspireDrill-Grasp-Cam1-2048-Force + AUX-TORQUE] task=${task_name} zarr(${#_zarr_paths[@]})=${data_path} pc=${pc_points}(cam1 2048+robot 512) state=${state_dim}(pos13+torque13) aux=state[${aux_start}:${aux_end}] beta=${aux_beta} zarr_backend=${zarr_backend} epochs=${num_epochs} exp=${exp_name} gpu=${gpu_id}\033[0m"

cd 3D-Diffusion-Policy

export PYTHONPATH="${HOME}/3D-Diffusion-Policy:${HOME}/3D-Diffusion-Policy/3D-Diffusion-Policy:${HOME}/3D-Diffusion-Policy/third_party/pytorch3d_simplified"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

# +policy.aux_torque.* 用 hydra 的 append 语法注入,simple_dp3.yaml 里没有这个键,也不需要加 ——
# 不传就是 aux_torque=None,SimpleDP3 走原来那条 13 通道路径,与本改动前逐字节相同。
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
    +policy.aux_torque.start=${aux_start} \
    +policy.aux_torque.end=${aux_end} \
    +policy.aux_torque.beta=${aux_beta} \
    training.num_epochs=${num_epochs}
