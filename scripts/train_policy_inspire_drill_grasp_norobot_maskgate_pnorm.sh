#!/bin/bash
# 4-CHANNEL POINT CLOUD + contact gate + percentile torque normalization.
#
# train_..._norobot_contactgate.sh 的三处改动叠加版:
#   1. 点云 3 -> 4 通道:[xyz | is_handle],走 slim PointNetEncoderXYZ
#   2. 力矩改用分位数定标 + clamp(下面详述)
#   3. 门的标签可选硬/软(GATE_LABEL,默认 hard,与 contactgate.sh 一致)
# 其余(aux_torque 13:26 beta 0.1、scope=per_finger、手臂 7 维从 force 分支剔除、
# UNet/调度器/epochs)全部保持 contactgate.sh 原样。
#
# ---- 1. 四通道点云 -------------------------------------------------------------------
# 数据要用 --save_mask 采:
#   python scripts/collect_dp3_data.py --stage1_only --disable_cam2 --no_robot \
#       --force_state --save_contact --save_mask --num_episodes 2614 --headless \
#       --output data/norobot_mask.zarr
# zarr 里 point_cloud (T,2048,3) 和 pc_mask (T,2048) 是两个数组,dataset.mask_channel=true
# 时拼成 (T,2048,4)。mask 经 limits normalizer 后是 -1/+1 而不是 0/1。
#
# 两个不能漏的开关:
#   policy.use_pc_color=true
#       simple_dp3 默认 point_cloud[..., :3],不开这个 mask 通道会被静默丢掉,采了白采。
#   +policy.pointcloud_encoder_cfg.slim_encoder=true
#       不开的话 4 通道被路由到 PointNetEncoderXYZRGB([64,128,256,512]),encoder 容量跟着
#       变,mask 的消融就混进了容量变量。slim 走 PointNetEncoderXYZ([64,128,256]),只有第
#       一层 Linear 从 3->4(+64 参数),与 3 通道基线容量一致。
#
# ---- 2. 力矩归一化(实测于 norobot.zarr,435781 帧)-----------------------------------
# agent_pos[13:26] 是关节力矩,它的 min/max 恰好是执行器的 effort limit(±87/±40/±10 Nm),
# 而只有 <=0.93% 的帧顶到过。拿 min/max 定标会把 98% 的力矩挤进 [-1,1] 的 12%:
#       归一化后 std   action 0.522   pos13 0.394   torque 0.106
# 扩散里每个通道 SNR_t = alpha_bar_t * sigma^2 / (1-alpha_bar_t),sigma 小 5 倍就是 SNR 低
# 25 倍:squaredcos_cap_v2 / 100 步下,力矩通道在 t=5 就掉到 SNR<1,而动作通道撑到 t=29。
# 也就是 ~95% 的训练步里力矩那半条轨迹与纯噪声无异,L_action + beta*L_torque 里力矩项占总
# 损失只有 0.4%(beta=0.1 会让你以为是 ~9%)。改用 p1/p99 定标后 sigma≈0.41,与 pos/action
# 同量级。这是逐维做的 —— 13 维力矩各自用自己的分位数,因为每一维都是"峰在 0 附近 + 稀疏
# 长尾"的形状,min/max 对这种分布必然失效。
#
# 代价:~2% 的饱和帧落到 [-1,1] 外(最大 ±8.9),MSE 是平方的,这 2% 会拿走约 11% 的力矩
# 梯度,而 clip_sample=True 意味着采样时根本输不出这种值。policy.clamp_agent_pos=true 夹回
# ±1,等价于在 p1/p99 处 winsorize。丢掉的信息只有"顶轨顶了多深"。
# 两个开关必须同开:只设 torque_start 而漏了 clamp_agent_pos,SimpleDP3 会在前 50 个 batch
# 打红字告警。
#
# ---- 3. 门的标签 ---------------------------------------------------------------------
# GATE_LABEL=hard(默认):(contact_force > 0.01) 的 0/1,与 contactgate.sh 一致。
# GATE_LABEL=soft       :clip(log1p(|F|)/log1p(c_ref), 0, 1),c_ref 取每组接触时力幅值的
#                        p90。门的前向本来就是连续 sigmoid,把它逼成二值的一直是硬标签。
# 注意 contact_loss 在软/硬之间不可比(软目标的 BCE 下界是目标熵)。gate/separation、
# gate/sep_*、gate/pos_rate 两边都按硬阈值算,可比。软模式多一个 gate/in_contact_std:
# 贴着 0 就说明门在软目标下又塌回二值了。
#
# ---- 对照组 --------------------------------------------------------------------------
#   3 通道 + 新归一化 + 硬门   train_..._norobot_gatehard_pnorm.sh      <- 本脚本的 mask 对照
#   3 通道 + 新归一化 + 软门   train_..._norobot_gatesoft_pnorm.sh
#   3 通道 + 旧归一化 + 硬门   train_..._norobot_contactgate.sh
# 本脚本 vs gatehard_pnorm 只差点云通道数,是 mask 的单变量测试。
# val_loss 跨 cell 一律不可比;看 bc_loss / val_action_mse_error / deploy 成功率。
#
# Usage:
#   bash scripts/train_policy_inspire_drill_grasp_norobot_maskgate_pnorm.sh [data_path] [seed] [gpu_id] [epochs] [gamma]
#   GATE_LABEL=soft bash scripts/train_policy_inspire_drill_grasp_norobot_maskgate_pnorm.sh

DEBUG=False
save_ckpt=True

task_name=inspire_drill_grasp_norobot_maskch_contact
config_name=simple_dp3
data_path=${1:-/home/zeyu/inspire_drill/data/norobot_mask.zarr}
seed=${2:-0}
gpu_id=${3:-0}
num_epochs=${4:-1000}
gamma=${5:-0.1}
aux_beta=${AUX_BETA:-0.1}
torque_pct=${TORQUE_PCT:-1.0}          # p1/p99;设 0.5 走更保守的 p0.5/p99.5
gate_label=${GATE_LABEL:-hard}         # hard | soft
zarr_backend=${ZARR_BACKEND:-auto}

EXPECTED_PC=${EXPECTED_PC:-2048}       # 2048 = cam1 alone (--disable_cam2 --no_robot)

case "${gate_label}" in
    hard) soft_flag=false ;;
    soft) soft_flag=true  ;;
    *) echo -e "\033[31m[ERROR] GATE_LABEL 只能是 hard 或 soft,收到 '${gate_label}'\033[0m"; exit 1 ;;
esac
addition_info=norobot_maskgate_pnorm_${gate_label}

IFS=',' read -ra _zarr_paths <<< "${data_path}"
for _p in "${_zarr_paths[@]}"; do
    read -r pc_points state_dim has_contact has_mask mask_pts <<< $(/home/zeyu/anaconda3/envs/dp3/bin/python -c "
import zarr; z = zarr.open('${_p}', mode='r')
d = z['data']
print(d['point_cloud'].shape[1], d['state'].shape[1],
      1 if 'contact' in d else 0,
      1 if 'pc_mask' in d else 0,
      d['pc_mask'].shape[1] if 'pc_mask' in d else -1)")
    if [ "${pc_points}" != "${EXPECTED_PC}" ]; then
        echo -e "\033[31m[ERROR] ${_p} 点云=${pc_points} != ${EXPECTED_PC}。本脚本要的是 --disable_cam2 --no_robot 采的纯 cam1 点云\033[0m"; exit 1
    fi
    if [ "${state_dim}" != "26" ]; then
        echo -e "\033[31m[ERROR] ${_p} state=${state_dim} != 26,采集时必须加 --force_state\033[0m"; exit 1
    fi
    if [ "${has_contact}" != "1" ]; then
        echo -e "\033[31m[ERROR] ${_p} 没有 data/contact。门控的 BCE 监督取自它,采集时必须加 --save_contact\033[0m"; exit 1
    fi
    # 没有 pc_mask 就没有第 4 通道。task yaml 声明的是 (2048,4),dataset 会在拼通道时直接
    # KeyError —— 但那是几分钟后加载完 zarr 才炸,不如现在就拦住
    if [ "${has_mask}" != "1" ]; then
        echo -e "\033[31m[ERROR] ${_p} 没有 data/pc_mask。第 4 通道(is_handle)取自它,采集时必须加 --save_mask\033[0m"; exit 1
    fi
    if [ "${mask_pts}" != "${pc_points}" ]; then
        echo -e "\033[31m[ERROR] ${_p} pc_mask 点数=${mask_pts} != 点云=${pc_points},数据不自洽\033[0m"; exit 1
    fi
done

exp_name="${task_name}-${config_name}-${addition_info}"
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m[Train NoRobot + MASK 4ch + AUX-TORQUE + ${gate_label^^} GATE + p${torque_pct} 力矩归一化] zarr=${data_path} pc=${pc_points}x4(cam1 only + is_handle) state=${state_dim}(pos13+torque13) contact=13 beta=${aux_beta} gamma=${gamma} epochs=${num_epochs} gpu=${gpu_id}\033[0m"

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
    policy.use_pc_color=true \
    +policy.pointcloud_encoder_cfg.slim_encoder=true \
    +policy.aux_torque.enabled=true \
    +policy.aux_torque.start=13 \
    +policy.aux_torque.end=26 \
    +policy.aux_torque.beta=${aux_beta} \
    policy.contact_gate.enabled=true \
    policy.contact_gate.beta=${gamma} \
    policy.contact_gate.soft_label=${soft_flag} \
    training.num_epochs=${num_epochs}
