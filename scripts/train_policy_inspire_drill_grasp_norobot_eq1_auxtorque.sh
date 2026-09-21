#!/bin/bash
# Eq.1 Torque Gating + 辅助力矩预测 (AUX=on) —— 与 train_policy_inspire_drill_grasp_norobot_eq1.sh
# 配对的那一格。除了打开辅助目标,**其他一切完全相同**。
#
# 为什么是 wrapper 而不是复制一份:
#   eq1.sh 的前处理不是样板 —— 它做了 zarr 元数据探针(点云 2048 / state 26 / 有没有
#   data/contact)、scope 与 gate_pos 的合法性校验、per_finger+feature 的静默降级拦截、
#   run_dir 撞名保护、cd 守卫。复制一份就等于给自己留两份会各自漂移的校验逻辑:改了一边
#   忘了另一边,消融的两格就不再是同一条流水线,而这种偏差不会报错,只会体现在成功率上。
#   这里 exec 过去,所以"前面处理"是同一份代码,不是同一段文字。
#
# 打开 AUX 之后到底变了什么(见 simple_dp3.py):
#   * UNet 的轨迹通道 13 -> 26: Z_t = [A_t ; T_t],A 是 13 维关节目标,T 是 13 维力矩。
#     两者在同一次去噪里被联合预测,损失是 L = L_action + beta * L_torque (TA-VLA 的 Eq.)。
#   * 力矩目标**左移一格**: zarr 的行是 (动作前的观测, 动作),而 applied_torque 是
#     env.step() 之后读的、写进下一行,所以 agent_pos[i] 是对 action[i-1] 的响应。不移位
#     的话辅助任务会退化成"复述你已经感觉到的力",而前 n_obs_steps 帧原样就在 conditioning
#     里 —— 白送,学不到东西。移位后最后一格的真值落在采样窗口外,已被排除出力矩损失。
#   * 推理不受影响: predict_action 取 nsample[..., :Da],多出来的 13 个通道直接丢掉,
#     deploy_dp3_sim.py 拿到的仍是 13 维 —— 现有部署命令原样可用,不需要任何改动。
#
# 观测侧一个字都没动: agent_pos 仍是 26 维,编码器仍看全部 26 维,13 维力矩仍过 force_mlp
# 和 Eq.1 的门。AUX 只增加**预测目标**。所以 task.dataset.torque_start /
# torque_percentile / clamp_agent_pos 照旧由 eq1.sh 下发 —— 那是力矩作为观测的归一化,
# 和预测目标无关。
#
# run_dir 由 eq1.sh 拼成 ..._norobot_eq1_${scope}_${gate_pos}_auxon_seed${seed},与 auxoff
# 那个 run 天然不同名,不会触发 training.resume=True 去加载别人的 latest.ckpt。
#
# 可比性: 只有 **bc_loss** 能跨 AUX 开关比较(它始终只是动作项)。val_loss 不行(项数不同),
# torque_loss 不行(AUX=off 时根本不存在)。另外看 val_action_mse_error /
# gate/separation / gate/mean_free。最终判据仍是 deploy 的成功率。
#
# 预期: 原论文里 "Auxiliary Goals" 是 **baseline 而非其方法**,那一格 28.0%,比纯视觉的
# 30.0% 还低,论文归因于他们预测的是自由段无规律的 external torque。我们的 applied_torque
# 性质更接近 TA-VLA 的 commanded torque(p1/p99 归一化后 std 0.593,与 action 同量级),
# 所以那个失败原因未必成立 —— 这正是这一格要测的,不要预设它更好。
#
# Usage(参数与 eq1.sh 完全一致,原样透传):
#   bash scripts/train_policy_inspire_drill_grasp_norobot_eq1_auxtorque.sh [data_path] [seed] [gpu_id] [epochs] [gamma]
#   AUX_BETA=0.05 bash scripts/..._eq1_auxtorque.sh    # 改辅助项权重(默认 0.1)
#   SCOPE=finger_single bash scripts/..._eq1_auxtorque.sh
#
# 必须从 repo 根目录运行(eq1.sh 里有 cd 守卫)。

set -e

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_base="${_here}/train_policy_inspire_drill_grasp_norobot_eq1.sh"

if [ ! -f "${_base}" ]; then
    echo -e "\033[31m[ERROR] 找不到基脚本: ${_base}\033[0m"
    exit 1
fi

export AUX=on
export AUX_BETA="${AUX_BETA:-0.1}"

echo -e "\033[36m[AUX] 辅助力矩预测已打开 (beta=${AUX_BETA}): UNet 轨迹 [A13;T13]=26 通道,\033[0m"
echo -e "\033[36m      联合预测动作与 13 维关节力矩。其余设置完全沿用 $(basename "${_base}")\033[0m"

exec bash "${_base}" "$@"
