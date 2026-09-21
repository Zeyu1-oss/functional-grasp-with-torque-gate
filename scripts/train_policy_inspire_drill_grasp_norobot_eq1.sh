
save_ckpt=True

task_name=inspire_drill_grasp_norobot_contact
config_name=simple_dp3
data_path=${1:-/home/zeyu/inspire_drill/data/norobot.zarr}
seed=${2:-0}
gpu_id=${3:-0}
num_epochs=${4:-1000}
gamma=${5:-0.1}
torque_pct=${TORQUE_PCT:-1.0}
scope=${SCOPE:-global}                 # global | finger_single | per_finger
gate_pos=${GATE_POS:-feature}          # feature | input
aux=${AUX:-off}                        # off | on
aux_beta=${AUX_BETA:-0.1}
aux_start=${AUX_START:-13}             # agent_pos slice start for the aux torque TARGET (13=all 13 joints, 20=hand-only 6)
zarr_backend=${ZARR_BACKEND:-auto}

EXPECTED_PC=${EXPECTED_PC:-2048}

case "${scope}" in
    global|finger_single|per_finger) ;;
    *) echo -e "\033[31m[ERROR] SCOPE 只能是 global | finger_single | per_finger,收到 '${scope}'\033[0m"; exit 1 ;;
esac
case "${gate_pos}" in
    feature|input) ;;
    *) echo -e "\033[31m[ERROR] GATE_POS 只能是 feature 或 input,收到 '${gate_pos}'\033[0m"; exit 1 ;;
esac
case "${aux}" in
    on|off) ;;
    *) echo -e "\033[31m[ERROR] AUX 只能是 on 或 off,收到 '${aux}'\033[0m"; exit 1 ;;
esac
# per_finger + feature 会被 DP3Encoder 静默降级成单门(64 维特征切不成 6 份),那不是本脚本
# 想跑的任何一格 —— 与其静默跑歪不如直接拒绝
if [ "${scope}" = "per_finger" ] && [ "${gate_pos}" = "feature" ]; then
    echo -e "\033[31m[ERROR] per_finger + feature: 64 维特征无法切成 6 份,编码器会静默降级成单门。\033[0m"
    echo -e "\033[31m        要单门就用 SCOPE=global(全 13 维力矩)或 finger_single(仅手指 6 维)。\033[0m"
    exit 1
fi

# run_dir 必须带全标识,否则会撞上已有的 run 并触发 training.resume=True 去加载它的
# latest.ckpt —— scope 变了 force_mlp/gate_mlp 的 shape 就变了,那会直接报错白等一轮。
addition_info=norobot_eq1_${scope}_${gate_pos}_aux${aux}
# gamma (contact_gate.beta) and AUX_BETA are NOT in the name above, so two runs that differ only
# in one of those weights land in the SAME run_dir and the second one resumes from the first's
# latest.ckpt -- silently, since the shapes match. That has already produced one confusing pair
# (an AUX_BETA=0.5 run sitting in a directory named like the 0.1 ones). Append whichever weight
# departs from its default; runs at the defaults keep their existing names untouched.
if [ "${gamma}" != "0.1" ]; then
    addition_info="${addition_info}_gamma${gamma}"
fi
if [ "${aux}" = "on" ] && [ "${aux_beta}" != "0.1" ]; then
    addition_info="${addition_info}_beta${aux_beta}"
fi
if [ "${aux}" = "on" ] && [ "${aux_start}" != "13" ]; then
    addition_info="${addition_info}_auxstart${aux_start}"
fi

IFS=',' read -ra _zarr_paths <<< "${data_path}"
for _p in "${_zarr_paths[@]}"; do
    read -r pc_points state_dim has_contact <<< $(/home/zeyu/anaconda3/envs/dp3/bin/python -c "
import zarr; z = zarr.open('${_p}', mode='r')
print(z['data']['point_cloud'].shape[1], z['data']['state'].shape[1],
      1 if 'contact' in z['data'] else 0)")
    # 探针失败(路径不存在/不是 zarr)时 $() 为空,三个变量都是空串,下面的检查会报出
    # "点云= != 2048" 这种指向错误原因的信息 —— 先把真正的原因拦下来
    if [ -z "${pc_points}" ]; then
        echo -e "\033[31m[ERROR] 读不出 ${_p} 的元数据。检查路径是否存在、是否是 zarr 目录\033[0m"; exit 1
    fi
    if [ "${pc_points}" != "${EXPECTED_PC}" ]; then
        echo -e "\033[31m[ERROR] ${_p} 点云=${pc_points} != ${EXPECTED_PC}。本脚本要的是 --no_robot 采的纯相机点云\033[0m"; exit 1
    fi
    if [ "${state_dim}" != "26" ]; then
        echo -e "\033[31m[ERROR] ${_p} state=${state_dim} != 26,采集时必须加 --force_state\033[0m"; exit 1
    fi
    # 没有 contact 就没有门的监督信号,phi 会退化成自由变量 —— 与其静默跑歪不如直接拒绝
    if [ "${has_contact}" != "1" ]; then
        echo -e "\033[31m[ERROR] ${_p} 没有 data/contact。门的 BCE 监督取自它,采集时必须加 --save_contact\033[0m"; exit 1
    fi
done

exp_name="${task_name}-${config_name}-${addition_info}"
run_dir="data/outputs/${exp_name}_seed${seed}"

if [ "${scope}" = "global" ]; then
    _tau_desc="全 13 维(手臂7+手指6)"
elif [ "${scope}" = "finger_single" ]; then
    _tau_desc="仅手指 6 维(手臂被丢弃)"
else
    _tau_desc="仅手指 6 维,6 个门"
fi
_aux_desc="${aux}"
if [ "${aux}" = "on" ]; then
    if [ "${aux_start}" = "13" ]; then _aux_desc="on(全13维)"; else _aux_desc="on(agent_pos[${aux_start}:26])"; fi
fi
echo -e "\033[33m[Eq.1 Torque Gating] scope=${scope} 力矩支路输入=${_tau_desc} | 门=${gate_pos} | 辅助力矩预测=${_aux_desc}\033[0m"
echo -e "\033[33m  zarr=${data_path} pc=${pc_points} state=${state_dim}(pos13+torque13) p${torque_pct}力矩归一化 gamma=${gamma} epochs=${num_epochs} gpu=${gpu_id}\033[0m"
echo -e "\033[33m  run_dir=${run_dir}\033[0m"

# 必须从 repo 根目录跑: train.py 在 3D-Diffusion-Policy/ 下,而 hydra.run.dir 是相对路径。
# 不加守卫的话 cd 失败会**继续执行**,拿当前目录的 train.py 跑、输出落到错误的地方。
cd 3D-Diffusion-Policy || {
    echo -e "\033[31m[ERROR] 进不去 3D-Diffusion-Policy/ —— 本脚本必须从 repo 根目录运行:\033[0m"
    echo -e "\033[31m        cd ${HOME}/3D-Diffusion-Policy && bash scripts/$(basename "$0")\033[0m"
    exit 1
}

export PYTHONPATH="${HOME}/3D-Diffusion-Policy:${HOME}/3D-Diffusion-Policy/3D-Diffusion-Policy:${HOME}/3D-Diffusion-Policy/third_party/pytorch3d_simplified"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

# 辅助力矩目标: AUX=off 时这几个 override 整组不下发。simple_dp3.yaml 里没有 aux_torque 这个
# key(所以上面用的是 '+'),不传 = SimpleDP3(aux_torque=None) -> aux_dim=0,UNet 轨迹回到 13 通道。
_aux_args=()
if [ "${aux}" = "on" ]; then
    _aux_args=(
        +policy.aux_torque.enabled=true
        +policy.aux_torque.start=${aux_start}
        +policy.aux_torque.end=26
        +policy.aux_torque.beta=${aux_beta}
    )
fi

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
    "${_aux_args[@]}" \
    policy.contact_gate.enabled=true \
    policy.contact_gate.scope=${scope} \
    policy.contact_gate.gate_position=${gate_pos} \
    policy.contact_gate.beta=${gamma} \
    policy.contact_gate.soft_label=false \
    training.num_epochs=${num_epochs}
