import copy

import numpy as np
import psutil
import torch
import zarr

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.dataset.realdex_dataset import RealDexDataset
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer)


def _zarr_nbytes(zarr_path, keys):
    """只读 zarr 的 array 元数据(.zarray 里的 shape/dtype),不解压任何 chunk,
    估算这些 key 整包解压进 RAM 需要多少字节。"""
    group = zarr.open(zarr_path, mode='r')
    return sum(int(np.prod(group['data'][k].shape)) * group['data'][k].dtype.itemsize
               for k in keys if k in group['data'])


class InspireDrillDataset(RealDexDataset):
    """Dataset loader for the InspireDrill task.

    Data format (produced by collect_dp3_data.py):
      - state:   (T, 13)  13 controlled joint positions (7 arm + 6 hand)
      - action:  (T, 13)  7 arm joints + 6 hand joints
      - point_cloud: (T, total_pc, 3)  fused env-local point cloud
                     (camera + robot-FK + drill-oracle + ground)

    Zarr layout:
      /data/state         (N_steps, 13) float32
      /data/action        (N_steps, 13) float32
      /data/point_cloud   (N_steps, total_pc, 3) float32
      /meta/episode_ends  (N_episodes,) int64  cumulative lengths

    内存:与官方数据集一致,copy_from_path 整包解压进 RAM(当前数据集 ~8 GB,放得下)。
    get_normalizer 的 point_cloud 仍用分块流式统计(结果与官方 limits fit 等价,
    且避免 reshape 再复制一份大数组)。

    多 zarr 混训:zarr_path 传列表(hydra CLI: task.dataset.zarr_path='[a.zarr,b.zarr]')
    即可把多个同布局 zarr(如 串联全量 + collect --stage1_only 抓取加强)直接混训,
    等价于 merge_zarr.py 合并后单 zarr,但不用落盘拷贝。
    """

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.05,
        max_train_episodes=None,
        task_name=None,
        agent_pos_dim=None,
        mask_channel=False,
        load_contact=False,
        zarr_backend='auto',
        ram_budget_frac=0.5,
        torque_start=None,
        torque_percentile=1.0,
    ):
        # zarr_backend: 'auto'(默认)/'numpy'/'zarr'。
        #   numpy = ReplayBuffer.copy_from_path,整包解压进 RAM,快但吃内存。
        #   zarr  = ReplayBuffer.create_from_path,磁盘直读(zarr 自带 chunk 级解压缓存),
        #           慢一些但不会把点云整个搬进内存,数据比内存大时用这个。
        #   auto  = 按 psutil 当前可用内存 vs 所有 zarr 的 state/action/point_cloud 估算字节数
        #           (只读 .zarray 元数据,不解压)比较,总量超过 ram_budget_frac * 可用内存时
        #           整批退化成 zarr 直读,否则整批走 numpy(混着来意义不大,统一判断更好预测)。
        # 2026-07-27 实测:这台机器 15GB 内存,单个 13.78GB 的 zarr 用 numpy backend 直接被系统
        # OOM killer 杀掉训练进程(dmesg: Out of memory, anon-rss:12466260kB)。
        # mask_channel: True 时把 zarr 的 data/pc_mask 作为点云第 4 通道拼进 point_cloud
        #   -> obs['point_cloud'] 变 (T, N, 4) = [xyz | is_handle]。需 shape_meta.point_cloud.shape=[N,4]
        #   且 policy.use_pc_color=True(保留全部通道)。zarr 必须是 --save_mask 采的(含 pc_mask)。
        # load_contact: 额外读 zarr 的 data/contact (T,13) 每个手部 link 与电钻的接触力幅值,
        #   作为 batch['contact'] 返回,供 policy.contact_gate 的 BCE 监督用。它是特权信息,
        #   只进损失不进网络输入,所以 shape_meta 不需要声明,deploy 也不需要它。
        #   zarr 必须是 --save_contact 采的。
        # torque_start / torque_percentile: 见 _agent_pos_normalizer。None = 关闭,
        #   agent_pos 全维走官方的 min/max limits fit(旧行为,逐字节一致)。
        self.mask_channel = bool(mask_channel)
        self.load_contact = bool(load_contact)
        self.torque_start = None if torque_start is None else int(torque_start)
        self.torque_percentile = float(torque_percentile)
        _keys = ['state', 'action', 'point_cloud'] + (['pc_mask'] if mask_channel else []) \
            + (['contact'] if load_contact else [])
        # agent_pos_dim: 只取 state 的前 N 维作为 agent_pos(None=全取)。
        # 用途:zarr 用 --force_state 存了 26 维 [pos13|force13],训练时想只用位置
        # 13 维(不重采数据),在 task yaml 里设 dataset.agent_pos_dim: 13 并把
        # shape_meta.obs.agent_pos.shape 改成 [13] 即可;deploy 会从 ckpt 自适应。
        # Skip RealDexDataset.__init__ to avoid hardcoded 'img' key requirement.
        # Replicate the needed setup directly.
        from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
        from diffusion_policy_3d.common.sampler import (
            SequenceSampler, get_val_mask, downsample_mask)

        self.task_name = task_name
        # zarr_path: 单个路径 或 路径列表(如 [串联全量, stage1_only 抓取加强])。
        # 多 zarr 直接混训,免去 merge_zarr.py 落盘一份拷贝;各 zarr 布局必须同代
        # (点云段序/点数、state/action 维度一致),否则下面 assert 拦住。
        # hydra CLI: task.dataset.zarr_path='[a.zarr,b.zarr]'
        if isinstance(zarr_path, str):
            paths = [zarr_path]
        else:
            paths = [str(p) for p in zarr_path]   # list/tuple/ListConfig
        # 官方同款:整包解压进 RAM(只取训练用的 3 个 key,忽略 privileged 等)。
        # fp16 存的旧 zarr 进内存后仍是 fp16(省一半 RAM),_sample_to_data 读时转 fp32。
        # -- 除非 zarr_backend 判定装不下,那就整批退化成磁盘直读(见上面 zarr_backend 的注释)。
        if zarr_backend == 'auto':
            total_bytes = sum(_zarr_nbytes(p, _keys) for p in paths)
            avail_bytes = psutil.virtual_memory().available
            use_numpy = total_bytes <= ram_budget_frac * avail_bytes
            print(f"[InspireDrillDataset] zarr_backend=auto: 待加载 {total_bytes/1e9:.2f}GB, "
                  f"当前可用内存 {avail_bytes/1e9:.2f}GB, 预算 {ram_budget_frac*avail_bytes/1e9:.2f}GB "
                  f"-> 选择 {'numpy(整包进RAM)' if use_numpy else 'zarr(磁盘直读, 不会OOM但更慢)'}",
                  flush=True)
        else:
            use_numpy = (zarr_backend == 'numpy')
        if use_numpy:
            self.replay_buffers = [
                ReplayBuffer.copy_from_path(p, keys=_keys)
                for p in paths]
        else:
            self.replay_buffers = [
                ReplayBuffer.create_from_path(p)
                for p in paths]
        _ref = self.replay_buffers[0]
        for p, rb in zip(paths, self.replay_buffers):
            for k in ('state', 'action', 'point_cloud'):
                if rb[k].shape[1:] != _ref[k].shape[1:]:
                    raise ValueError(
                        f"{p} 的 {k} 形状 {rb[k].shape[1:]} != {_ref[k].shape[1:]}"
                        f"(不同代数据不可混训)")

        # 每个 zarr 独立划 train/val(val_ratio 逐 zarr 生效)并建 sampler;
        # max_train_episodes 逐 zarr 上限(当前配置为 null,不受影响)。
        self.samplers, self.train_masks = [], []
        for rb in self.replay_buffers:
            val_mask = get_val_mask(
                n_episodes=rb.n_episodes, val_ratio=val_ratio, seed=seed)
            train_mask = downsample_mask(
                mask=~val_mask, max_n=max_train_episodes, seed=seed)
            self.samplers.append(SequenceSampler(
                replay_buffer=rb, sequence_length=horizon,
                pad_before=pad_before, pad_after=pad_after,
                episode_mask=train_mask))
            self.train_masks.append(train_mask)

        # 启动自证:训练日志里直接看到加载了哪些 zarr、各多少数据(排查"到底用没用两个zarr")
        print(f"[InspireDrillDataset] 加载 {len(paths)} 个 zarr:", flush=True)
        for p, rb, m in zip(paths, self.replay_buffers, self.train_masks):
            print(f"  - {p}: {rb.n_episodes} episodes ({int(m.sum())} train / "
                  f"{int((~m).sum())} val), {rb['action'].shape[0]} 帧", flush=True)

        # 单 zarr 兼容别名(外部代码可能读 .replay_buffer/.sampler/.train_mask)
        self.replay_buffer = self.replay_buffers[0]
        self.sampler = self.samplers[0]
        self.train_mask = self.train_masks[0]
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.agent_pos_dim = agent_pos_dim

    def __len__(self) -> int:
        return sum(len(s) for s in self.samplers)

    def __getitem__(self, idx: int):
        # 多 sampler 索引拼接:[0, len(s0)) -> s0, [len(s0), len(s0)+len(s1)) -> s1 ...
        for s in self.samplers:
            n = len(s)
            if idx < n:
                sample = s.sample_sequence(idx)
                break
            idx -= n
        else:
            raise IndexError(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)

    def get_validation_dataset(self):
        # 覆盖 RealDex 版本:每个 zarr 各自取 ~train_mask 的 episode 做验证集
        val_set = copy.copy(self)
        from diffusion_policy_3d.common.sampler import SequenceSampler
        val_set.samplers = [
            SequenceSampler(replay_buffer=rb, sequence_length=self.horizon,
                            pad_before=self.pad_before, pad_after=self.pad_after,
                            episode_mask=~m)
            for rb, m in zip(self.replay_buffers, self.train_masks)]
        val_set.train_masks = [~m for m in self.train_masks]
        val_set.sampler = val_set.samplers[0]
        val_set.train_mask = val_set.train_masks[0]
        return val_set

    def _sample_to_data(self, sample):
        # 覆盖 RealDex 版本:支持 agent_pos_dim 切片(如 26 维 state 只取前 13 位关节位置)
        agent_pos = sample['state'][:, ].astype(np.float32)
        if self.agent_pos_dim is not None:
            agent_pos = agent_pos[..., :int(self.agent_pos_dim)]
        pc = sample['point_cloud'][:, ].astype(np.float32)          # (T, N, 3)
        if self.mask_channel:
            m = sample['pc_mask'][:, ].astype(np.float32)[..., None]  # (T, N, 1)
            pc = np.concatenate([pc, m], axis=-1)                   # (T, N, 4) = [xyz | is_handle]
        data = {
            'obs': {
                'point_cloud': pc,
                'agent_pos': agent_pos,
            },
            'action': sample['action'].astype(np.float32),
        }
        if self.load_contact:
            # top level, not under 'obs': it is a training-only label, never an encoder input.
            data['contact'] = sample['contact'][:, ].astype(np.float32)
        return data

    @staticmethod
    def _limits_normalizer_from_stats(input_min, input_max, input_mean, input_std,
                                      output_min=-1.0, output_max=1.0, range_eps=1e-4):
        """复刻 normalizer._fit 的 mode='limits', fit_offset=True 数学,
        但 min/max/mean/std 由外部流式统计传入(避免整块 [:] 读入内存)。"""
        input_min = input_min.clone().float()
        input_max = input_max.clone().float()
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
        input_stats = {
            'min': input_min, 'max': input_max,
            'mean': input_mean.float(), 'std': input_std.float(),
        }
        return SingleFieldLinearNormalizer.create_manual(scale, offset, input_stats)

    def _agent_pos_normalizer(self, state, mode='limits', **kwargs):
        """agent_pos 的 limits normalizer。torque_start=None 时就是官方的 min/max fit。

        给定 torque_start 时,只有 [torque_start:] 这一段(力矩)改用
        p / (100-p) 分位数定标,前面的关节位置仍用 min/max。

        为什么:力矩的 min/max 恰好是执行器的 effort limit(±87/±40/±10 Nm),而只有
        ≤0.8% 的帧顶到过这些轨。拿它定标会把 98% 的力矩挤进 [-1,1] 的 12%——实测
        归一化后 std 0.106,而 action 0.522 / pos13 0.394。扩散里每个通道的
        SNR_t = ᾱ_t·σ²/(1-ᾱ_t),σ 小 5 倍就是 SNR 低 25 倍:力矩通道在 t>5 就掉到
        SNR<1,也就是 95% 的训练步里与纯噪声无异,L_action + beta·L_torque 里力矩项
        的实际权重比 beta 小约 24 倍。改成 p1/p99 后 σ≈0.44,与 action/pos 同量级。

        代价:约 2% 的饱和帧会落到 [-1,1] 之外(最大 ±8.9)。由 policy 的
        clamp_agent_pos 夹回去 —— 夹在 policy 而不是这里,是因为 normalizer 存进 ckpt
        后 deploy 会原样加载,clamp 写在 policy 里训练和部署才走同一条路径。
        两个开关必须同开;漏开时 SimpleDP3 会在前若干步打红字告警。
        """
        k = self.torque_start
        if k is None:
            return SingleFieldLinearNormalizer.create_fit(state, mode=mode, **kwargs)
        if k >= state.shape[1]:
            # agent_pos_dim=13 之类:力矩块被切掉了,没有可分位定标的东西
            print(f"[InspireDrillDataset] torque_start={k} >= agent_pos 维度 "
                  f"{state.shape[1]},没有力矩块 -> 全维退回 min/max", flush=True)
            return SingleFieldLinearNormalizer.create_fit(state, mode=mode, **kwargs)

        p = self.torque_percentile
        t = torch.from_numpy(np.asarray(state, dtype=np.float32))
        lim_min = t.min(dim=0).values.clone()
        lim_max = t.max(dim=0).values.clone()
        lim_min[k:] = torch.from_numpy(
            np.percentile(state[:, k:], p, axis=0).astype(np.float32))
        lim_max[k:] = torch.from_numpy(
            np.percentile(state[:, k:], 100.0 - p, axis=0).astype(np.float32))

        # 自证:打出改动前后力矩的归一化 std,训练日志里一眼能看出定标是否真的生效
        _raw = state[:, k:]
        _span_old = np.maximum(_raw.max(0) - _raw.min(0), 1e-4)
        _span_new = np.maximum((lim_max[k:] - lim_min[k:]).numpy(), 1e-4)
        _std_raw = _raw.std(0)
        _clipped = np.clip(2 * (_raw - lim_min[k:].numpy()) / _span_new - 1, -1, 1)
        print(f"[InspireDrillDataset] 力矩块 [{k}:{state.shape[1]}] 改用 p{p}/p{100 - p} 定标: "
              f"归一化 std {float((2 * _std_raw / _span_old).mean()):.3f} -> "
              f"{float(_clipped.std(0).mean()):.3f} (clamp 后), "
              f"越界帧 {float((np.abs(2 * (_raw - lim_min[k:].numpy()) / _span_new - 1) > 1).mean()) * 100:.2f}% "
              f"(需 policy.clamp_agent_pos=true)", flush=True)

        return self._limits_normalizer_from_stats(lim_min, lim_max, t.mean(dim=0), t.std(dim=0))

    def _streaming_pc_normalizer(self, block_frames=4096):
        """分块扫 point_cloud(所有 zarr),逐 xyz 维统计 min/max/mean/std。
        结果与官方 limits fit 等价;分块只是避免 reshape 时再复制一份大数组。"""
        run_min = run_max = None
        D = int(self.replay_buffers[0]['point_cloud'].shape[-1])   # 3
        s = torch.zeros(D, dtype=torch.float64)
        ss = torch.zeros(D, dtype=torch.float64)
        n = 0
        for rb in self.replay_buffers:
            pc = rb['point_cloud']                      # zarr (T, P, 3),磁盘
            T = int(pc.shape[0])
            for start in range(0, T, block_frames):
                arr = pc[start:start + block_frames]    # numpy (b, P, 3),仅这一块解压
                t = torch.from_numpy(np.asarray(arr, dtype=np.float32)).reshape(-1, D)
                bmin = t.min(dim=0).values
                bmax = t.max(dim=0).values
                run_min = bmin if run_min is None else torch.minimum(run_min, bmin)
                run_max = bmax if run_max is None else torch.maximum(run_max, bmax)
                td = t.double()
                s += td.sum(dim=0)
                ss += (td * td).sum(dim=0)
                n += t.shape[0]
        mean = (s / max(n, 1)).float()
        var = (ss / max(n, 1)).float() - mean * mean
        std = var.clamp_min(0).sqrt()
        if self.mask_channel:
            # 第 4 通道(is_handle,0/1):min=0 max=1,归一化后 0->-1、1->+1。
            frac = float(np.mean([np.asarray(rb['pc_mask'][:]).mean()
                                  for rb in self.replay_buffers]))
            run_min = torch.cat([run_min, torch.tensor([0.0])])
            run_max = torch.cat([run_max, torch.tensor([1.0])])
            mean = torch.cat([mean, torch.tensor([frac])])
            std = torch.cat([std, torch.tensor([(frac * (1 - frac)) ** 0.5])])
        return self._limits_normalizer_from_stats(run_min, run_max, mean, std)

    def get_normalizer(self, mode='limits', **kwargs):
        assert mode == 'limits', f"InspireDrillDataset 只实现了 limits(收到 {mode})"
        normalizer = LinearNormalizer()
        # action / state 很小(~0.1 GB),拼接所有 zarr 后正常 fit。
        _action = np.concatenate(
            [np.asarray(rb['action'][:]) for rb in self.replay_buffers], axis=0)
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            _action, mode=mode, **kwargs)
        _state = np.concatenate(
            [np.asarray(rb['state'][:]) for rb in self.replay_buffers], axis=0)
        if self.agent_pos_dim is not None:            # 与 _sample_to_data 的切片一致
            _state = _state[:, :int(self.agent_pos_dim)]
        normalizer['agent_pos'] = self._agent_pos_normalizer(_state, mode=mode, **kwargs)
        # point_cloud 分块统计(覆盖所有 zarr),与官方 limits fit 等价。
        normalizer['point_cloud'] = self._streaming_pc_normalizer()
        return normalizer
