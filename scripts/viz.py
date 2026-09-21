"""点云可视化:从 zarr(data/point_cloud + meta/episode_ends)取一帧或多帧,
matplotlib 3D scatter 存成 png。兼容 xyz(3维,按高度上色)和 xyz+rgb(6维,按真实颜色)点云。

不依赖 IsaacLab/torch,只要 zarr+numpy+matplotlib,普通 python 环境就能跑,方便随手看数据。
逻辑跟 /home/zeyu/inspire_drill/tools/viz.py 同一份,取代原来 visualizer/(Flask+Plotly,
无人引用)的旧点云可视化工具。

用法:
  python scripts/viz.py --zarr <path> --episode 0 --frame 0
  python scripts/viz.py --zarr <path> --episode 0 --frames 0,10,20,30 --out grid.png   # 多帧网格对比
"""
import argparse
import os

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="zarr 点云可视化(matplotlib 3D scatter)")
    p.add_argument("--zarr", type=str, required=True, help="zarr 路径(读 data/point_cloud, meta/episode_ends)")
    p.add_argument("--episode", type=int, default=0, help="第几条 episode(0-indexed)")
    p.add_argument("--frame", type=int, default=0, help="该 episode 内第几帧(0-indexed);--frames 指定时忽略")
    p.add_argument("--frames", type=str, default=None, help="逗号分隔多帧,如 '0,10,20,30',画成网格对比")
    p.add_argument("--out", type=str, default=None, help="输出 png 路径;默认存在 zarr 同目录下")
    p.add_argument("--elev", type=float, default=20.0)
    p.add_argument("--azim", type=float, default=45.0)
    p.add_argument("--point_size", type=float, default=3.0)
    p.add_argument("--max_points", type=int, default=None, help="超过则随机下采样,加快绘图")
    return p.parse_args()


def _episode_slice(z, episode):
    ends = np.asarray(z["meta/episode_ends"][:])
    starts = np.concatenate([[0], ends[:-1]])
    if episode < 0 or episode >= len(ends):
        raise ValueError(f"episode {episode} 超范围,该 zarr 共 {len(ends)} 条 episode")
    return int(starts[episode]), int(ends[episode])


def _plot_one(ax, pc, elev, azim, point_size, max_points, title):
    if max_points is not None and pc.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pc.shape[0], max_points, replace=False)
        pc = pc[idx]
    xyz = pc[:, :3]
    if pc.shape[1] >= 6:
        colors = np.clip(pc[:, 3:6] / 255.0, 0.0, 1.0)
    else:
        # 没有 rgb:按高度(z)上色,给个直观的深度提示
        zc = xyz[:, 2]
        zn = (zc - zc.min()) / max(zc.max() - zc.min(), 1e-6)
        colors = plt.cm.viridis(zn)[:, :3]
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=point_size, marker=".", linewidths=0)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)
    # 等比例坐标轴,避免点云看起来被拉伸变形
    mins = xyz.min(0); maxs = xyz.max(0)
    center = (mins + maxs) / 2
    half = max((maxs - mins).max() / 2, 1e-3)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def main():
    args = parse_args()
    z = zarr.open(args.zarr, mode="r")
    pc_arr = z["data/point_cloud"]
    s, e = _episode_slice(z, args.episode)
    ep_len = e - s

    if args.frames:
        frames = [int(x) for x in args.frames.split(",")]
    else:
        frames = [args.frame]
    for f in frames:
        if f < 0 or f >= ep_len:
            raise ValueError(f"frame {f} 超出 episode {args.episode} 长度 {ep_len}")

    n = len(frames)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4.5 * ncols, 4.2 * nrows), dpi=140)

    zarr_name = os.path.basename(args.zarr.rstrip("/"))
    pt_counts = []
    for i, f in enumerate(frames):
        pc = np.asarray(pc_arr[s + f]).astype(np.float32)
        pt_counts.append(pc.shape[0])
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        _plot_one(ax, pc, args.elev, args.azim, args.point_size, args.max_points,
                  title=f"{zarr_name}\nep{args.episode} frame{f} (N={pc.shape[0]})")

    fig.tight_layout()
    out = args.out
    if out is None:
        base = zarr_name.replace(".zarr", "")
        tag = "_".join(str(x) for x in frames)
        out_dir = os.path.dirname(args.zarr.rstrip("/")) or "."
        out = os.path.join(out_dir, f"viz_{base}_ep{args.episode}_f{tag}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    print(f"[VIZ] saved -> {out}  (points/frame: {pt_counts})")


if __name__ == "__main__":
    main()
