
import argparse

import numpy as np
import zarr

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--data", default='/home/zeyu/inspire_drill/data/norobot.zarr',)
_ap.add_argument("--episode", type=int, default=0)
_ap.add_argument("--frame", type=int, default=None,
                 help="给了就单帧模式(负数=从末尾数,-1=最后一帧);不给=整条 episode 动画")
_ap.add_argument("--stride", type=int, default=1,
                 help="动画抽帧间隔(1=逐帧,HTML 更大;90帧 episode 用 2 足够流畅)")
_ap.add_argument("--hide", nargs="*", default=[],
                 help="默认隐藏的段名(如 --hide ground robot),图例里仍可点开")
_ap.add_argument("--only", nargs="*", default=None,
                 help="只看指定段(其余默认隐藏,图例里仍可点开)。如 --only cam1 只看 cam1;"
                      "给了 --only 时 --hide 被忽略。段名: cam1 cam2 plate robot drill ground")
_args = _ap.parse_args()

data_path = _args.data
episode_idx = _args.episode
animate = _args.frame is None
frame_in_ep = _args.frame if _args.frame is not None else -1
frame_stride = _args.stride
# --only 优先:只显示 only 里的段(其余隐藏);否则用 --hide 隐藏指定段
only_segments = tuple(_args.only) if _args.only is not None else None
hide_segments = tuple(_args.hide)


def detect_segments(N):
    """按总点数识别段布局 -> [(名字, 起, 止), ...]"""
    if N == 2304:   # 串联 robot_drill: cam1|cam2|plate|robot|drill|ground
        return [("cam1", 0, 512), ("cam2", 512, 1024), ("plate", 1024, 1536),
                ("robot", 1536, 1696), ("drill", 1696, 2208), ("ground", 2208, 2304)]
    if N == 1792:   # 串联 robot(无 drill 段): cam1|cam2|plate|robot|ground
        return [("cam1", 0, 512), ("cam2", 512, 1024), ("plate", 1024, 1536),
                ("robot", 1536, 1696), ("ground", 1696, 1792)]
    if N == 1696:   # 串联 robot,--ground_points 0(无 drill 段、无 ground 段): cam1|cam2|plate|robot
        return [("cam1", 0, 512), ("cam2", 512, 1024), ("plate", 1024, 1536),
                ("robot", 1536, 1696)]
    if N == 1184:   # 纯抓取 --stage1_only(cam3 关,无 plate/drill/ground 段): cam1|cam2|robot
        return [("cam1", 0, 512), ("cam2", 512, 1024), ("robot", 1024, 1184)]
    if N == 1280:   # 单段现行方案: cam1|cam2|robot|ground
        return [("cam1", 0, 512), ("cam2", 512, 1024),
                ("robot", 1024, 1184), ("ground", 1184, 1280)]
    # --stage1_only --disable_cam2:相机段整块给 cam1,后面直接接 FK 机器人段,没有
    # plate/drill/ground。两种 robot 预算只差机器人段长度,前 2048 都是 cam1。
    if N == 3328:   # --pc_num_points 2048 + --robot_pc_points 1280(手部 R_* 专用预算)
        return [("cam1", 0, 2048), ("robot", 2048, 3328)]
    if N == 2560:   # --pc_num_points 2048 + --robot_pc_points 512
        # 曾被当成旧的 camera2048|ground512 —— 那会把 512 个 FK 机器人点画成"地面"。
        # 现行采集(collect_dp3_data.py --disable_cam2,--ground_points 默认 0)不产生 ground 段。
        return [("cam1", 0, 2048), ("robot", 2048, 2560)]
    if N == 3810:   # 旧 robot 模式
        return [("camera", 0, 2048), ("robot+drill", 2048, 3298), ("ground", 3298, 3810)]
    return [("all", 0, N)]


SEG_COLOR = {
    "cam1": "rgb(40,90,255)",      # 蓝:cam1 相机段
    "cam2": "rgb(0,200,255)",      # 青:cam2 (串联=侧上方广域)
    "plate": "rgb(255,160,0)",     # 橙:plate 相机(cam3)
    "robot": "rgb(230,30,30)",     # 红:FK 机器人段
    "drill": "rgb(230,0,230)",     # 品红:电钻 mesh 完整云(robot_drill 段,真值位姿)
    "robot+drill": "rgb(230,30,30)",
    "camera": "rgb(40,90,255)",
    "ground": "rgb(150,150,150)",  # 灰:合成地面(原本也是红,会和 robot 撞色)
    "all": "rgb(40,90,255)",
}

z = zarr.open(data_path, mode='r')
episode_ends = z['meta']['episode_ends'][:]
n_ep = len(episode_ends)
starts = np.concatenate([[0], episode_ends[:-1]])

assert 0 <= episode_idx < n_ep, f"episode 越界:{episode_idx},共 {n_ep} 条(0..{n_ep-1})"
s, e = int(starts[episode_idx]), int(episode_ends[episode_idx])
ep_len = e - s
ep_pc = np.asarray(z['data']['point_cloud'][s:e]).astype(np.float32)   # (T, N, 3) 或 (T, N, 4)
N = ep_pc.shape[1]
segments = detect_segments(N)

# ---- 把手 mask(可选):独立数据集 data/pc_mask,或 point_cloud 的第 4 通道 ----
# 有 mask 时按 mask 两色上色(mask=1 把手红,其余灰),盖过分段配色,直接看标签对不对。
ep_mask = None
if 'pc_mask' in z['data']:
    ep_mask = np.asarray(z['data']['pc_mask'][s:e]).astype(np.uint8)   # (T, N)
elif ep_pc.shape[-1] >= 4:
    ep_mask = (ep_pc[..., 3] > 0.5).astype(np.uint8)                   # 第 4 通道当 mask
    ep_pc = ep_pc[..., :3]
MASK_MODE = ep_mask is not None
if MASK_MODE:
    print(f"[MASK] 检测到把手 mask,按 mask 两色上色(红=把手 mask=1,灰=其余)。"
          f"首帧 mask=1 点数={int(ep_mask[0].sum())}/{N}")

print(f"数据集共 {n_ep} 条 episode;episode {episode_idx}: 全局帧 [{s},{e}), 长度 {ep_len}")
print(f"point_cloud 每帧 {N} 点,分段: " + " | ".join(f"{n}[{a}:{b}]" for n, a, b in segments))
for name, a, b in segments:
    seg = ep_pc[0, a:b]
    print(f"  首帧 {name:8s} x=[{seg[:,0].min():.3f},{seg[:,0].max():.3f}] "
          f"y=[{seg[:,1].min():.3f},{seg[:,1].max():.3f}] "
          f"z=[{seg[:,2].min():.3f},{seg[:,2].max():.3f}]")


def _nonzero(pts):
    return pts[np.abs(pts).sum(-1) > 1e-6]


def wrist_check(ep_pc, segments):
    """整条 episode 的 cam2→参照段(除 cam2/ground 外全部)最近邻中位数(m)。

    参照必须是并集而非只 cam1:运输/对齐阶段手臂在 plate 区域,cam1 视野覆盖
    不到(纯遮挡会把 cam2→cam1 推到 0.2m,误报),那里由 plate(cam3)/robot 段
    兜底;ground 是合成平面会假匹配,排除。实测:外参正确 ≈1cm 恒定,腕相机
    冻结外参 bug 的旧数据随机械臂移动漂到几十 cm。返回 (T,),算不了的帧为 nan。
    """
    segd = {n: (a, b) for n, a, b in segments}
    if "cam2" not in segd:
        return None
    a2, b2 = segd["cam2"]
    refs = [(a, b) for n, (a, b) in segd.items() if n not in ("cam2", "ground")]
    out = np.full(ep_pc.shape[0], np.nan, dtype=np.float32)
    for t in range(ep_pc.shape[0]):
        p2 = _nonzero(ep_pc[t, a2:b2])
        ref = np.concatenate([_nonzero(ep_pc[t, a:b]) for a, b in refs])
        if len(ref) == 0 or len(p2) == 0:
            continue
        d = np.linalg.norm(p2[:, None, :] - ref[None, :, :], axis=-1).min(1)
        out[t] = np.median(d)
    return out


wrist_nn = wrist_check(ep_pc, segments)
wrist_title = ""
if wrist_nn is not None and np.isfinite(wrist_nn).any():
    _med, _max = np.nanmedian(wrist_nn), np.nanmax(wrist_nn)
    _verdict = "OK(外参正确)" if _max < 0.05 else "异常!疑似腕相机冻结外参的旧数据集"

import plotly.graph_objs as go
import plotly.io as pio


def mask_traces(pc, mask):
    """有把手 mask 时:两色上色(mask=1 把手红,其余灰),盖过分段配色。
    pc:(N,3) 一帧;mask:(N,) uint8。只画非零点。"""
    xyz = pc[:, :3]
    keep = np.abs(xyz).sum(-1) > 1e-6                 # 去掉零填充占位点
    xyz = xyz[keep]; m = mask[keep].astype(bool)
    out = []
    for name, sel, color in [("handle (mask=1)", m, "rgb(230,30,30)"),
                             ("other (mask=0)", ~m, "rgb(170,170,170)")]:
        p = xyz[sel]
        out.append(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers', name=name,
            legendgroup=name, visible=True,
            marker=dict(size=2.8 if 'handle' in name else 2.0,
                        color=color, opacity=0.95 if 'handle' in name else 0.55)))
    return out


def seg_traces(pc, mask=None):
    """一帧的点云 -> 每段一个 trace(按段固定配色,图例可单独开关某段)。
    有 mask(MASK_MODE)时改走 mask_traces 两色上色。
    只用前三维 xyz(6 通道点云的 rgb 后三维忽略),只画非零点
    (相机段有零填充占位,画出来会在原点堆一团)。"""
    if MASK_MODE and mask is not None:
        return mask_traces(pc, mask)
    out = []
    for name, a, b in segments:
        seg = pc[a:b, :3]                              # 只取 xyz,丢弃可能存在的 rgb
        seg = seg[np.abs(seg).sum(-1) > 1e-6]          # 去掉零填充占位点
        # --only 给了:只显示 only 里的段,其余隐藏;否则按 --hide 隐藏
        if only_segments is not None:
            _visible = True if name in only_segments else 'legendonly'
        else:
            _visible = 'legendonly' if name in hide_segments else True
        out.append(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
            mode='markers', name=name,          # 稳定名(动画各帧点数会变,名不能变否则图例错乱)
            legendgroup=name,
            visible=_visible,
            marker=dict(size=2.5, color=SEG_COLOR.get(name, "rgb(180,180,180)"), opacity=0.9)))
    return out


# 坐标范围用整条 episode 的 bbox 固定住,动画时视角不跳(只取 xyz,兼容 6 通道点云)。
# 只用真实点(非零填充)算 bbox:否则相机段的零占位点会把范围拉到原点、真实点被挤成一小团。
# xyz 比例尺一致:三个轴的 range 长度统一取最大 extent(各自居中),配 aspectmode='cube'
# ->既是固定立方体框,又保证 1 米在 x/y/z 视觉长度相同、几何不失真;range 覆盖全部真实点不裁剪。
flat = ep_pc[..., :3].reshape(-1, 3)
flat = flat[np.abs(flat).sum(-1) > 1e-6]              # 去掉零填充占位点,只按真实点定范围
pad = 0.03
mins = flat.min(0); maxs = flat.max(0)
centers = (mins + maxs) / 2
half = float((maxs - mins).max()) / 2 + pad           # 立方体半边长 = 最大 extent 的一半(+pad)
rng = [(centers[i] - half, centers[i] + half) for i in range(3)]
scene = dict(
    xaxis=dict(range=rng[0]), yaxis=dict(range=rng[1]), zaxis=dict(range=rng[2]),
    aspectmode='cube',        # 立方体框 + 等边 range -> xyz 比例尺一致,几何不变形,所有点在视野内
    bgcolor='white',
)

if animate:
    ts = list(range(0, ep_len, frame_stride))
    if ts[-1] != ep_len - 1:
        ts.append(ep_len - 1)          # 结尾帧(对齐姿态)必看
    frames = [go.Frame(data=seg_traces(ep_pc[t], None if ep_mask is None else ep_mask[t]),
                       name=str(t)) for t in ts]
    fig = go.Figure(data=seg_traces(ep_pc[ts[0]], None if ep_mask is None else ep_mask[ts[0]]),
                    frames=frames)
    fig.update_layout(
        scene=scene,
        title=f"episode {episode_idx} ({ep_len} 帧, 抽帧 stride={frame_stride}){wrist_title}",
        updatemenus=[dict(
            type="buttons", x=0.05, y=1.05,
            buttons=[
                dict(label="▶ 播放", method="animate",
                     args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)]),
                dict(label="⏸ 暂停", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ])],
        sliders=[dict(
            currentvalue=dict(prefix="帧 "),
            steps=[dict(method="animate", label=str(t),
                        args=[[str(t)], dict(mode="immediate",
                                             frame=dict(duration=0, redraw=True))])
                   for t in ts])],
    )
    html_path = f'/home/zeyu/inspire_drill/data/viz_pc_ep{episode_idx}_anim.html'
else:
    t = frame_in_ep + ep_len if frame_in_ep < 0 else frame_in_ep
    assert 0 <= t < ep_len, f"帧越界:episode {episode_idx} 只有 {ep_len} 帧(0..{ep_len-1})"
    fig = go.Figure(data=seg_traces(ep_pc[t], None if ep_mask is None else ep_mask[t]))
    fig.update_layout(scene=scene, title=f"episode {episode_idx} 第 {t} 帧{wrist_title}")
    html_path = f'/home/zeyu/inspire_drill/data/viz_pc_ep{episode_idx}_t{t}.html'

pio.write_html(fig, html_path)
print(f"Saved to {html_path}")
