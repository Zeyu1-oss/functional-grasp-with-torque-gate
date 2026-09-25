"""Runtime module: generate a "full robot point cloud" via FK from the robot's current link poses.

Collection (collect_dp3_data.py) and deployment (deploy_dp3_sim.py / real robot) share the same
canonical points (generated offline from URDF by tools/build_robot_pointcloud.py), guaranteeing
point-for-point identical geometry with zero sim2real gap.

Mechanism: canonical points are in each link's local frame; each frame they are rigid-transformed
to world with the corresponding link's world pose, then env_origin is subtracted for env-local
points. The pose source is pluggable:
  - IsaacLab (collect / sim deploy): franka.data.body_pos_w / body_quat_w (already FK results)
  - real robot: compute link poses from joint angles via URDF + pytorch_kinematics, then pass in

Only depends on torch; the collect/deploy env needs no mesh / URDF library.
"""
import numpy as np
import torch

# Link-name prefixes of the Inspire hand, for RobotPointCloudFK(link_filter=...).
# The canonical npz holds 27 links: R_* (the hand, 837 pts), fr3_link0..7 (the arm, 256 pts) and
# L_flange (the wrist adapter, 187 pts -- part of neither, which is why it must be named explicitly
# rather than inferred). Defined here so collect and deploy import the SAME tuple: the filter does
# not change the point count (--robot_pc_points still decides that), so a mismatch between the two
# would leave the zarr shape identical and go completely unnoticed.
HAND_LINK_PREFIXES = ("R_",)
HAND_AND_FLANGE_PREFIXES = ("R_", "L_flange")


def _quat_to_mat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """(..., 4) wxyz quaternion -> (..., 3, 3) rotation matrix. Matches IsaacLab body_quat_w."""
    q = quat_wxyz / (torch.norm(quat_wxyz, dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def make_ground_points(workspace, n: int, z: float, device) -> torch.Tensor:
    """Synthetic ground/table point cloud: exactly n points inside the workspace x/y rectangle at
    height z (env-local m). Deterministic grid -> collect and deploy point-for-point identical
    (given the same workspace/n/z).
    workspace=[x_min, x_max, y_min, y_max, ...] (last two, z range, unused)."""
    import math
    x_min, x_max, y_min, y_max = (float(workspace[0]), float(workspace[1]),
                                  float(workspace[2]), float(workspace[3]))
    aspect = (x_max - x_min) / max(y_max - y_min, 1e-6)
    ny = max(1, int(round(math.sqrt(n / max(aspect, 1e-6)))))
    nx = max(1, int(math.ceil(n / ny)))                       # nx*ny >= n
    xs = torch.linspace(x_min, x_max, nx, device=device)
    ys = torch.linspace(y_min, y_max, ny, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1),
                       torch.full((nx * ny,), float(z), device=device)], dim=-1)
    return pts[:n]                                            # take first n, deterministic


def jitter_ground_xy(ground_batch: torch.Tensor, std: float,
                     x_min: float, x_max: float, y_min: float, y_max: float) -> torch.Tensor:
    """Per-frame add N(0,std) gaussian noise to ground xy (z unchanged), then clamp back into the
    [x_min,x_max]x[y_min,y_max] region. collect and deploy call the same function with the same std
    (single source in hyperparameters) -> same distribution, equivalent data augmentation that breaks
    the grid regularity. std<=0 returns unchanged (disabled). ground_batch: (B,N,3)."""
    if std is None or std <= 0:
        return ground_batch
    B, N, _ = ground_batch.shape
    out = ground_batch.clone()
    out[..., 0] = (ground_batch[..., 0]
                   + torch.randn(B, N, device=ground_batch.device) * std).clamp(x_min, x_max)
    out[..., 1] = (ground_batch[..., 1]
                   + torch.randn(B, N, device=ground_batch.device) * std).clamp(y_min, y_max)
    return out


def drill_crop_bounds(drill_pos_w: torch.Tensor, env_origins: torch.Tensor, half: float,
                      z_floor=None):
    """Cube crop box centered at the drill with side length 2*half. Returns 6 (B,1) tensors
    (x_min,x_max,y_min,y_max,z_min,z_max), env-local; pass directly as the workspace for
    depth_to_pointcloud_batch (mask comparison broadcasts per env). oracle: drill_pos_w uses GT pose.
    When z_floor is given, the box bottom z_min is clamped to >= z_floor (= table height, so the camera
    does not pick up the real table/ground; ground is handled by the synthetic cloud), avoiding the
    cube reaching below the table when the drill lies down."""
    c = drill_pos_w - env_origins                            # (B,3) env-local drill center
    z_lo = c[:, 2:3] - half
    if z_floor is not None:
        z_lo = z_lo.clamp_min(float(z_floor))                # box bottom not below table
    return (c[:, 0:1] - half, c[:, 0:1] + half,
            c[:, 1:2] - half, c[:, 1:2] + half,
            z_lo, c[:, 2:3] + half)


class RobotPointCloudFK:
    """Load canonical points + FK-transform each frame into a full robot point cloud."""

    def __init__(self, npz_path: str, body_names, device, verbose: bool = True,
                 max_points: int = None, link_filter=None,
                 per_link_points: int = None, fill_link: str = "R_hand_base_link"):
        """
        Args:
            npz_path:   .npz produced by tools/build_robot_pointcloud.py
            body_names: body-name order of the robot articulation (franka.body_names);
                        canonical links are mapped to body indices by this.
            device:     torch device.
            max_points: upper bound of points after downsample (None=no downsample). Fixed seed +
                        per-link stratified, so collect/deploy select identical points given the
                        same npz and max_points.
            link_filter: tuple of link-name PREFIXES to keep (e.g. HAND_LINK_PREFIXES), or None for
                        every link. Applied BEFORE the downsample, so max_points is spent entirely
                        on the surviving links -- filtering to the hand does not shrink the cloud,
                        it re-spends the same budget at higher density on the hand.
            per_link_points: exact per-link quota for every kept link except `fill_link`, which
                        absorbs whatever is left of max_points. None = the proportional stratified
                        downsample instead.
                        WARNING: the canonical npz has a hard per-link ceiling (the offline mesh
                        sampling decided it), and for the Inspire hand only R_thumb_proximal has 50+
                        points -- the five R_*_tip links have 6. A quota above the ceiling is met by
                        sampling WITH REPLACEMENT, i.e. duplicate coordinates that add no geometry
                        and only consume budget. Raise the density in
                        tools/build_robot_pointcloud.py if genuinely more finger detail is wanted.
            fill_link:  link that receives max_points minus the sum of the quotas.
        """
        data = np.load(npz_path, allow_pickle=True)
        points = data["points"].astype(np.float32)        # (M0,3) each in its link frame
        link_idx = data["link_idx"].astype(np.int64)      # (M0,) index into link_names
        link_names = [str(n) for n in data["link_names"]]

        name_to_body = {n: i for i, n in enumerate(body_names)}
        # canonical link -> articulation body index; drop points whose link does not exist in the
        # articulation (tips etc. merged away by fixed joints) and warn.
        keep = np.ones(len(points), dtype=bool)
        pt_body = np.zeros(len(points), dtype=np.int64)
        missing = {}
        excluded = {}
        for li, lname in enumerate(link_names):
            mask = link_idx == li
            bi = name_to_body.get(lname, None)
            if bi is None:
                missing[lname] = int(mask.sum())
                keep[mask] = False
            elif link_filter is not None and not lname.startswith(tuple(link_filter)):
                excluded[lname] = int(mask.sum())
                keep[mask] = False
            else:
                pt_body[mask] = bi

        pts = points[keep]
        pb = pt_body[keep]
        if len(pts) == 0:
            raise ValueError(f"link_filter={link_filter} matched no link in {npz_path}; "
                             f"available links: {link_names}")
        self.link_filter = tuple(link_filter) if link_filter is not None else None
        self.kept_link_names = [n for n in link_names
                                if n in name_to_body and (link_filter is None
                                                          or n.startswith(tuple(link_filter)))]

        # exact per-link quota: every kept link gets `per_link_points`, `fill_link` takes the rest.
        # Same fixed seed as the stratified path, so collect and deploy still select identical
        # points (including identical duplicates when a quota exceeds the link's ceiling).
        self.oversampled = {}
        if per_link_points is not None:
            if max_points is None:
                raise ValueError("per_link_points requires max_points (the total budget)")
            q, tgt = int(per_link_points), int(max_points)
            rs = np.random.RandomState(20240601)
            fill_bi = name_to_body.get(fill_link, None)
            body_to_name = {i: n for n, i in name_to_body.items()}
            sel_chunks, quota_total = [], 0
            for bi in np.unique(pb):
                if bi == fill_bi:
                    continue
                idx_b = np.where(pb == bi)[0]
                if len(idx_b) >= q:
                    pick = idx_b[rs.permutation(len(idx_b))[:q]]
                else:   # quota above this link's ceiling -> duplicate (see the warning in the docstring)
                    pick = np.concatenate([idx_b, idx_b[rs.randint(0, len(idx_b), q - len(idx_b))]])
                    self.oversampled[body_to_name.get(bi, bi)] = (len(idx_b), q)
                sel_chunks.append(pick)
                quota_total += q
            rem = tgt - quota_total
            if rem < 0:
                raise ValueError(
                    f"per_link_points={q} x {len(sel_chunks)} links = {quota_total} exceeds "
                    f"max_points={tgt}; nothing left for '{fill_link}'. Raise max_points to at "
                    f"least {quota_total + 1} or lower per_link_points.")
            if fill_bi is not None and rem > 0:
                idx_f = np.where(pb == fill_bi)[0]
                if len(idx_f) >= rem:
                    sel_chunks.append(idx_f[rs.permutation(len(idx_f))[:rem]])
                else:
                    sel_chunks.append(np.concatenate(
                        [idx_f, idx_f[rs.randint(0, len(idx_f), rem - len(idx_f))]]))
                    self.oversampled[fill_link] = (len(idx_f), rem)
            sel = np.sort(np.concatenate(sel_chunks))
            pts, pb = pts[sel], pb[sel]

        # deterministic stratified downsample: per-link quota (at least 1 point per link), so small
        # links (fingertips) are not emptied. Fixed seed -> collect/deploy identical points given the
        # same npz + max_points.
        elif max_points is not None and int(max_points) < len(pts):
            tgt = int(max_points)
            M0 = len(pts)
            rs = np.random.RandomState(20240601)
            sel_chunks = []
            for bi in np.unique(pb):
                idx_b = np.where(pb == bi)[0]
                quota = max(1, int(round(len(idx_b) * tgt / M0)))
                pick = rs.permutation(len(idx_b))[:quota]
                sel_chunks.append(idx_b[pick])
            sel = np.concatenate(sel_chunks)
            # per-link rounding may be a few points off: trim or top up
            if len(sel) > tgt:
                sel = sel[rs.permutation(len(sel))[:tgt]]
            elif len(sel) < tgt:
                rest = np.setdiff1d(np.arange(M0), sel)
                extra = rest[rs.permutation(len(rest))[:tgt - len(sel)]]
                sel = np.concatenate([sel, extra])
            sel = np.sort(sel)
            pts, pb = pts[sel], pb[sel]

        self.points = torch.from_numpy(pts).to(device)      # (M,3)
        self.pt_body = torch.from_numpy(pb).to(device)      # (M,)
        self.num_points = int(self.points.shape[0])
        self.device = device

        if verbose:
            print(f"[RobotPointCloudFK] {self.num_points} pts kept "
                  f"({len(points) - self.num_points} dropped from {len(missing)} "
                  f"merged links)")
            if missing:
                print(f"    dropped links (not live bodies): {missing}")
            # printed loudly: the filter does not change the point count, so a collect/deploy
            # mismatch is invisible in the zarr shape and would silently feed the policy a
            # differently-shaped robot segment than it was trained on.
            print(f"    link_filter = {self.link_filter}"
                  f"  -> {len(self.kept_link_names)} links: {self.kept_link_names}")
            if excluded:
                print(f"    EXCLUDED by link_filter: {excluded}")
            if self.oversampled:
                print(f"    OVERSAMPLED (duplicate points, no new geometry) "
                      f"{{link: (npz has, quota)}}: {self.oversampled}")

    @torch.no_grad()
    def __call__(self, body_pos_w: torch.Tensor, body_quat_w: torch.Tensor,
                 env_origins: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            body_pos_w:  (B, L, 3) each body's world position (IsaacLab franka.data.body_pos_w)
            body_quat_w: (B, L, 4) each body's world quaternion wxyz
            env_origins: (B, 3) env origin; if given, convert to env-local (same frame as camera cloud)
        Returns:
            (B, M, 3) full robot point cloud.
        """
        # take the pose of the link each canonical point belongs to: (B, M, ...)
        pos = body_pos_w[:, self.pt_body, :]                         # (B,M,3)
        quat = body_quat_w[:, self.pt_body, :]                       # (B,M,4)
        R = _quat_to_mat(quat)                                       # (B,M,3,3)
        local = self.points.unsqueeze(0)                            # (1,M,3)
        world = torch.einsum("bmij,bmj->bmi", R, local.expand(pos.shape[0], -1, -1)) + pos
        if env_origins is not None:
            world = world - env_origins.unsqueeze(1)
        return world
