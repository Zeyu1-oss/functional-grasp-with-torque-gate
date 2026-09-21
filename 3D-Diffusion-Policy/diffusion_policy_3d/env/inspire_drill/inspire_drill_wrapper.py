import os
import sys
import gym
import numpy as np
import torch
from gym import spaces
from typing import Dict

INSPIRE_DRILL_ROOT = os.path.expanduser("~/inspire_drill")
if INSPIRE_DRILL_ROOT not in sys.path:
    sys.path.insert(0, INSPIRE_DRILL_ROOT)


def farthest_point_sample_numpy(points: np.ndarray, num_points: int) -> np.ndarray:
    """GPU-free FPS for use in non-GPU contexts."""
    N = points.shape[0]
    if N <= num_points:
        idx = np.arange(N)
        pad = np.zeros(num_points - N, dtype=np.int64)
        return np.concatenate([idx, pad])
    farthest = np.zeros(num_points, dtype=np.int64)
    dists = np.ones(N, dtype=np.float64) * 1e10
    active = np.ones(N, dtype=bool)
    farthest[0] = np.random.randint(N)
    dists = np.sum((points - points[farthest[0]]) ** 2, axis=1)

    for i in range(1, num_points):
        candidates = dists.copy()
        candidates[~active] = -np.inf
        farthest[i] = np.argmax(candidates)
        active[farthest[i]] = False
        new_dists = np.sum((points - points[farthest[i]]) ** 2, axis=1)
        closer = new_dists < dists
        dists[closer] = new_dists[closer]
    return farthest


def depth_to_pointcloud_single(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    pos_w: np.ndarray,
    quat_w: np.ndarray,
    workspace: tuple,
    num_points: int,
) -> np.ndarray:
    """Convert a single depth image to world-space point cloud.

    Args:
        depth: (H, W, 1) depth image
        intrinsic: (3, 3) camera intrinsic matrix
        pos_w: (3,) camera position in world frame
        quat_w: (4,) camera quaternion (w,x,y,z)
        workspace: (6,) bounding box (x_min, x_max, y_min, y_max, z_min, z_max)
        num_points: target number of points

    Returns:
        (num_points, 3) world-space point cloud
    """
    H, W = depth.shape[:2]
    j_coords, i_coords = np.meshgrid(np.arange(W, dtype=np.float32),
                                      np.arange(H, dtype=np.float32))
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]

    z = depth[..., 0].astype(np.float32)
    z = np.where(np.isfinite(z), z, 0.0)
    x = (j_coords - cx) * z / fx
    y = -(i_coords - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # quaternion (w,x,y,z) to rotation matrix
    q = quat_w / (np.linalg.norm(quat_w) + 1e-8)
    w, x_, y_, z_ = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*(y_**2 + z_**2), 2*(x_*y_ - w*z_), 2*(x_*z_ + w*y_)],
        [2*(x_*y_ + w*z_), 1 - 2*(x_**2 + z_**2), 2*(y_*z_ - w*x_)],
        [2*(x_*z_ - w*y_), 2*(y_*z_ + w*x_), 1 - 2*(x_**2 + y_**2)],
    ], dtype=np.float32)
    pts_world = pts_cam @ R.T + pos_w

    x_min, x_max, y_min, y_max, z_min, z_max = workspace
    mask = (
        (pts_world[:, 0] > x_min) & (pts_world[:, 0] < x_max) &
        (pts_world[:, 1] > y_min) & (pts_world[:, 1] < y_max) &
        (pts_world[:, 2] > z_min) & (pts_world[:, 2] < z_max)
    )
    valid_pts = pts_world[mask]

    if valid_pts.shape[0] >= num_points:
        idx = farthest_point_sample_numpy(valid_pts, num_points)
        return valid_pts[idx]
    elif valid_pts.shape[0] > 0:
        result = np.zeros((num_points, 3), dtype=np.float32)
        result[:valid_pts.shape[0]] = valid_pts
        repeats = (num_points - valid_pts.shape[0] + valid_pts.shape[0] - 1) // valid_pts.shape[0]
        padded = np.repeat(valid_pts, repeats, axis=0)[:num_points - valid_pts.shape[0]]
        result[valid_pts.shape[0]:] = padded
        return result
    else:
        return np.zeros((num_points, 3), dtype=np.float32)


class InspireDrillEnv(gym.Env):
    """DP3 environment wrapper for the IsaacLab GraspDrillEnv.

    This wrapper:
    1. Launches IsaacLab via AppLauncher (headless or GUI)
    2. Wraps GraspDrillEnv with gym.Env interface
    3. Converts depth images from two cameras (managed by GraspDrillEnv as scene.sensors)
       into a single world-frame point cloud
    4. Exposes agent_pos (26,) = joint_pos(13,) + joint_vel(13,) and point_cloud (N,3)

    Data format matches what collect_dp3_data.py produces:
      - state: (26,) = joint_pos(13,) + joint_vel(13,)
      - action: (13,)
      - point_cloud: (N, 3)
    """

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cuda:0",
        headless: bool = True,
        num_points: int = 500,
        img_height: int = 240,
        img_width: int = 320,
        workspace: tuple = (-0.5, 1.0, -0.5, 0.5, 0.0, 1.5),
        drill_config_path: str = None,
        debug: bool = False,
        seed: int = 42,
    ):
        super().__init__()

        self.num_envs = num_envs
        self.device = device
        self.num_points = num_points
        self.half_points = num_points // 2
        self.workspace = workspace
        self.img_height = img_height
        self.img_width = img_width
        self._debug = debug
        self._seed = seed
        self._drill_config_path = drill_config_path

        # Lazy init: actual IsaacLab app / env created on first reset
        self._simulation_app = None
        self._env = None
        self._env_unwrapped = None
        self._controlled_indices = None
        self._initialized = False

        # DP3 action dim: 13 (7 arm + 6 hand)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32),
            "point_cloud": spaces.Box(low=-np.inf, high=np.inf, shape=(num_points, 3), dtype=np.float32),
        })

    def _lazy_init(self):
        """Initialize IsaacLab app and env. Called on first reset."""
        if self._initialized:
            return

        from isaaclab.app import AppLauncher
        from tasks.grasp_drill_env import GraspDrillEnv, create_grasp_drill_env_cfg

        app_launcher = AppLauncher(headless=True, enable_cameras=True)
        self._simulation_app = app_launcher.app

        cfg = create_grasp_drill_env_cfg(
            num_envs=self.num_envs,
            device=self.device,
            headless=True,
            debug=self._debug,
            drill_config_path=self._drill_config_path,
            img_height=self.img_height,
            img_width=self.img_width,
            enable_cameras=True,
        )
        cfg.seed = self._seed

        self._env_unwrapped = GraspDrillEnv(cfg=cfg, debug=self._debug)
        self._env = self._env_unwrapped

        # Access existing scene sensors: GraspDrillEnv already creates cam1 & cam2
        # via cfg.scene.cam1 / cfg.scene.cam2 in create_grasp_drill_env_cfg
        self._controlled_indices = self._env_unwrapped.controlled_joint_indices.cpu()

        # Warmup: timeline + steps to prime camera pipeline
        import carb
        import omni.timeline
        dt = self._env_unwrapped.step_dt
        settings = carb.settings.get_settings()
        settings.set("/app/player/useFixedTimeStepping", True)
        settings.set("/app/player/targetFrameRate", int(1.0 / dt))
        settings.set("/app/runLoops/rendering_0/rateLimitEnabled", True)
        settings.set("/app/runLoops/rendering_0/rateLimit", 60)
        settings.set("/physics/updateToUsd", False)
        settings.set("/physics/updateParticlesToUsd", False)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        zero = torch.zeros(self.num_envs, 13, device=self.device, dtype=torch.float32)
        for _ in range(5):
            self._env_unwrapped.step(zero)
            # Prime scene cameras
            _ = self._env_unwrapped._get_camera_observations()

        self._initialized = True

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current agent_pos and point_cloud from the env.

        Uses GraspDrillEnv's existing _get_camera_observations() which accesses
        the scene sensors (scene.sensors["cam1"] and scene.sensors["cam2"])
        that are already managed by the env.
        """
        # Get depth from the env's own camera sensors
        camera_obs = self._env_unwrapped._get_camera_observations()
        # camera_obs keys: "cam1", "cam2"  values: (B, H, W, 1) depth tensors

        # Agent state: joint_pos + joint_vel for controlled joints
        jp = self._env_unwrapped.franka.data.joint_pos
        jv = self._env_unwrapped.franka.data.joint_vel
        ctrl = self._controlled_indices
        agent_pos = torch.cat([jp[:, ctrl], jv[:, ctrl]], dim=-1).cpu().numpy()

        # Get camera extrinsics from scene sensors
        scene = self._env_unwrapped.scene
        cam1_sensor = scene.sensors.get("cam1")
        cam2_sensor = scene.sensors.get("cam2")

        batch_pc1 = []
        batch_pc2 = []
        depth1 = camera_obs.get("cam1")
        depth2 = camera_obs.get("cam2")

        if depth1 is not None:
            depth1 = depth1.cpu().numpy()
        if depth2 is not None:
            depth2 = depth2.cpu().numpy()

        for env_id in range(self.num_envs):
            # Cam1
            if depth1 is not None and cam1_sensor is not None:
                d1 = depth1[env_id]
                intr1 = cam1_sensor.data.intrinsic_matrices[env_id].cpu().numpy()
                pos1 = cam1_sensor.data.pos_w[env_id].cpu().numpy()
                quat1 = cam1_sensor.data.quat_w_ros[env_id].cpu().numpy()
                pc1 = depth_to_pointcloud_single(d1, intr1, pos1, quat1, self.workspace, self.half_points)
            else:
                pc1 = np.zeros((self.half_points, 3), dtype=np.float32)
            batch_pc1.append(pc1)

            # Cam2
            if depth2 is not None and cam2_sensor is not None:
                d2 = depth2[env_id]
                intr2 = cam2_sensor.data.intrinsic_matrices[env_id].cpu().numpy()
                pos2 = cam2_sensor.data.pos_w[env_id].cpu().numpy()
                quat2 = cam2_sensor.data.quat_w_ros[env_id].cpu().numpy()
                pc2 = depth_to_pointcloud_single(d2, intr2, pos2, quat2, self.workspace, self.half_points)
            else:
                pc2 = np.zeros((self.half_points, 3), dtype=np.float32)
            batch_pc2.append(pc2)

        pc1 = np.stack(batch_pc1, axis=0)
        pc2 = np.stack(batch_pc2, axis=0)

        # Fuse: first half from cam1, second half from cam2
        pc_fused = np.zeros((self.num_envs, self.num_points, 3), dtype=np.float32)
        pc_fused[:, :self.half_points] = pc1
        pc_fused[:, self.half_points:] = pc2

        return {
            "agent_pos": agent_pos.astype(np.float32),
            "point_cloud": pc_fused,
        }

    def reset(self) -> Dict[str, np.ndarray]:
        if not self._initialized:
            self._lazy_init()

        obs = self._env_unwrapped.reset()
        return self._get_obs()

    def step(self, action) -> tuple:
        """Step the environment.

        Args:
            action: (13,) or (num_envs, 13) action array

        Returns:
            obs: dict with agent_pos (num_envs, 26) and point_cloud (num_envs, num_points, 3)
            reward: (num_envs,)
            done: (num_envs,)
            info: dict
        """
        if not self._initialized:
            self._lazy_init()

        if isinstance(action, np.ndarray):
            if action.ndim == 1:
                action = np.broadcast_to(action, (self.num_envs, 13))
            actions_t = torch.from_numpy(action).to(self.device).float()
        else:
            actions_t = action

        obs, reward, terminated, truncated, info = self._env_unwrapped.step(actions_t)
        is_done = np.logical_or(terminated, truncated)

        # Expose success for the runner
        try:
            success = self._env_unwrapped._cached_lenient_success.cpu().numpy()
        except AttributeError:
            success = np.zeros(self.num_envs, dtype=bool)
        info["goal_achieved"] = success

        gym_obs = self._get_obs()
        return gym_obs, reward.cpu().numpy(), is_done.cpu().numpy(), info

    def render(self, mode="rgb_array"):
        """Render using the first env's RGB camera (cam1)."""
        if not self._initialized:
            self._lazy_init()

        camera_obs = self._env_unwrapped._get_camera_observations()
        # The scene camera sensors produce distance_to_image_plane, not RGB.
        # Fall back to a zero frame for rendering.
        return np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

    def is_success(self) -> np.ndarray:
        """Return success flag for each env in the last step."""
        try:
            return self._env_unwrapped._cached_lenient_success.cpu().numpy()
        except AttributeError:
            return np.zeros(self.num_envs, dtype=bool)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
        if self._simulation_app is not None:
            self._simulation_app.close()
            self._simulation_app = None
        self._initialized = False

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
