import wandb
import numpy as np
import torch
import tqdm
from collections import deque
from diffusion_policy_3d.env import InspireDrillEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint


class InspireDrillRunner(BaseRunner):
    def __init__(
        self,
        output_dir,
        eval_episodes=20,
        max_steps=300,
        n_obs_steps=8,
        n_action_steps=8,
        fps=10,
        crf=22,
        render_size=84,
        tqdm_interval_sec=5.0,
        num_envs=1,
        device="cuda:0",
        num_points=500,
        img_height=240,
        img_width=320,
        workspace=(-0.5, 1.0, -0.5, 0.5, 0.0, 1.5),
        drill_config_path=None,
        seed=42,
        headless=True,
    ):
        super().__init__(output_dir)
        self.eval_episodes = eval_episodes
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.fps = fps
        self.crf = crf
        self.num_envs = num_envs
        self.device = device
        self.num_points = num_points
        self.img_height = img_height
        self.img_width = img_width
        self.workspace = workspace
        self.drill_config_path = drill_config_path
        self.seed = seed
        self.headless = headless
        self.tqdm_interval_sec = tqdm_interval_sec

        steps_per_render = max(10 // fps, 1)

        def env_fn():
            raw_env = InspireDrillEnv(
                num_envs=num_envs,
                device=device,
                headless=headless,
                num_points=num_points,
                img_height=img_height,
                img_width=img_width,
                workspace=workspace,
                drill_config_path=drill_config_path,
                debug=False,
                seed=seed,
            )
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(raw_env, steps_per_render=steps_per_render),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method="sum",
            )

        self.env = env_fn()
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_success_rates = []
        all_goal_achieved = []

        for episode_idx in tqdm.tqdm(
            range(self.eval_episodes),
            desc=f"Eval InspireDrill",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        ):
            obs = env.reset()
            policy.reset()

            done = False
            num_goal_achieved = 0
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device)
                )

                obs_dict_input = {}
                obs_dict_input["point_cloud"] = obs_dict["point_cloud"].unsqueeze(0)
                obs_dict_input["agent_pos"] = obs_dict["agent_pos"].unsqueeze(0)

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(
                    action_dict,
                    lambda x: x.detach().to("cpu").numpy(),
                )
                action = np_action_dict["action"].squeeze(0)

                obs, reward, done, info = env.step(action)
                num_goal_achieved += np.sum(info.get("goal_achieved", 0))
                done = np.all(done)

            all_success_rates.append(info.get("goal_achieved", np.array([False])))
            all_goal_achieved.append(num_goal_achieved)

        success_rate_mean = np.mean([float(np.any(sr)) for sr in all_success_rates])

        log_data = dict()
        log_data["mean_n_goal_achieved"] = np.mean(all_goal_achieved)
        log_data["mean_success_rates"] = success_rate_mean
        log_data["test_mean_score"] = success_rate_mean
        cprint(f"test_mean_score: {success_rate_mean}", "green")

        self.logger_util_test.record(success_rate_mean)
        self.logger_util_test10.record(success_rate_mean)
        log_data["SR_test_L3"] = self.logger_util_test.average_of_largest_K()
        log_data["SR_test_L5"] = self.logger_util_test10.average_of_largest_K()

        videos = env.env.get_video()
        if videos is not None and len(videos.shape) == 5:
            videos = videos[:, 0]
        if videos is not None:
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data["sim_video_eval"] = videos_wandb

        _ = env.reset()
        del env

        return log_data
