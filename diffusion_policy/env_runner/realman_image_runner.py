import numpy as np
import tqdm
import collections
import time

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
import torch
class RealmanImageRunner(BaseImageRunner):

    def __init__(
        self,
        output_dir,
        env,
        env_seeds=None,
        env_prefixs=None,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        past_action=False,
        tqdm_interval_sec=5.0,
    ):
        super().__init__(output_dir)

        self.env = MultiStepWrapper(
            env,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps
        )

        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.tqdm_interval_sec = tqdm_interval_sec

        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs

    def run(self, policy: BaseImagePolicy):

        device = policy.device
        env = self.env

        n_envs = env.env.n_envs

        obs = env.reset()
        policy.reset()

        past_action = None
        done = np.zeros(n_envs, dtype=bool)

        # 用 numpy array 更高效
        max_rewards = np.zeros(n_envs)

        pbar = tqdm.tqdm(
            total=self.max_steps,
            desc="Realman Eval",
            leave=False,
            mininterval=self.tqdm_interval_sec
        )

        step_count = 0

        while step_count < self.max_steps:

            np_obs_dict = dict(obs)

            if self.past_action and (past_action is not None):
                np_obs_dict['past_action'] = past_action[
                    :, -(self.n_obs_steps-1):
                ].astype(np.float32)

            obs_dict = dict_apply(
                np_obs_dict,
                lambda x: torch.from_numpy(x).to(device=device)
            )

            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            np_action_dict = dict_apply(
                action_dict,
                lambda x: x.detach().cpu().numpy()
            )

            action = np_action_dict['action']
            # print("action:", action)

            if not np.all(np.isfinite(action)):
                raise RuntimeError("Nan or Inf action")

            # ---------- 对已完成环境屏蔽 action ----------
            # action[done] = 0.0

            obs, privileged_obs, reward, done_step, info = env.step(action)
            # print("action:", action)


            # init_action = np.array([
            #     -2.8338876,
            #     -2.0879967,
            #     2.7086096,
            #     -1.5472047,
            #     0.6880786,
            #     1.8933139,
            #     -1.7004603,
            #     1
            # ])

            # init_action = np.tile(init_action, (1,8,1))  # (1,8,8)

            # obs, privileged_obs, reward, done_step, info = env.step(init_action)

            # ---------- 累积 done ----------
            done = np.logical_or(done, done_step)

            # ---------- 记录最大 reward ----------
            max_rewards = np.maximum(max_rewards, reward)

            past_action = action

            step_count += self.n_action_steps
            pbar.update(self.n_action_steps)

        pbar.close()

        # ---------- 统计 ----------
        log_data = {}

        for i in range(n_envs):
            prefix = self.env_prefixs[i] if self.env_prefixs else ""
            log_data[prefix + f"sim_max_reward_{i}"] = max_rewards[i]

        # mean score
        mean_score = np.mean(max_rewards)
        log_data["mean_score"] = mean_score

        return log_data
