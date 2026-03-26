import numpy as np
import tqdm
import collections
import time

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
import os
import torch
import cv2

class RealmanImageRunner(BaseImageRunner):

    def __init__(
        self,
        output_dir,
        env,
        env_seeds=None,
        env_prefixs=None,
        max_steps=400,
        n_obs_steps=1,
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

        # past_action = None
        # done = np.zeros(n_envs, dtype=bool)

        # 用 numpy array 更高效
        # max_rewards = np.zeros(n_envs)

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
            # print_obs_dict(obs_dict, "TORCH_OBS")
            # inspect_and_save_obs(obs_dict)

            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
                # print("finished policy.predict_action")

            np_action_dict = dict_apply(
                action_dict,
                lambda x: x.detach().cpu().numpy()
            )

            action = np_action_dict['action']
            print("action:", action)

            if not np.all(np.isfinite(action)):
                raise RuntimeError("Nan or Inf action")

            obs, privileged_obs, reward, done_step, info = env.step(action)

            # done = np.logical_or(done, done_step)
            # max_rewards = np.maximum(max_rewards, reward)
            # past_action = action

            step_count += self.n_action_steps
            pbar.update(self.n_action_steps)

        pbar.close()

        # ---------- 统计 ----------
        log_data = {}

        # for i in range(n_envs):
        #     prefix = self.env_prefixs[i] if self.env_prefixs else ""
        #     log_data[prefix + f"sim_max_reward_{i}"] = max_rewards[i]

        # # mean score
        # mean_score = np.mean(max_rewards)
        # log_data["mean_score"] = mean_score

        return log_data

# 打印obs内部结构范围
def print_obs_dict(obs_dict, name="obs"):
    def _print(x, key):
        # 嵌套 dict
        if isinstance(x, dict):
            print(f"{key}: (dict)")
            for k, v in x.items():
                _print(v, f"{key}.{k}")

        # torch tensor
        elif torch.is_tensor(x):
            x_detach = x.detach()
            print(f"{key}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
            print(f"  min={x_detach.min().item():.4f}, max={x_detach.max().item():.4f}, mean={x_detach.mean().item():.4f}")

        # numpy
        elif isinstance(x, np.ndarray):
            print(f"{key}: shape={x.shape}, dtype={x.dtype}")
            print(f"  min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")

        else:
            print(f"{key}: {type(x)}")

    print(f"\n===== {name} STRUCTURE =====")
    _print(obs_dict, name)

# 打印state,保存图片
def inspect_and_save_obs(obs_dict, save_dir="compare_play"):
    os.makedirs(save_dir, exist_ok=True)

    # ===== 自动计数器 =====
    if not hasattr(inspect_and_save_obs, "counter"):
        inspect_and_save_obs.counter = 0

    idx = inspect_and_save_obs.counter
    inspect_and_save_obs.counter += 1

    # -------- 1. 打印关节信息 --------
    qpos = obs_dict["robot0_qpos"].detach().cpu().numpy().squeeze()
    gripper = obs_dict["robot0_gripper_qpos"].detach().cpu().numpy().squeeze()

    print("\n===== ROBOT STATE =====")
    print("robot0_qpos:", qpos)
    print("robot0_gripper_qpos:", gripper)

    # -------- 2. 处理图像 --------
    img = obs_dict["agentview_head_image"]

    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    img = img.squeeze()  # (3, H, W)

    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))  # -> HWC

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # -------- 3. 保存 --------
    save_path = os.path.join(save_dir, f"obs_{idx:05d}.png")
    cv2.imwrite(save_path, img)

    print(f"image saved to: {save_path}")