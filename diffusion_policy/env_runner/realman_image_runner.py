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
        max_steps=1000,
        n_obs_steps=1,
        n_action_steps=8,

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


    def run(self, policy: BaseImagePolicy) -> dict:

        device = policy.device
        env = self.env

        n_envs = env.env.n_envs

        obs = env.reset()
        policy.reset()

        step_count = 0

        results = {}

        # ===== 新增：评估指标统计记录 =====
        total_episodes = 0
        total_successes = 0
        # 记录并行环境中，每个环境在当前回合内是否触发过成功 (reward == 1)
        env_has_succeeded = np.zeros(n_envs, dtype=bool)


        while step_count < self.max_steps:

            np_obs_dict = dict(obs)

            obs_dict = dict_apply(
                np_obs_dict,
                lambda x: torch.from_numpy(x).to(device=device)
            )

            # 检查输入obs
            # print_obs_dict(obs_dict, "TORCH_OBS")
            # inspect_and_save_obs(obs_dict)

            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            np_action_dict = dict_apply(
                action_dict,
                lambda x: x.detach().cpu().numpy()
            )

            action = np_action_dict['action']
            print("action:", action.shape)

            if not np.all(np.isfinite(action)):
                raise RuntimeError("Nan or Inf action")
            
            obs, privileged_obs, reward, done_step, info = env.step(action)

            for i in range(n_envs):

                if reward[i] >= 0.99:  # 为了防止浮点精度问题，可以用 >= 0.99
                    env_has_succeeded[i] = True
                
                if done_step[i]:
                    total_episodes += 1
                    if env_has_succeeded[i]:
                        total_successes += 1
                    
                    # 记得清空该环境成功标志，准备下一回合
                    env_has_succeeded[i] = False

            step_count += self.n_action_steps
    
        # ===== 循环结束：打印并在结果中返回 =====
        success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
        print("\n================ EVALUATION SUMMARY ================")
        print(f"Total Steps run: {step_count}")
        print(f"Total Episodes completed: {total_episodes}")
        print(f"Total Successes: {total_successes}")
        print(f"Success Rate: {success_rate*100:.2f}%")
        print("====================================================\n")

        results['success_rate'] = success_rate
        results['total_episodes'] = total_episodes
        results['total_successes'] = total_successes
        
        return results





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