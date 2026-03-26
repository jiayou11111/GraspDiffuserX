from typing import Optional
import numpy as np
import gym
from gym import spaces

# from configs.LinkerHandGrasp_config import LinkGraspCfg
# from env.TaskRobotEnv import RealmanGraspSingleGym
from RL_agent_1.env.TaskRobotEnv import RealmanGraspSingleGym
from RL_agent_1.configs.RealmanGrasp_config import RealGraspCfg
import torch

import os
class RealManImageWrapper(gym.Env):

    def __init__(self, env, shape_meta, render_obs_key='camera_image'):

        self.env = env
        self.n_envs = env.num_envs 
        self.render_obs_key = render_obs_key
        self.shape_meta = shape_meta
        self.render_cache = None
        self.flag = 0

        # 创建 data 目录（如果不存在）
        os.makedirs("data", exist_ok=True)

        # -------- Action Space --------
        action_shape = shape_meta['action']['shape']

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.n_envs, *action_shape),
            dtype=np.float32
        )

        # -------- Observation Space --------
        observation_space = spaces.Dict()

        for key, value in shape_meta['obs'].items():
            shape = (self.n_envs, *value['shape'])

            if key.endswith('image'):
                low, high = 0, 1
            else:
                low, high = -np.inf, np.inf

            observation_space[key] = spaces.Box(
                low=low,
                high=high,
                shape=shape,
                dtype=np.float32
            )

        self.observation_space = observation_space

    def _to_numpy(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x

    def reset(self):
        raw_obs = self.env.reset()
        return self.get_raw_observation(raw_obs)

    def step(self, action):

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.env.device)
        # print("仿真输入action:", action)
        raw_obs, privileged_obs, reward, done, info = self.env.step(action)

        obs = self.get_raw_observation(raw_obs)

        reward = self._to_numpy(reward).astype(np.float32)
        done = self._to_numpy(done).astype(bool)

        return obs, privileged_obs, reward, done, info

    def align_sim_to_real(self, img_tensor, real_mean, real_std):
        """
        Align simulation image distribution to real image distribution using batch statistics.
        img_tensor: Tensor (B, C, H, W)
        real_mean: List[float] [R, G, B]
        real_std: List[float] [R, G, B]
        """
        if img_tensor.shape[0] == 0:
            return img_tensor

        device = img_tensor.device
        dtype = img_tensor.dtype

        # Real statistics
        # shape: (1, C, 1, 1) to broadcast over B, H, W
        target_mean = torch.tensor(real_mean, device=device, dtype=dtype).view(1, -1, 1, 1)
        target_std = torch.tensor(real_std, device=device, dtype=dtype).view(1, -1, 1, 1)

        # Sim statistics (computed from the current batch)
        # Compute mean and std over (Batch, Height, Width), keeping Channel dimension separate
        sim_mean = img_tensor.mean(dim=(0, 2, 3), keepdim=True)
        sim_std = img_tensor.std(dim=(0, 2, 3), keepdim=True)

        # precise_step trick to avoid div by zero, although eps is usually enough
        sim_std = torch.clamp(sim_std, min=1e-6)

        # Align: (Sim - Mu_sim) / Std_sim * Std_real + Mu_real
        aligned_img = (img_tensor - sim_mean) / sim_std * target_std + target_mean

        # Clip to valid range [0, 1]
        aligned_img = torch.clamp(aligned_img, 0.0, 1.0)

        return aligned_img

    def get_raw_observation(self, raw_obs=None, keys_to_return=None):

        if raw_obs is None:
            raw_obs = self.env.reset()

        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]

        n_envs = raw_obs.shape[0]

        H, W = 240, 320
        img_dim = 3 * H * W

        wrist_flat = raw_obs[:, :img_dim]
        head_flat = raw_obs[:, img_dim:2*img_dim]

        state_obs = raw_obs[:, 2*img_dim:]

        wrist_img = wrist_flat.view(n_envs, 3, H, W)
        head_img = head_flat.view(n_envs, 3, H, W)

        robot0_qpos = state_obs[:, :7]
        robot0_gripper_qpos = state_obs[:, 7:8]
        robot_ee_pos = state_obs[:, 8:11]
        robot_ee_orn = state_obs[:, 11:15]

        obs_dict = {
            "agentview_image": wrist_img,
            "agentview_head_image": head_img,
            "robot0_qpos": robot0_qpos,
            "robot0_gripper_qpos": robot0_gripper_qpos,
            "robot_ee_pos": robot_ee_pos,
            "robot_ee_orn": robot_ee_orn,
        }

        # 筛选 key
        if keys_to_return is not None:
            obs_dict = {k: obs_dict[k] for k in keys_to_return if k in obs_dict}

        # 转 numpy（给 diffusion policy）
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu().numpy().astype("float32")

        return obs_dict
        # if self.flag == 0:
        #     wrist_np = wrist_img.detach().cpu().numpy()

        #     # 保存
        #     np.save("data/wrist_img.npy", wrist_np)

        #     print("Saved to data/wrist_img.npy")
        #     print("Shape:", wrist_np.shape)
        #     self.flag = 1

        # # --- Align Sim Images to Real Data Statistics ---
        
        # # AgentView Image Stats (applied to wrist_img because it is mapped to agentview_image key)
        # # Mean: [0.4705, 0.4749, 0.4760] | Std: [0.0854, 0.0808, 0.0789]
        # wrist_img = self.align_sim_to_real(
        #     wrist_img, 
        #     real_mean=[0.4705401, 0.4749347, 0.47601163],
        #     real_std=[0.08544321, 0.08081585, 0.07889712]
        # )

        # # AgentView Head Image Stats
        # # Mean: [0.4682, 0.4698, 0.4682] | Std: [0.0861, 0.0867, 0.0887]
        # head_img = self.align_sim_to_real(
        #     head_img, 
        #     real_mean=[0.4682231, 0.46976656, 0.46816772],
        #     real_std=[0.08610459, 0.08672088, 0.08871145]
        # )
        # # ------------------------------------------------

        # robot0_qpos = state_obs[:, :7]
        # robot0_gripper_qpos = state_obs[:, 7:8]
        # robot_ee_pos = state_obs[:, 8:11]
        # robot_ee_orn = state_obs[:, 11:15]

        # obs_dict = {
        #     "agentview_image": wrist_img,
        #     "agentview_head_image": head_img,
        #     "robot0_qpos": robot0_qpos,
        #     "robot0_gripper_qpos": robot0_gripper_qpos,
        #     "robot_ee_pos": robot_ee_pos,
        #     "robot_ee_orn": robot_ee_orn,
        # }

        # # 筛选 key
        # if keys_to_return is not None:
        #     obs_dict = {k: obs_dict[k] for k in keys_to_return if k in obs_dict}

        # # 转 numpy（给 diffusion policy）
        # for k in obs_dict:
        #     obs_dict[k] = obs_dict[k].cpu().numpy().astype("float32")

        # return obs_dict


    def render(self, mode='rgb_array'):

        if self.render_cache is None:
            raise RuntimeError("Call reset or step first")

        img = (self.render_cache[0] * 255).astype(np.uint8)
        return img



# def test_dp_wrapper():

#     cfg = LinkGraspCfg()
#     base_env = RealmanGraspSingleGym(cfg)

#     shape_meta = {
#         "action": {
#             "shape": (8,)
#         },
#         "obs": {
#             "agentview_image": {"shape": (3, 84, 84)},
#             "agentview_head_image": {"shape": (3, 84, 84)},
#             "robot0_qpos": {"shape": (7,)},
#             "robot0_gripper_qpos": {"shape": (1,)},
#         }
#     }

#     env = RealManImageWrapper(base_env, shape_meta)

#     obs = env.reset()

#     print("\n--- Observation after reset ---")
#     for k, v in obs.items():
#         print(f"{k}: shape={v.shape}, dtype={v.dtype}")
#         assert isinstance(v, np.ndarray)
#         assert np.all(np.isfinite(v)), f"{k} contains NaN"

#     print("\nRunning one random step...")

#     action = np.random.uniform(
#         low=-1,
#         high=1,
#         size=(env.n_envs, 8)
#     ).astype(np.float32)

#     obs, privileged_obs, reward, done, info = env.step(action)

#     print("\n--- After step ---")

#     for k, v in obs.items():
#         print(f"{k}: shape={v.shape}, dtype={v.dtype}")
#         assert isinstance(v, np.ndarray)

#     print(f"reward shape: {reward.shape}")
#     print(f"reward dtype: {reward.dtype}")

#     print(f"done shape: {done.shape}")
#     print(f"done dtype: {done.dtype}")

#     print("\n========== TEST SUCCESS ==========")


# if __name__ == "__main__":
#     test_dp_wrapper()
