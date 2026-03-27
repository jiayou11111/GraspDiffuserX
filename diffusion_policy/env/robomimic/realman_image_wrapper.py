from typing import Optional
import numpy as np
import gym
from gym import spaces
import torch

import os
class RealManImageWrapper(gym.Env):

    def __init__(self, env, shape_meta):

        self.env = env
        self.n_envs = env.num_envs 
        self.shape_meta = shape_meta
        self.flag = 0


        # ===== 必须：给 MultiStepWrapper 用 =====
        action_dim = self.shape_meta['action']['shape'][0]

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(action_dim,),
            dtype=np.float32
        )

        # ===== 动态生成 observation_space =====
        # 这里还需要看一下
        obs_spaces = {}
        for key, meta in self.shape_meta['obs'].items():
            shape = tuple(meta['shape'])
            dtype = np.float32
            if meta.get('type', 'low_dim') == 'rgb':
                low = 0
                high = 1
            else:
                low = -np.inf
                high = np.inf
            obs_spaces[key] = spaces.Box(low=low, high=high, shape=shape, dtype=dtype)

        self.observation_space = spaces.Dict(obs_spaces)


    def reset(self):
        raw_obs = self.env.reset()
        return self.get_raw_observation(raw_obs)


    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.env.device)
        raw_obs, privileged_obs, reward, done, info = self.env.step(action)
            # reward 转 float/numpy
        if torch.is_tensor(reward):
            reward = reward.detach().cpu().numpy().astype(np.float32)
        obs = self.get_raw_observation(raw_obs)

        return obs, privileged_obs, reward, done, info


    def get_raw_observation(self, raw_obs):
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


        # ===== 组装 dict =====
        obs = {
            "agentview_image": wrist_img,
            "agentview_head_image": head_img,
            "robot0_qpos": state_obs[:, :7],
            "robot0_gripper_qpos": state_obs[:, 7:8],
            "robot_ee_pos": state_obs[:, 8:11],
            "robot_ee_orn": state_obs[:, 11:15],
        }

        # ===== 根据 shape_meta 选择 obs =====
        selected_obs = {}
        for key in self.shape_meta.get('obs', {}):
            if key in obs:
                val = obs[key]
                # 转成 numpy.float32
                if torch.is_tensor(val):
                    val = val.cpu().numpy().astype(np.float32)
                else:
                    val = val.astype(np.float32)
                selected_obs[key] = val

                # 打印 shape 和取值范围
                # print(f"{key}: shape={val.shape}, min={val.min()}, max={val.max()}")

        return selected_obs

