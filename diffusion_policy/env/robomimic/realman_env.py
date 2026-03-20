# 对接NexusMinds代码时不需要这个
# import numpy as np
# from NexusMinds_RL.env.Robot.gym_env.sim.pygym_DexGrasp import Gym
# import torch


# class RealmanEnv:
#     """
#     Isaac Gym batched environment
#     支持 n_envs 并行
#     """

#     def __init__(self, config, args):
#         self.env = Gym(args)

#         self.config = config
#         self.n_envs = config.get("n_envs", 8)

#         # ===== 基本参数 =====
#         self.max_steps = config.get("max_steps", 400)
#         # self.table_height = config.get("table_height", 0.0)
#         # self.lift_threshold = config.get("lift_threshold", 0.05)
#         self.success_hold_steps = config.get("success_hold_steps", 5)


#         self.asset_root = config.get("asset_root", "./assets")
#         self.asset_file = config.get("asset_filt", "realman_*.urdf")
#         self.base_pos = [0, 0.25, 0]
#         self.base_orn = [0, 0, 0, 1]
#         self.control_type = config.get("position")
#         self.obs_type = None
#         self.robot_type = config.get("robot_type", "realman")
#         self.d_max = 0.1409
#         self.q_max = -0.91

#         self.env.pre_simulate(self.n_envs, self.asset_root, self.asset_file, self.base_pos, self.base_orn, self.control_type, self.obs_type, self.robot_type)
 
#         # ===== batched 状态变量 =====
#         self.step_count = np.zeros(self.n_envs, dtype=np.int32)
#         self.success_counter = np.zeros(self.n_envs, dtype=np.int32)

#         # 状态必须是 batch
#         self._state = None

#     # ======================================================
#     # Reset 成功的环境再重置
#     # ======================================================
#     def reset(self):

#         self.env.reset_joint_states(env_ids)
#         self.env.reset_object_states(env_ids)

#         self.step_count[:] = 0
#         self.success_counter[:] = 0

#         # self._state = self._init_state()  # 必须返回 batch state

#         # self._apply_state(self._state)

#         return self.get_raw_observation()

#     # ======================================================
#     # Step
#     # ======================================================
#     def step(self, action: np.ndarray):
#         """
#         action shape = (n_envs, action_dim)
#         """
#         #action扩展成24维，前16维就初始关节角就行
#         u1 = action[:, :16]
#         u2 = action[:, 16:23]
#         d = action[:, 23] 
#         q1 = (d/self.d_max * self.q_max).unsqueeze(-1)
#         q2 = -q1.unsqueeze(-1)
#         q3 = q1.unsqueeze(-1)
#         q4 = -q1.unsqueeze(-1)
#         q5 = -q1.unsqueeze(-1)
#         q6 = -q1.unsqueeze(-1)
#         gripper_q = torch.cat([q1, q2, q3, q4, q5, q6], dim=1)
#         u = torch.cat([u1, u2, gripper_q], dim=1)

       
#         # 1️⃣ 应用 batch 动作
#         self.env.step(u, self.control_type, self.obs_type)  

#         # 2️⃣ 更新 batch 状态
#         # self._state = self._update_state()

#         # 3️⃣ 步数增加
#         self.step_count += 1

#         # 4️⃣ 生成观测
#         raw_obs = self.get_raw_observation()

#         # 5️⃣ 计算 reward & success
#         reward, success = self._compute_reward()

#         # 6️⃣ 计算 done
#         done = self._check_done(success)

#         info = {
#             "success": success,
#             "step_count": self.step_count.copy()
#         }

#         return raw_obs, reward, done, info

#     # ======================================================
#     # Reward
#     # ======================================================
#     def _compute_reward(self):

#         # obj_height shape = (n_envs,)
#         middle_point_to_object_distance = self.env.get_right_gripper_to_object_distance()
#         threshold = 0.01

#         success = middle_point_to_object_distance < threshold

#         # 更新 success counter（batch 版本）
#         self.success_counter = np.where(
#             success,
#             self.success_counter + 1,
#             0
#         )

#         stable_success = self.success_counter >= self.success_hold_steps

#         reward = stable_success.astype(np.float32)

#         return reward, stable_success

#     # ======================================================
#     # Done
#     # ======================================================
#     def _check_done(self, success):

#         timeout = self.step_count >= self.max_steps

#         done = np.logical_or(success, timeout)

#         return done

#     # ======================================================
#     # Observation
#     # ======================================================
#     def get_raw_observation(self):

#         return {
#             # 形状必须是 (n_envs, C, H, W)
#             "camera_image": self.env.get_right_wrist_image(),

#             "camera_head_image": self.env.get_head_image(),

#             # (n_envs, 7)
#             "robot_qpos": self.env.get_joint_pos()[:, 16:23].squeeze(-1),

#             # (n_envs, 1)
#             "robot_end_qpos": self.env.get_gripper_width(),

#             # (n_envs, 3)
#             "ee_pos": self.env.get_right_ee_position(),

#             # (n_envs, 3)
#             "ee_quat": self.env.get_right_ee_orientation(),#这里是四元数需要转欧拉角
#         }












