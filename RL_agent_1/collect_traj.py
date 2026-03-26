import os
import re
import argparse
from typing import Optional, List, Dict, Any

from configs.RealmanGrasp_config import RealGraspCfg
from env.TaskRobotEnv import RealmanGraspSingleGym

import torch
import torch.nn.functional as F
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.configs import rslCfgPPO
from rsl_rl.utils import class_to_dict


def _find_latest_checkpoint(log_dir: str) -> Optional[str]:
    if not os.path.isdir(log_dir):
        return None
    pattern = re.compile(r"model_(\d+)\.pt$")
    cands: List[str] = []
    for fn in os.listdir(log_dir):
        if pattern.search(fn):
            cands.append(fn)
    if not cands:
        return None
    cands.sort(key=lambda x: int(pattern.search(x).group(1)))
    return os.path.join(log_dir, cands[-1])


# =========================
# ✅ 你需要实现的成功判定
# =========================
def is_success(traj: Dict[str, Any]) -> torch.Tensor:
    """
    traj:
        dict containing trajectory info for each env
    return:
        success mask: shape [num_envs]
    """

    # 判定条件：物体提升高度 > 0.05 且 夹爪宽度 < 0.07
    obj_height = traj["obj_height"]  # [T, N]
    # gripper_width = traj["gripper_width"]  # [T, N]
    
    max_height = obj_height.max(dim=0).values  # [N]
    # min_gripper_width = gripper_width.min(dim=0).values  # [N]
    
    height_success = max_height > 0.05
    # width_success = min_gripper_width < 0.07
    
    success = height_success #& width_success
    
    return success

def collect_trajectories(
    model_path: Optional[str] = None,
    episodes_per_env: int = 20,
    save_dir: str = "./trajectories",
):

    os.makedirs(save_dir, exist_ok=True)

    cfg = RealGraspCfg()
    train_cfg = class_to_dict(rslCfgPPO())
    env = RealmanGraspSingleGym(cfg)

    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=None,
        device=str(env.device),
    )

    if model_path is None:
        model_path = _find_latest_checkpoint("logs_important")
        if model_path is None:
            raise FileNotFoundError("未找到 checkpoint")

    checkpoint = torch.load(model_path, map_location=env.device)

    runner.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
    runner.alg.actor_critic.to(env.device)

    policy = runner.get_inference_policy(device=env.device)

    num_envs = env.num_envs
    max_len = int(env.max_episode_length)
    
    total_episodes = episodes_per_env * num_envs
    
    # ===== 频率对齐：240Hz -> 10Hz =====
    # 采样间隔：240 / 10 = 24 步
    sample_interval = 1

    saved_count = 0
    total_done = 0

    # 每个环境需要收集的episode数
    env_episodes_needed = torch.full((num_envs,), episodes_per_env, dtype=torch.int32, device=env.device)
    env_episodes_collected = torch.zeros((num_envs,), dtype=torch.int32, device=env.device)

    # 当前活跃的环境掩码（正在收集轨迹的环境）
    active_envs = torch.ones((num_envs,), dtype=torch.bool, device=env.device)

    # 为每个环境维护独立的轨迹缓冲区
    traj_buffers = []
    obj_height_history = []  # 用于成功检测的高度历史（每步更新）
    env_init_obj_pos = []
    env_init_obj_quat = []
    env_step_counts = torch.zeros((num_envs,), dtype=torch.int32, device=env.device)

    for i in range(num_envs):
        traj_buffers.append({
            "joint_angles": [],  # 关节角（只在采样步记录）
            "robot_qpos": [],    # 特定关节角 (索引16:23)（只在采样步记录）
            "ee_pos": [],        # 末端位置（只在采样步记录）
            "ee_quat": [],       # 末端旋转（只在采样步记录）
            "head_rgb": [],      # 头部RGB图像（只在采样步记录）
            "hand_rgb": [],      # 手部RGB图像（只在采样步记录）
            "obj_height": [],    # 物体提升高度（只在采样步记录）
            "gripper_width": [], # 夹爪宽度
        })
        obj_height_history.append([])  # 用于成功检测的高度历史（每步更新）
        env_init_obj_pos.append(None)
        env_init_obj_quat.append(None)

    # 重置所有环境开始收集
    obs, _ = env.reset()

    # 获取初始物体位姿
    all_init_obj_pos = env.sim.get_top_obj_initial_position()  # [num_envs, 3]
    all_init_obj_quat = env.sim.get_top_obj_initial_quaternion()  # [num_envs, 4]

    # 为每个活跃环境设置初始位姿
    for i in range(num_envs):
        if active_envs[i]:
            env_init_obj_pos[i] = all_init_obj_pos[i].clone()
            env_init_obj_quat[i] = all_init_obj_quat[i].clone()

    step_count = 0  # 全局步数计数器（用于采样频率）
    # 持续运行直到所有环境都收集了足够的episode
    while (env_episodes_collected < env_episodes_needed).any():
        # 执行一步仿真
        with torch.no_grad():
            actions = policy(obs)

        next_obs, _, _, _, _ = env.step(actions)

        # ===== 存数据（按采样频率） =====
        # 仅在采样间隔时保存完整数据
        if step_count % sample_interval == 0:
            # 获取图像（所有环境）
            head_rgb_list = env.sim.get_head_image()  # 列表，每个环境一个tensor
            hand_rgb_list = env.sim.get_right_wrist_image()  # 列表

            # Stack并提取RGB通道
            head_img = torch.stack(head_rgb_list, dim=0)[..., :3]  # [num_envs, H, W, 3]
            hand_img = torch.stack(hand_rgb_list, dim=0)[..., :3]  # [num_envs, H, W, 3]

            # 排列维度为 [num_envs, 3, H, W]
            head_img = head_img.permute(0, 3, 1, 2).contiguous().float()
            hand_img = hand_img.permute(0, 3, 1, 2).contiguous().float()

            # 标准化
            head_img = head_img.to(torch.float32) / 255.0
            hand_img = hand_img.to(torch.float32) / 255.0

            # 插值到固定大小
            head_img = F.interpolate(
                head_img,
                size=(240, 320),
                mode="bilinear",
                align_corners=False
            )
            hand_img = F.interpolate(
                hand_img,
                size=(240, 320),
                mode="bilinear",
                align_corners=False
            )

            # 转回 [num_envs, H, W, 3] 便于保存
            head_img = head_img.permute(0, 2, 3, 1).contiguous()
            hand_img = hand_img.permute(0, 2, 3, 1).contiguous()

            # 获取关节角、末端位置/旋转、物体高度、夹爪宽度
            joint_angles = env.sim.get_joint_pos()  # [num_envs, num_dofs]
            robot_qpos = joint_angles[:, 1:8].squeeze(-1)  # 特定关节角
            ee_pos = env.sim.get_right_ee_position()  # [num_envs, 3]
            ee_quat = env.sim.get_right_ee_orientation()  # [num_envs, 4]
            obj_height = env.sim.get_obj_height()  # [num_envs]
            gripper_width = env.sim.get_gripper_width()  # [num_envs]
            # 夹爪宽度归一化到 (0, 1)
            min_width = 0.0
            max_width = 0.1409
            normalized_gripper_width = (gripper_width - min_width) / (max_width - min_width)
            normalized_gripper_width = torch.clamp(normalized_gripper_width, 0.0, 1.0)

            # 将数据添加到活跃环境的缓冲区
            for i in range(num_envs):
                if active_envs[i]:
                    traj_buffers[i]["joint_angles"].append(joint_angles[i].clone())
                    traj_buffers[i]["robot_qpos"].append(robot_qpos[i].clone())
                    traj_buffers[i]["ee_pos"].append(ee_pos[i].clone())
                    traj_buffers[i]["ee_quat"].append(ee_quat[i].clone())
                    traj_buffers[i]["head_rgb"].append(head_img[i].clone())
                    traj_buffers[i]["hand_rgb"].append(hand_img[i].clone())
                    traj_buffers[i]["obj_height"].append(obj_height[i].clone())
                    traj_buffers[i]["gripper_width"].append(normalized_gripper_width[i].clone())
                    # 同时更新成功检测用的高度历史
                    obj_height_history[i].append(obj_height[i].clone())
        else:
            # 非采样步骤：只获取物体高度用于成功检测，不存储到缓冲区
            obj_height = env.sim.get_obj_height()  # [num_envs]
            # 更新成功检测用的高度历史
            for i in range(num_envs):
                if active_envs[i]:
                    obj_height_history[i].append(obj_height[i].clone())

        step_count += 1

        # ===== 每步检查成功条件 =====
        # 检查每个活跃环境是否成功
        for i in range(num_envs):
            if active_envs[i] and len(obj_height_history[i]) > 0:
                # 使用高度历史进行成功检测（每步更新）
                obj_height_list = obj_height_history[i]
                # 检查当前物体高度是否达到成功条件
                # 使用整个轨迹的最大高度来判断成功
                obj_height_tensor = torch.stack(obj_height_list, dim=0)  # [T]
                max_height = obj_height_tensor.max().item()

                # 成功条件：物体提升高度 > 0.05
                if max_height > 0.05:
                    # 环境成功，保存轨迹
                    print(f"Env {i} succeeded at step {step_count}, max_height={max_height:.4f}")

                    # 构建完整的轨迹数据
                    single_traj = {}

                    # 找到所有数据的最小长度（处理不同数据类型的采样频率不同）
                    min_length = float('inf')
                    for key in traj_buffers[i]:
                        if len(traj_buffers[i][key]) > 0:
                            min_length = min(min_length, len(traj_buffers[i][key]))

                    if min_length < float('inf'):
                        # 只保存所有数据类型都有的那些步
                        for key in traj_buffers[i]:
                            if len(traj_buffers[i][key]) > 0:
                                # 取最后min_length个数据点（最近的数据）
                                data_list = traj_buffers[i][key][-min_length:]
                                data = torch.stack(data_list, dim=0)
                                if data.dim() > 1:
                                    # 如果是多维数据，移除额外的维度
                                    data = data.squeeze(1) if data.shape[1] == 1 else data
                                single_traj[key] = data.cpu()

                    # 添加初始物体位姿
                    single_traj["init_obj_pos"] = env_init_obj_pos[i].cpu()
                    single_traj["init_obj_quat"] = env_init_obj_quat[i].cpu()

                    save_path = os.path.join(save_dir, f"traj_{saved_count:06d}.pt")
                    torch.save(single_traj, save_path)

                    saved_count += 1
                    env_episodes_collected[i] += 1
                    total_done += 1

                    # 立即重置该环境（异步重置）
                    env.robot.reset_ids(torch.tensor([i], device=env.device))

                    # 获取新的初始物体位姿
                    new_init_obj_pos = env.sim.get_top_obj_initial_position()[i].clone()
                    new_init_obj_quat = env.sim.get_top_obj_initial_quaternion()[i].clone()

                    # 重置该环境的缓冲区和状态
                    traj_buffers[i] = {
                        "joint_angles": [],
                        "robot_qpos": [],
                        "ee_pos": [],
                        "ee_quat": [],
                        "head_rgb": [],
                        "hand_rgb": [],
                        "obj_height": [],
                        "gripper_width": [],
                    }
                    obj_height_history[i] = []  # 重置高度历史
                    env_init_obj_pos[i] = new_init_obj_pos
                    env_init_obj_quat[i] = new_init_obj_quat
                    env_step_counts[i] = 0

                    # 检查该环境是否已完成配额
                    if env_episodes_collected[i] >= env_episodes_needed[i]:
                        print(f"Env {i} completed quota ({env_episodes_needed[i]} episodes), deactivating")
                        active_envs[i] = False
                    # 否则环境保持活跃状态，继续收集

        # 检查是否达到最大步数
        for i in range(num_envs):
            if active_envs[i] and env_step_counts[i] >= max_len:
                # 达到最大步数仍未成功，放弃当前轨迹并重置环境
                print(f"Env {i} reached max steps ({max_len}) without success, resetting")
                env.robot.reset_ids(torch.tensor([i], device=env.device))

                # 获取新的初始物体位姿
                new_init_obj_pos = env.sim.get_top_obj_initial_position()[i].clone()
                new_init_obj_quat = env.sim.get_top_obj_initial_quaternion()[i].clone()

                # 重置该环境的缓冲区和状态
                traj_buffers[i] = {
                    "joint_angles": [],
                    "robot_qpos": [],
                    "ee_pos": [],
                    "ee_quat": [],
                    "head_rgb": [],
                    "hand_rgb": [],
                    "obj_height": [],
                    "gripper_width": [],
                }
                obj_height_history[i] = []  # 重置高度历史
                env_init_obj_pos[i] = new_init_obj_pos
                env_init_obj_quat[i] = new_init_obj_quat
                env_step_counts[i] = 0
                # 环境保持活跃状态，继续收集

        # 更新活跃环境的步数计数
        for i in range(num_envs):
            if active_envs[i]:
                env_step_counts[i] += 1

        obs = next_obs

        # 打印进度
        if step_count % (sample_interval * 10) == 0:
            active_count = active_envs.sum().item()
            success_rate = saved_count / total_done if total_done > 0 else 0
            print(f"Step {step_count}: Active envs={active_count}, "
                  f"Saved={saved_count}/{total_episodes} ({success_rate:.1%}), "
                  f"Progress={total_done/total_episodes*100:.1f}%")

    # 最终打印
    print(single_traj["head_rgb"])
    print(f"Collection finished: Saved {saved_count}/{total_episodes} trajectories")

if __name__ == "__main__":
    # Hardcode for testing due to gymutil argument conflict
    episodes_per_env = 5
    save_dir = "./trajectories"
    model_path = None

    collect_trajectories(
        model_path=model_path,
        episodes_per_env=episodes_per_env,
        save_dir=save_dir,
    )