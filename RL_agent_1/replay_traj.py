import os
import argparse
from configs.RealmanGrasp_config import RealGraspCfg
from env.TaskRobotEnv import RealmanGraspSingleGym
import time
import torch

def replay_trajectory(traj_path: str, headless: bool = False):
    """
    Replay a single trajectory by setting the environment to the saved states.
    """
    if not os.path.isfile(traj_path):
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    # Load trajectory
    traj = torch.load(traj_path, map_location='cpu')
    print(f"Loaded trajectory with keys: {list(traj.keys())}")
    print(f"Trajectory length: {traj['joint_angles'].shape[0]}")

    # Create environment
    cfg = RealGraspCfg()
    if headless:
        cfg.gymcfg.headless = True
    env = RealmanGraspSingleGym(cfg)

    # Reset to initial state
    obs, _ = env.reset()

    # Set initial object pose if available
    if 'init_obj_pos' in traj and 'init_obj_quat' in traj:
        # Note: May need to implement setting object pose in sim
        print(f"Initial obj pos: {traj['init_obj_pos']}")
        print(f"Initial obj quat: {traj['init_obj_quat']}")

    # Start timing
    start_time = time.time()

    # Replay each step
    for t in range(len(traj['joint_angles'])):
        joint_angles = traj['joint_angles'][t].to(env.device)
        ee_pos = traj['ee_pos'][t].to(env.device)
        ee_quat = traj['ee_quat'][t].to(env.device)
        obj_height = traj['obj_height'][t].to(env.device)

        # 直接使用sim.step()设置目标关节位置
        # 对于位置控制，u应该是目标关节位置
        target_joints = joint_angles.unsqueeze(0)  # [1, num_dofs]

        # 获取控制类型和观察类型
        control_type = env.cfg.all.control_type_sim
        obs_type = env.cfg.all.obs_type_sim

        # 执行仿真步骤
        env.sim.step(target_joints, control_type, obs_type)

        # 渲染
        env.sim.render()
        time.sleep(0.1)  # Slow down for visualization

        print(f"Step {t}: obj_height={obj_height.item():.4f}")

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Replay finished in {execution_time:.2f} seconds")

if __name__ == "__main__":
    # Hardcode for testing - replace with your trajectory path
    traj_path = "trajectories/traj_000000.pt"
    headless = False
    replay_trajectory(traj_path, headless)