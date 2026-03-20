# This file is created to replay dataset actions in the simulation environment.

import sys
import os
import isaacgym  # Must import before torch
import h5py
import numpy as np
import time

# Add NexusMinds_RL to path to find env and configs
sys.path.append(os.path.join(os.getcwd(), "NexusMinds_RL"))

from env.TaskRobotEnv import RealmanGraspSingleGym
from configs.LinkerHandGrasp_config import LinkGraspCfg
from diffusion_policy.env.robomimic.realman_image_wrapper import RealManImageWrapper

def replay_dataset():
    dataset_path = "data/robomimic/datasets/lift/mh/pink_weita_delete0data_initdofchange_camerachange_BGR_240320.hdf5"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    # Load dataset
    print(f"Opening dataset: {dataset_path}")
    f = h5py.File(dataset_path, 'r')
    demos = list(f['data'].keys())
    # Sort demos naturally if possible (demo_0, demo_1, ...)
    try:
        demos.sort(key=lambda x: int(x.split('_')[1]))
    except:
        pass
    print(f"Found {len(demos)} demos.")

    # Initialize Environment
    print("Initializing Environment...")
    cfg = LinkGraspCfg()
    
    # Force number of environments to 1 for replay
    cfg.all.num_envs = 1

    base_env = RealmanGraspSingleGym(cfg)
    
    # Shape meta for wrapper (assuming 8 dof action)
    shape_meta = {
        "action": {"shape": (8,)},
        "obs": {
            "agentview_image": {"shape": (3, 240, 320)},
            "agentview_head_image": {"shape": (3, 240, 320)},
            "robot0_qpos": {"shape": (7,)},
            "robot0_gripper_qpos": {"shape": (1,)},
        }
    }
    
    env = RealManImageWrapper(base_env, shape_meta)
    env.reset()
    
    print("Environment initialized. Starting replay...")

    for demo_key in demos:
        print(f"Playing {demo_key} ...")
        
        # Get actions
        if 'actions' in f[f'data/{demo_key}']:
            actions = f[f'data/{demo_key}/actions'][:]
        else:
            print(f"No actions found in {demo_key}")
            continue
            
        print(f"  Action shape: {actions.shape}")

        # Reset env for new episode
        env.reset()
        
        # Iterate actions
        for i in range(len(actions)):
            action = actions[i]
            
            # Action shape is (8,). Wrapper expects (n_envs, 8) i.e. (1, 8)
            action_batch = action[None, :] 
            print("action_batch:", action_batch)    
            # Step environment
            obs, privileged_obs, reward, done, info = env.step(action_batch)
            # head_img = obs["agentview_head_image"][0]   # (3,240,320)

            # save_dir = "data/replay_head_img"
            # os.makedirs(save_dir, exist_ok=True)

            # np.save(
            #     os.path.join(save_dir, f"{demo_key}_step_{i}.npy"),
            #     head_img
            # )
                        
            if i % 10 == 0:
                print(f"  Step {i}/{len(actions)}", end='\r')
                
        print(f"  Step {len(actions)}/{len(actions)} Done.")
        time.sleep(0.5)

    print("Replay finished.")
    f.close()

if __name__ == "__main__":
    replay_dataset()