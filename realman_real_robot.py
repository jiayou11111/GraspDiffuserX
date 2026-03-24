#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
from collections import deque
from typing import Dict
import numpy as np

import os

from Robotic_Arm.rm_robot_interface import *

from camera_socket import CameraSocket
from camera_socket import RealSenseCameraBuffer

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseImageDataset

PORT = 5000


class RobotArm:
    def __init__(self):

        self.l_robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.r_robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

        self.l_handle = self.l_robot.rm_create_robot_arm("169.254.128.18", 8080)
        self.r_handle = self.r_robot.rm_create_robot_arm("169.254.128.19", 8080)


    def get_l_gripper_state(self):
        state = self.l_robot.rm_get_rm_plus_state_info()
        return state[1]['pos'][0]

    def get_r_gripper_state(self):
        state = self.r_robot.rm_get_rm_plus_state_info()
        return state[1]['pos'][0]


    def get_l_robot_joints(self):
        state = self.l_robot.rm_get_current_arm_state()
        return state[1]['joint']

    def get_r_robot_joints(self):
        state = self.r_robot.rm_get_current_arm_state()
        return state[1]['joint']

    def get_l_robot_pose(self):
        state = self.l_robot.rm_get_current_arm_state()
        return state[1]['pose']
    
    def get_r_robot_pose(self):
        state = self.r_robot.rm_get_current_arm_state()
        return state[1]['pose']
        
    def set_r_joints_angles(self, r_angles):
        self.r_robot.rm_movej_canfd(r_angles, True, 0, 1, 50)

    def set_l_joints_angles(self, l_angles):
        self.l_robot.rm_movej_canfd(l_angles, True, 0, 1, 50) 

    def rm_set_l_gripper_position(self, position):
        self.l_robot.rm_set_gripper_position(position, True, 10)     
 
    def rm_set_r_gripper_position(self, position):
        self.r_robot.rm_set_gripper_position(position, True, 10)

    def rm_l_movej(self, angles, speed=20, acc=0, time=0, radius=1):
        self.l_robot.rm_movej(angles, speed, acc, time, radius)

    def rm_r_movej(self, angles, speed=20, acc=0, time=0, radius=1):
        self.r_robot.rm_movej(angles, speed, acc, time, radius)    


class TimeStampedBuffer:
    def __init__(self, maxlen=3000):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def push(self, timestamp: float, data):
        with self.lock:
            self.buffer.append((timestamp, data))



class ArmStateRecorder(threading.Thread):
    def __init__(self, arm, buffer, freq=50):
        super().__init__(daemon=True)
        self.arm = arm
        self.buffer = buffer
        self.dt = 1.0 / freq
        self.running = True

    def run(self):
        while self.running:
            ts = time.time()

            joint = self.arm.get_r_robot_joints()
            for i in range(len(joint)):
                joint[i] = joint[i] / 180.0 * np.pi  # deg -> rad
            pos = self.arm.get_r_robot_pose()[:3]
            orn = self.arm.get_r_robot_pose()[3:]
            gripper = [self.arm.get_r_gripper_state()/1000.0]
            
            q = np.array(joint + gripper + pos + orn, dtype=np.float32)
            self.buffer.push(ts, {'q': q, 'robot_receive_timestamp': ts})

            time.sleep(self.dt)

    def stop(self):
        self.running = False


class SharedSequenceBuffer:
    def __init__(self, capacity=5000, n_obs_steps=5):
        self.capacity = capacity
        self.obs = [None] * capacity
        self.action = [None] * capacity
        self.timestamp = np.zeros(capacity, dtype=np.float64)
        self.ptr = 0
        self.size = 0
        self.lock = threading.Lock()
        self.n_obs_steps = n_obs_steps
        # ===== 新增：图片保存目录 =====
        self.save_dir = "saved_images"
        os.makedirs(self.save_dir, exist_ok=True)

    def add(self, obs: Dict, action: np.ndarray, timestamp: float):
        with self.lock:
            self.obs[self.ptr] = obs
            self.action[self.ptr] = action
            self.timestamp[self.ptr] = timestamp

            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def clear(self):
        """
        Clear all stored data and reset buffer state.
        """
        with self.lock:
            self.obs = [None] * self.capacity
            self.action = [None] * self.capacity
            self.timestamp = np.zeros(self.capacity, dtype=np.float64)
            self.ptr = 0
            self.size = 0

    def get_latest_obs(self):
        """
        Get the latest observation stored in the buffer.
        Returns:
            obs (Dict) or None if buffer is empty
        """
        with self.lock:
            if self.size == 0:
                return None
            latest_idx = (self.ptr - 1) % self.capacity
            raw_obs = self.obs[latest_idx]


        # ========= robot state =========
        q_all = raw_obs["q"]  # (T, q_dim)

        obs = {
            "robot0_qpos": q_all[:, 0:7],          # (T, 7)
            "robot0_gripper_qpos": q_all[:, 7:8],  # (T, 1)
            "robot0_end_pos": q_all[:, 8:11],      # (T, 3)
            "robot0_end_rxryrz": q_all[:, 11:14],  # (T, 3)
        }

        # ========= images =========
        def resize_seq(img_seq):
            return np.stack(
                [
                    cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
                    # cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
                    for img in img_seq
                ],
                axis=0
            ).astype(np.uint8)

        prefix = f"idx_{latest_idx:06d}"
        agentview_image = resize_seq(raw_obs["cam_right"])
        agentview_head_image = resize_seq(raw_obs["cam_head"])


        # self.save_image_seq(agentview_image, "agentview_image", prefix)
        self.save_image_seq(agentview_head_image, "agentview_head_image", prefix)

        obs["agentview_image"] = agentview_image
        obs["agentview_head_image"] = agentview_head_image
        return obs
    
    def save_image_seq(self, img_seq, subdir, prefix):

        save_path = os.path.join(self.save_dir, subdir, prefix)
        os.makedirs(save_path, exist_ok=True)

        for t, img in enumerate(img_seq):
            cv2.imwrite(
                os.path.join(save_path, f"{t:03d}.png"),
                img
            )


    def __len__(self):
        return self.size


class PolicyRunner(threading.Thread):
    def __init__(
        self,
        arm,
        policy,
        shared_buffer: SharedSequenceBuffer,
        cfg,
        device,
    ):
        super().__init__(daemon=True)
        self.arm = arm
        self.policy = policy
        self.shared_buffer = shared_buffer
        self.cfg = cfg
        self.device = device

        self.running = False
        self.exit_flag = False


    def start_policy(self):
        # print("[PolicyRunner] start")
        self.policy.reset()
        self.running = True

    def stop_policy(self):
        print("[PolicyRunner] stop")
        self.running = False

    def shutdown(self):
        self.exit_flag = True

    def run(self):
        while not self.exit_flag:
            if not self.running:
                time.sleep(0.01)
                continue

            obs = self.shared_buffer.get_latest_obs()

            if obs is None:
                time.sleep(0.01)
                continue

            # with torch.no_grad():
            obs_dict_np = get_real_obs_dict(
                env_obs=obs,
                shape_meta=self.cfg.task.shape_meta
            )
            obs_dict = dict_apply(
                obs_dict_np,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
            )

            result = self.policy.predict_action(obs_dict)
            action = result['action'][0].detach().cpu().numpy()

            # H-step absolute poses
            target_poses = [a for a in action]
            print("target_poses len:", target_poses)

            for i, pose in enumerate(target_poses):
                if i==7:
                    joints = pose[:7]  # 前7个是关节角度（弧度）
                    gripper = pose[7]  # 第8个是夹爪位置
                                    
                    # 如果需要转换为角度
                    joints_deg = joints * 180.0 / np.pi
                    gripper1 = gripper * 1000.0  # m -> mm

                    self.arm.rm_r_movej(joints_deg, 20, 0, 0, 1)
                    self.arm.rm_set_r_gripper_position(int(gripper1))
                continue

                # if i == 2:
                #     continue
            print("finished executing action sequence")


class MultiCameraDatasetBuilder:
    def __init__(self, arm_buffer: TimeStampedBuffer, shared_buffer: SharedSequenceBuffer, cam_buffers: Dict[str, RealSenseCameraBuffer], n_obs_steps=5, frequency=10.0):
        self.arm_buffer = arm_buffer
        self.shared_buffer = shared_buffer
        self.cam_buffers = cam_buffers
        self.n_obs_steps = n_obs_steps
        self.frequency = frequency
        self.prev_q = None

    def step(self):
        # print("start")
        # 摄像头最新时间戳
        last_timestamps = []
        for buf in self.cam_buffers.values():
            with buf.lock:
                if len(buf.buffer) == 0:
                    print("no camera data")
                    return
                last_timestamps.append(buf.buffer[-1][0])
        last_timestamp = max(last_timestamps)

        # 对齐序列时间戳
        dt = 1.0 / self.frequency
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)

        # 获取机器人数据
        all_robot_data = list(self.arm_buffer.buffer)
        if not all_robot_data:
            print("no robot data")
            return
        robot_timestamps = np.array([x[0] for x in all_robot_data])
        this_idxs = []
        for t in obs_align_timestamps:
            idxs = np.nonzero(robot_timestamps <= t)[0]
            this_idx = idxs[-1] if len(idxs) > 0 else 0
            this_idxs.append(this_idx)


        # n_obs_steps帧数据
        q_seq = []
        for idx in this_idxs:
            q_seq.append(all_robot_data[idx][1]['q'])
        q_seq = np.stack(q_seq, axis=0)  # shape: (n_obs_steps, q_dim)

        q = all_robot_data[this_idxs[-1]][1]['q'] 

        if self.prev_q is None:
            self.prev_q = q
            return
        # print(q_seq.shape)

        # 摄像头对齐帧
        cam_images = {}
        for cid, buf in self.cam_buffers.items():
            imgs = []
            for t in obs_align_timestamps:
                img = buf.get_closest_before(t)
                if img is None:
                    return
                imgs.append(img)
            cam_images[cid] = np.array(imgs)  # shape: (n_obs_steps, H, W, C)

        obs = {'q': q_seq, **cam_images}
        action = self.prev_q
        self.shared_buffer.add(obs, action, last_timestamp)
        self.prev_q = q


OmegaConf.register_new_resolver("eval", eval, replace=True)


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-d', '--device', default='cuda:0')

def main(checkpoint, device):
    payload = torch.load(checkpoint, pickle_module=dill)
    cfg = payload['cfg']

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload)

    # 2. get policy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # 3. build EXACT SAME dataset as training

    dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)

    BUFFER_SIZE = 1500
    N_OBS_STEPS = 1
    time.sleep(3)

    arm = RobotArm()

    arm_buffer = TimeStampedBuffer(maxlen=150)
    arm_recorder = ArmStateRecorder(arm, arm_buffer, freq=100) #100hz运行
    arm_recorder.start()


    CAMERAS = [
        {"name": "cam_head",  "port": 5001, "c_ip": "169.254.128.20"},
        {"name": "cam_right", "port": 5002, "c_ip": "169.254.128.20"},
        # {"name": "cam_left",  "port": 5003, "c_ip": "169.254.128.20"},
    ]
    
    cam_buffers = {}
    for cam in CAMERAS:

        cam_buf =  RealSenseCameraBuffer(maxlen=150)

        client = CameraSocket(cam,cam_buf) #不断的在运行
    
        cam_buffers[cam["name"]] = cam_buf

        client.start()


    shared_buffer = SharedSequenceBuffer(capacity=BUFFER_SIZE, n_obs_steps=N_OBS_STEPS)
    builder = MultiCameraDatasetBuilder(arm_buffer, shared_buffer, cam_buffers, n_obs_steps=N_OBS_STEPS, frequency=10.0) 


    policy_runner = PolicyRunner(
        arm=arm,
        policy=policy,
        shared_buffer=shared_buffer,
        cfg=cfg,
        device=device,
    )
    policy_runner.start()

    print("Ready. Human mode by default.")
    policy_runner.start_policy()
    time.sleep(0.02)
    print("111111111111")


    while True:

        builder.step()
        time.sleep(0.1)



if __name__ == "__main__":
    main()
