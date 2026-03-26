import sys
import torch
import dill
import hydra
import click
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseImageDataset
import torch

import os
import numpy as np
from PIL import Image
import shutil
from diffusion_policy.real_world.real_inference_util import get_real_obs_dict
from diffusion_policy.common.pytorch_util import dict_apply
import cv2

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, device):

    # 1. Load checkpoint & config
    # 使用 dill 加载 pickle 文件，确保能加载 lambda 函数等复杂结构
    payload = torch.load(checkpoint, pickle_module=dill)
    cfg = payload['cfg']

    # 2. Build dataset
    # 实例化数据集主要是为了获取底层的 replay_buffer
    dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
    rb = dataset.replay_buffer

    # 3. Random Sample a Demo
    # 随机选择一个演示片段
    n_demos = len(rb.episode_ends)
    demo_id = np.random.randint(0, n_demos)
    
    print("=" * 60)
    print(f"随机选择的 Demo ID: {demo_id} (总共 {n_demos} 个)")
    print("=" * 60)

    # 计算该 Demo 在 ReplayBuffer 中的绝对起止索引
    start_idx = 0 if demo_id == 0 else rb.episode_ends[demo_id - 1]
    end_idx = rb.episode_ends[demo_id]
    demo_len = end_idx - start_idx
    
    print(f"Demo Length: {demo_len} frames")
    print(f"Buffer Global Range: [{start_idx}, {end_idx})")

    # 准备保存图片的文件夹
    save_dir = os.path.join("save_img", f"demo_{demo_id}")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[Info] 图片将保存在: {save_dir}")

    # -------------------------------------------------------------
    # 4. 处理 Demo 的所有帧观测数据 (Observation)
    # -------------------------------------------------------------
    print(f"\n[Demo {demo_id} 观测数据 (Observation)]")
    
    # 获取需要观测的 Key 列表 (从 Config 中读取)
    obs_meta = cfg.task.dataset.shape_meta.obs
    target_obs_keys = list(obs_meta.keys())
    
    # 建立 Key 到 ReplayBuffer 实际路径的映射
    # 因为 RB 内可能存储为 "obs/image" 或直接 "image"
    rb_key_map = {}
    for k in target_obs_keys:
        if k in rb.keys():
            rb_key_map[k] = k
        elif f"obs/{k}" in rb.keys():
            rb_key_map[k] = f"obs/{k}"
            
    # 从 ReplayBuffer 切片获取数据 (Numpy backend)
    # 这样获取的是最原始的、未经过 transform 的数据
    demo_data = {}
    for key_name, rb_key in rb_key_map.items():
        demo_data[key_name] = rb[rb_key][start_idx:end_idx]

    # 遍历每一帧进行打印/保存
    for i in range(demo_len):
        global_idx = start_idx + i
        print(f"\n>>> Frame {i} (Global Index: {global_idx}):")
        
        for key in target_obs_keys:
            if key not in demo_data:
                continue
                
            value = demo_data[key][i] # 取第 i 帧
            
            # --- 处理图像数据 ---
            is_image = False
            img_np = None
            
            # 判断逻辑：维度为3 且 (C=3 或 last_dim=3)
            # ReplayBuffer 里的图片通常是 (C,H,W) 或 (H,W,C)
            if value.ndim == 3:
                c, h, w = value.shape
                if c == 3: # (3, H, W) -> 转 (H, W, 3)
                    img_np = np.transpose(value, (1, 2, 0))
                    is_image = True
                elif value.shape[2] == 3: # (H, W, 3)
                    img_np = value
                    is_image = True
            
            if is_image:
                # 归一化处理：如果是 float [0,1]，转 uint8 [0,255]
                if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                    # 裁减到合法范围
                    img_np = np.clip(img_np, 0, 1)
                    img_np = (img_np * 255).astype(np.uint8)
                
                # 保存图片
                save_path = os.path.join(save_dir, f"frame_{i}_{key}.png")
                Image.fromarray(img_np).save(save_path)
                print(f"  {key}: [Image Saved] -> {save_path}")
            
            else:
                # --- 处理低维数据 ---
                # 展平数组
                flat_list = np.array(value).flatten().tolist()
                
                # 单位转换：如果是关节角度 (qpos, joint)，弧度 -> 角度
                if 'qpos' in key or 'joint' in key:
                    val_strs = [f"{(x * 180.0 / np.pi):.4f}" for x in flat_list]
                    unit_hint = "(Degree)"
                else:
                    val_strs = [f"{x:.4f}" for x in flat_list]
                    unit_hint = ""

                # 格式化打印: [val, val, val]
                formatted_str = "[" + ", ".join(val_strs) + "]"
                print(f"  {key} {unit_hint}: {formatted_str}")

    # -------------------------------------------------------------
    # 5. 打印 Demo 的全部 Action (Ground Truth) - 转换为角度
    # -------------------------------------------------------------
    print(f"\n[Demo {demo_id} 全部 Action 数据 (Degree 角度, Ground Truth)]")
    print(f"Total Steps: {demo_len}\n")

    # 从 ReplayBuffer 获取 Action 切片
    # 注意：ReplayBuffer 中的 Action 通常直接对齐到 Obs 的时间步
    msg_action = rb['action'][start_idx:end_idx]

    for t, action in enumerate(msg_action):
        # 展平
        action_flat = np.array(action).flatten()
        
        # 弧度 -> 角度
        # action_deg = action_flat * (180.0 / np.pi)
        
        # 格式化打印
        action_str = ", ".join([f"{x:.4f}" for x in action_flat])
        print(f"Step {t}: [{action_str}]")

    # 6. 使用模型预测 Action 并打印 (Degree 角度, Predicted)
    # -------------------------------------------------------------
    print(f"\n[Demo {demo_id} 预测 Action 数据 (Degree 角度, Predicted)]")
    print(f"Total Predicted Steps: {demo_len - cfg.policy.n_obs_steps}\n")

    # 实例化 workspace 和 policy
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.to(device)
    policy.eval()

    # 准备观测数据用于预测 (滑动窗口预测每个动作)
    n_obs_steps = cfg.policy.n_obs_steps
    pred_actions = []

    for i in range(n_obs_steps, demo_len):
        # 构建 env_obs dict (使用前 n_obs_steps 帧观测预测第 i 帧的动作)
        env_obs = {}
        for key in target_obs_keys:
            if key in demo_data:
                env_obs[key] = demo_data[key][i - n_obs_steps:i]

        obs_dict_np = get_real_obs_dict(
            env_obs=env_obs,
            shape_meta=cfg.task.shape_meta
        )
        obs_dict = dict_apply(
            obs_dict_np,
            lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
        )
        print_obs_dict(obs_dict, "TORCH_OBS")
        inspect_and_save_obs(obs_dict)
        result = policy.predict_action(obs_dict)
        pred_action = result['action'][0].detach().cpu().numpy()
        pred_actions.append(pred_action)
        
    for t, action_seq in enumerate(pred_actions):
        for j, action in enumerate(action_seq):
            # 展平
            action_flat = action.flatten()
            
            # 弧度 -> 角度
            # action_deg = action_flat * (180.0 / np.pi)
            
            # 格式化打印
            action_str = ", ".join([f"{x:.4f}" for x in action_flat])
            print(f"Step {t + n_obs_steps + j}: [{action_str}]")

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
def inspect_and_save_obs(obs_dict, save_dir="compare_eval"):
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

if __name__ == "__main__":
    main()