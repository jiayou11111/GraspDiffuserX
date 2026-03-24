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
        action_deg = action_flat * (180.0 / np.pi)
        
        # 格式化打印
        action_str = ", ".join([f"{x:.4f}" for x in action_deg])
        print(f"Step {t}: [{action_str}]")

if __name__ == "__main__":
    main()