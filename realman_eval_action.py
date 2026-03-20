import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

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


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-d', '--device', default='cuda:0')


def main(checkpoint, device):

    #  load checkpoint & workspace
    payload = torch.load(checkpoint, pickle_module=dill)
    cfg = payload['cfg']

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload)

    # get policy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # build EXACT SAME dataset as training
    dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)


    # take FIRST sample (第一条数据，第一个片段)

    sample = get_sample(dataset, demo_id=5, timestep=1)
    obs = sample["obs"]

    obs = {
        k: v.unsqueeze(0).float().to(device)  
        for k, v in obs.items()
    }
    save_obs_images(obs, out_dir="debug_obs")

    print("\n===== OBS =====")
    for k, v in obs.items():
        print(f"{k}: {tuple(v.shape)}")

    # 5. run policy
    with torch.no_grad():
        result = policy.predict_action(obs)
        pre_action = result["action_pred"]
        

    print("\n===== FIRST PRE_ACTION (t=0) =====")
    print(pre_action)


    action_deg = pre_action[0] * 180.0 / torch.pi
    print(action_deg)

def get_sample(dataset, demo_id, timestep):
    rb = dataset.replay_buffer
    episode_ends = rb.episode_ends


    assert demo_id < len(episode_ends)

    start = 0 if demo_id == 0 else episode_ends[demo_id - 1]
    end = episode_ends[demo_id]

    length = end - start
    assert timestep < length, f"timestep {timestep} >= demo length {length}"

    idx = start + timestep
    return dataset[idx]

def save_obs_images(obs, out_dir="debug_obs"):
    """
    保存送入 policy.predict_action(obs) 的所有时间帧图像
    obs: dict, key -> tensor (B, T, 3, H, W)
    """
    os.makedirs(out_dir, exist_ok=True)

    for k, v in obs.items():
        # 只处理 RGB 图像
        if v.ndim == 5 and v.shape[2] == 3:
            # v: (B, T, 3, H, W)
            B, T, _, H, W = v.shape

            for t in range(T):
                img = v[0, t].detach().cpu()  # (3, H, W)

                # (3, H, W) -> (H, W, 3)
                img = img.permute(1, 2, 0)

                # 假设输入已经是 0~1（robomimic / diffusion_policy 默认）
                img = img.clamp(0, 1)
                img = (img * 255).byte().numpy()
                
                img = img[..., ::-1] #BGR格式

                save_path = os.path.join(out_dir, f"{k}_t{t}.png")
                Image.fromarray(img).save(save_path)

                print(f"[saved] {save_path}")

if __name__ == "__main__":
    main()
# python eval_single_sample.py -c data/outputs/latest.ckpt -d cuda:0