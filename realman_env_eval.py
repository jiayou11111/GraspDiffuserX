import sys
import os
import isaacgym
import dill
import hydra
import tqdm
import numpy as np

from RL_agent_1.env.TaskRobotEnv import RealmanGraspSingleGym
from RL_agent_1.configs.RealmanGrasp_config import RealGraspCfg

from diffusion_policy.env.robomimic.realman_image_wrapper import RealManImageWrapper
from diffusion_policy.env_runner.realman_image_runner import RealmanImageRunner

import torch
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.pytorch_util import dict_apply

def main(checkpoint, device='cuda:0'):
    device = torch.device(device)

    # 1️⃣ 加载 checkpoint 和 workspace
    payload = torch.load(checkpoint, pickle_module=dill)
    cfg = payload['cfg']

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.to(device)
    policy.eval()

    dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)


    shape_meta = cfg.task.shape_meta
    cfg1 = RealGraspCfg()
    base_env = RealmanGraspSingleGym(cfg1)
    env = RealManImageWrapper(base_env, shape_meta)

    # 3️⃣ 创建评估 runner
    runner = RealmanImageRunner(
        output_dir=None,
        env=env
    )

    # 4️⃣ 运行 rollout
    runner.run(policy)


if __name__ == "__main__":
    # 保证 stdout 不缓冲输出
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

    #在此处直接修改路径
    checkpoint_path = "data/outputs/epoch=0200-val_loss=0.055.ckpt"
    
    main(checkpoint_path)
