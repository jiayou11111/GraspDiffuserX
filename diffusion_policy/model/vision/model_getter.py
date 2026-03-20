import math
import torch
import torch.nn as nn
import torchvision

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model

class LoRA(nn.Module):
    """
    针对 Query (Q) 和 Value (V) 的低秩适应 (LoRA) 模块
    参考自 dinov3-finetune 项目
    """
    def __init__(
        self,
        qkv: nn.Linear,
        linear_a_q: nn.Linear,
        linear_b_q: nn.Linear,
        linear_a_v: nn.Linear,
        linear_b_v: nn.Linear,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x) -> torch.Tensor:
        # 1. 计算原始 QKV (冻结的)
        qkv = self.qkv(x)  # Shape: (B, N, 3 * dim)

        # 2. 计算 LoRA 增量 (可训练的)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))

        # 3. 将增量加回原始输出
        # DINOv2 的 qkv 布局通常是 [q, k, v]
        # 更新 Q 部分
        qkv[:, :, : self.dim] += new_q
        # 更新 V 部分
        qkv[:, :, -self.dim :] += new_v

        return qkv

class DinoV2Shim(torch.nn.Module):
    def __init__(self, name, pretrained=True, freeze=True, use_lora=False, lora_rank=4):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', name, pretrained=pretrained)
        if freeze or use_lora:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        
        if use_lora:
            self._inject_lora(lora_rank)
    
    def _inject_lora(self, r: int):
        """遍历 Backbone，将 Attention 的 QKV 层替换为 LoRA 包装层"""
        print(f"Injecting LoRA with rank {r} into DINOv2 backbone...")
        # 遍历 DINOv2 的 Transformer Blocks
        
        # Use getattr to avoid static analysis errors if self.model is inferred as object
        blocks = getattr(self.model, 'blocks', [])
        
        for i, block in enumerate(blocks):
            # 获取原始的 qkv 线性层
            w_qkv_linear = block.attn.qkv
            dim = w_qkv_linear.in_features

            # 创建 LoRA 所需的 A 和 B 矩阵 (默认 requires_grad=True)
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)

            # --- 初始化权重 ---
            # A 矩阵使用 Kaiming 初始化
            nn.init.kaiming_uniform_(w_a_linear_q.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(w_a_linear_v.weight, a=math.sqrt(5))
            # B 矩阵初始化为 0 (确保初始状态下模型输出与预训练模型一致)
            nn.init.zeros_(w_b_linear_q.weight)
            nn.init.zeros_(w_b_linear_v.weight)

            # 替换原始层
            block.attn.qkv = LoRA(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        
        print(f"LoRA injected into {len(blocks)} blocks.")

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            # Try to return CLS token if available
            if 'x_norm_clstoken' in output:
                return output['x_norm_clstoken']
            elif 'x_clstoken' in output:
                return output['x_clstoken']
        return output

def get_dinov2(name, pretrained=True, freeze=True, use_lora=False, lora_rank=4):
    return DinoV2Shim(name, pretrained=pretrained, freeze=freeze, use_lora=use_lora, lora_rank=lora_rank)
