import torch
from torch import nn
import torch.nn.functional as F
from .dit import DiT
from .flow_matching import FlowMatchingInterpolant
from .wb_fuse import WassersteinBarycenterFuser

class LatentDiffusion(nn.Module):
    def __init__(
        self,
        config,
        d_x=512,            # 需要与你光谱 Encoder 输出的 d_model 保持一致
    ):
        super().__init__()
        self.self_condition = config.self_condition

        # 实例化 WB 特征融合器
        self.fuser = WassersteinBarycenterFuser()

        # 实例化 DiT，将分类等无用参数剔除
        self.denoiser = DiT(
            d_x=d_x,        
            d_model=config.d_model,        # DiT block内部隐藏维度
            nhead=config.num_heads,
            num_layers=config.num_layers,
            self_condition=self.self_condition,
        )

        # 实例化 Flow Matching
        self.interpolant = FlowMatchingInterpolant(
            min_t=config.min_t,
            corrupt=config.corrupt,
            timesteps=config.timesteps,
            self_condition=self.self_condition,
            self_condition_prob=0.5,
        )

    def forward(
        self, 
        target_latent, 
        x_1,
    ):
        """
        前向训练流程
        Args:
            target_latent: (B, N, d_model), 需要被预测补全的目标光谱隐变量 (来自目标光谱 Encoder)
            x_1: (B, N, d_model), 已知先验光谱的融合特征
            known_mask: (B, M), 表示对应先验光谱是否存在
        """

        # 3. 对目标光谱隐变量进行破坏加噪
        self.interpolant.device = target_latent.device
        x_1, x_t, x_0, t = self.interpolant.corrupt_batch(x_1)

        # 4. Self-Conditioning 处理
        x_sc = None
        if self.self_condition and torch.rand(1).item() < self.interpolant.self_condition_prob:
            with torch.no_grad():
                x_sc = self.denoiser(
                    x=x_t,
                    t=t,                 # (B, )
                    x_sc=None,
                ).detach()

        # 5. 执行去噪预测
        pred_x = self.denoiser(
            x=x_t,
            t=t,
            x_sc=x_sc,
        )

        # ------------------------------------------------------------------ #
        # 6. Flow Matching Loss (Eq.4, Eq.5)
        #    u_t(z^t|z^1)      = (z^1 - z^t) / (1 - t)
        #    u_theta(z^t, t)   = (z_hat^1 - z^t) / (1 - t)
        #    L = mean ||u_t - u_theta||^2
        #      = (1/(1-t)^2) * mean ||z^1 - z_hat^1||^2
        # ------------------------------------------------------------------ #
        z_1 = x_1
        z_t = x_t
        z_1_hat = pred_x

        one_minus_t = 1.0 - torch.minimum(t, t.new_tensor(0.9))
        one_minus_t = one_minus_t[:, None, None]  # (B, 1, 1)
        gt_vf = (z_1 - z_t) / one_minus_t
        pred_vf = (z_1_hat - z_t) / one_minus_t

        # 当前接口未传入 token_mask / diffuse_mask，先对全部 token 计入损失
        loss_mask = torch.ones_like(z_t[..., 0], dtype=z_t.dtype)
        loss_denom = torch.clamp(loss_mask.sum(dim=-1) * z_t.size(-1), min=1.0)
        x_loss = torch.sum((gt_vf - pred_vf).square() * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        loss = x_loss.mean()

        return {
            "loss": loss,
            "pred_x": pred_x,
            "z_noise": x_0,
            "z_fused": x_1,
        }

    
    @torch.no_grad()
    def sample(
        self,
        device,
        context_fused,           
        cfg_scale=4.0,
        timesteps=None,
    ):
        """Stage-2 推理：通过 Flow Matching 采样补全目标光谱特征。

        Returns:
            final_x: (B, N, D)，采样得到的干净特征
        """
        self.interpolant.device = device

        B, N, D = context_fused.shape

        result = self.interpolant.sample_with_classifier_free_guidance(
            batch_size=B,
            num_tokens=N,
            emb_dim=D,
            model=self.denoiser,
            context=context_fused,
            cfg_scale=cfg_scale,
            timesteps=timesteps,
        )

        # 取最后一步干净预测（条件半）
        final_x = result["clean_traj"][-1]   # (B, N, D)
        return final_x

    @torch.no_grad()
    def sample_unconditional(
        self,
        device,
        batch_size,
        num_tokens,
        emb_dim,
        timesteps=None,
    ):
        """无条件推理：不依赖融合先验，直接从噪声采样。"""
        self.interpolant.device = device

        result = self.interpolant.sample_unconditional(
            batch_size=batch_size,
            num_tokens=num_tokens,
            emb_dim=emb_dim,
            model=self.denoiser,
            timesteps=timesteps,
        )

        final_x = result["clean_traj"][-1]
        return final_x