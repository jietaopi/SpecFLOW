import torch
import torch.nn as nn

class WassersteinBarycenterFuser(nn.Module):
    def __init__(self, *, eps=1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        mus,
        logvars,
        known_mask,
    ):
        """Masked Wasserstein barycenter fusion for diagonal Gaussians.
        
        Args:
            mus: (B, M, N, D)  # B: Batch, M: 模态数, N: 序列长度, D: 特征维度
            logvars: (B, M, N, D)
            known_mask: (B, M) bool类型，表示对应的模态是否已知/存在
        Returns:
            fused_mu: (B, N, D)
            fused_logvar: (B, N, D)
        """
        assert mus.dim() == 4, ("mus/logvars must have shape (B, M, N, D)")
        assert known_mask.shape == mus.shape[:2], ("known_mask must have shape (B, M)")

        # Uniform weights over known modalities, per sample.
        # 将 known_mask 从 (B, M) 广播到 (B, M, 1, 1) 以匹配 (B, M, N, D)
        mask_f = known_mask.to(dtype=mus.dtype).unsqueeze(-1).unsqueeze(-1)  
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1, 1, 1)
        weights = mask_f / denom  # (B, M, 1, 1)

        fused_mu = (weights * mus).sum(dim=1)  # (B, N, D)

        # Diagonal Gaussian: sigma = exp(0.5 * logvar)
        logvars = logvars.clamp(min=-30.0, max=20.0)
        sigmas = torch.exp(0.5 * logvars).clamp(min=self.eps, max=1e6)
        
        # Wassertein Barycenter 融合公式
        fused_sigma = (weights * sigmas).sum(dim=1).clamp(min=self.eps, max=1e6)  # (B, N, D)
        fused_logvar = (2.0 * torch.log(fused_sigma)).clamp(min=-30.0, max=20.0)
        
        return fused_mu, fused_logvar

    def sample(self, mu, logvar):
        """重参数化采样，获取融合后的 Latent Token"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std