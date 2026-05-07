import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMLP(nn.Module):
    def __init__(self, token_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, token_dim),
        )
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x):
        # 残差连接有助于稳定深层特征的提取
        return self.norm(x + self.net(x))


class Decoder(torch.nn.Module):
    def __init__(self, config, d_x=512):
        super().__init__()
        self.d_x = d_x
        self.net = torch.nn.Sequential(
            nn.Linear(self.d_x, config.d_model),
            ResidualMLP(config.d_model, config.d_model),
            ResidualMLP(config.d_model, self.d_x)
        )

    def forward(self, x):
        return self.net(x)


class DilatedResBlock(nn.Module):
    """单个空洞残差卷积块，用于跨 patch 边界平滑。

    结构：pre-norm -> 空洞 Conv1d -> gated activation -> 1x1 proj -> 残差
    gate 让网络自主决定每个位置的修正幅度（峰边界处修正多，平坦区修正少）。
    """

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Conv1d(
            channels, channels * 2,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=pad,
        )
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv(x)
        feat, gate = x.chunk(2, dim=1)
        x = feat * torch.sigmoid(gate)
        return self.proj(x) + residual


class DecoderHead(nn.Module):
    '''
    数据流
    ------
        (B, N, D)
        -> patch_proj MLP: D -> 4D -> patch_len   非线性逐 token 投影，峰形草稿
        -> reshape (B, 1, N*patch_len)             平铺为连续光谱
        -> chan_in  Conv1d(1 -> C)                 升通道
        -> 2 × DilatedResBlock(dilation=1,2)       边界平滑，感受野 13 点
        -> chan_out Conv1d(C -> 1)                 压回单通道
        -> Softplus -> (B, spec_len)
    '''

    def __init__(self, config, d_x=512):
        super().__init__()
        self.d_x = d_x
        self.patch_len = config.patch_len
        self.d_model = config.d_model
        self.channels = config.channels

        # ── patch_proj：非线性 MLP，D -> 4D -> patch_len ─────────────────
        # 中间层用 4D=2048，给峰形细节足够的拟合能力
        # LayerNorm 稳定训练，GELU 引入非线性
        hidden = self.d_model * 4
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(self.d_x),
            nn.Linear(self.d_x, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.patch_len),
        )

        # ── 升通道，进入 refine 块 ────────────────────────────────────────
        self.chan_in = nn.Conv1d(1, self.channels, kernel_size=1)

        # ── 边界平滑：2 层空洞残差卷积，感受野 13 点（< 1/4 patch）────────
        # dilation=1: 感受野 5，修复紧邻边界的不连续
        # dilation=2: 感受野 13，覆盖边界两侧各约 6 个点
        self.refine = nn.Sequential(
            DilatedResBlock(self.channels, kernel_size=5, dilation=1),
            DilatedResBlock(self.channels, kernel_size=5, dilation=2),
        )

        # ── 压回单通道 ────────────────────────────────────────────────────
        self.chan_out = nn.Conv1d(self.channels, 1, kernel_size=1)
        
        # 物理约束：光谱强度通常为正
        self.activation = nn.Softplus() 

    def forward(self, x):
        """
        Args:
            x: (B, N, D)    N=50, D=512

        Returns:
            (B, spec_len)   spec_len = N * patch_len = 3200，值域 > 0
        """
        B, N, D = x.shape

        # MLP 逐 token 投影，生成光谱草稿
        x = self.patch_proj(x)                      # (B, N, patch_len)
        x = x.reshape(B, 1, N * self.patch_len)     # (B, 1, 3200)

        # 边界平滑
        x = self.chan_in(x)                          # (B, C, 3200)
        x = self.refine(x)                           # (B, C, 3200)
        x = self.chan_out(x)                         # (B, 1, 3200)

        return self.activation(x.squeeze(1))         # (B, 3200)