import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """通道注意力机制，让模型自动学习哪些光谱波段或特征更重要"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class MultiScaleInception(nn.Module):
    """多尺度卷积层：同时捕捉光谱中的窄峰(3)和宽峰(7, 11)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_out = out_channels // 4
        self.branch1 = nn.Conv1d(in_channels, branch_out, kernel_size=3, padding=1)
        self.branch2 = nn.Conv1d(in_channels, branch_out, kernel_size=7, padding=3)
        self.branch3 = nn.Conv1d(in_channels, branch_out, kernel_size=11, padding=5)
        self.branch4 = nn.Conv1d(in_channels, branch_out, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        res = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return self.gelu(self.bn(res))

class SpectraFGPredictor(nn.Module):
    def __init__(self, spectrum_maxlen, num_labels=17, hidden_dim=256):
        super().__init__()
        
        # 针对每种模态的编码器
        def make_encoder():
            return nn.Sequential(
                MultiScaleInception(1, 32),
                nn.MaxPool1d(2),
                MultiScaleInception(32, 64),
                SEBlock(64),
                nn.AdaptiveAvgPool1d(16), # 压缩长度
                nn.Flatten()
            )

        self.ir_enc = make_encoder()
        self.uv_enc = make_encoder()
        self.raman_enc = make_encoder()

        # 融合后的全连接层
        combined_dim = 64 * 16 * 3 # 3个模态
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x):
        # x shape: (B, 3, L)
        f_ir = self.ir_enc(x[:, 0:1, :])
        f_uv = self.uv_enc(x[:, 1:2, :])
        f_raman = self.raman_enc(x[:, 2:3, :])
        
        merged = torch.cat([f_ir, f_uv, f_raman], dim=1)
        logits = self.fusion(merged)
        return logits