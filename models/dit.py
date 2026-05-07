import math
import torch
import torch.nn as nn

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, frequency_embedding_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_dim)
        t_emb = self.mlp(t_freq)
        return t_emb

def get_pos_embedding(indices, emb_dim, max_len=2048):
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TanhGELU(nn.Module):
    """
    手动实现的 Tanh 近似版 GELU，兼容低版本 PyTorch。
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=0, bias=True, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        approx_gelu = lambda: TanhGELU()
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim, bias=True))

    def forward(self, x, c, padding_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        _x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(_x, _x, _x, key_padding_mask=padding_mask, need_weights=False)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    def __init__(
        self,
        d_x=512,            # 对应 Encoder 的 d_model
        d_model=768,        # DiT 内部的隐藏层维度
        num_layers=12,
        nhead=6,
        mlp_ratio=4.0,
        cond_dropout_prob=0.1, # 训练时随机丢弃条件以支持 CFG
        self_condition=False,
    ):
        super().__init__()
        self.d_x = d_x
        self.d_model = d_model
        self.self_condition = self_condition
        self.cond_dropout_prob = cond_dropout_prob

        # x_in_dim = noise_dim (d_x) + optional self_cond_dim (d_x)
        x_in_dim = d_x
        if self_condition:
            x_in_dim += d_x
            
        self.x_embedder = nn.Linear(x_in_dim, d_model, bias=True)
        self.t_embedder = TimestepEmbedder(d_model)

        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, nhead, mlp_ratio=mlp_ratio) for _ in range(num_layers)]
        )
        self.final_layer = FinalLayer(d_model, d_x)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, x_sc=None):
        """
        x: 加噪后的目标光谱特征 (B, N, d_x)
        t: 时间步 (B, 1) or (B,)
        context: 融合后的先验光谱特征 (B, N, d_x)
        """
        B, N, _ = x.shape
        # 无分类器引导的条件 Dropout (仅训练时)
        # if self.cond_dropout_prob > 0:
        #     drop_ids = torch.rand(x.shape[0], 1, 1, device=x.device) < self.cond_dropout_prob
        #     context = torch.where(drop_ids, torch.zeros_like(context), context)
        # input_x = torch.cat([x, context], dim=-1)
        token_index = torch.arange(N, device=x.device, dtype=torch.int64).unsqueeze(0).expand(B, -1)
        pos_emb = get_pos_embedding(token_index, self.d_model)
        
        input_x = x
        # 拼接目标特征和条件特征
        if self.self_condition:
            if x_sc is None:
                x_sc = torch.zeros_like(x)
            input_x = torch.cat([input_x, x_sc], dim=-1)        
            
        x_emb = self.x_embedder(input_x) + pos_emb
        t_emb = self.t_embedder(t)  # (B, d_model)
        c = t_emb 

        for block in self.blocks:
            x_emb = block(x_emb, c, None)

        out = self.final_layer(x_emb, c)
        return out

    def forward_with_cfg(self, x, t, context, mask, cfg_scale, x_sc=None):
        """用于推理阶段的 Classifier-Free Guidance forward"""
        # x 和 context 已经被沿 Batch 维度拼接了一倍 (2B, N, d)
        # 前半部分是条件预测，后半部分是无条件预测(context 为零)
        model_out = self.forward(x, t, context, mask, x_sc)  # (2B, N, d_x)
        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps