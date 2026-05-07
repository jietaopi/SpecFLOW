import torch

class FlowMatchingInterpolant:
    def __init__(
        self,
        min_t=0.01,
        corrupt=True,
        timesteps=100,
        self_condition=False,
        self_condition_prob=0.5,
        device="cpu",
    ):
        self.min_t = min_t
        self.corrupt = corrupt
        self.timesteps = timesteps
        self.self_condition = self_condition
        self.self_condition_prob = self_condition_prob
        self.device = device

    def _sample_t(self, batch_size):
        t = torch.rand(batch_size, device=self.device)
        return t * (1 - 2 * self.min_t) + self.min_t

    def _centered_gaussian(self, batch_size, num_tokens, emb_dim):
        noise = torch.randn(batch_size, num_tokens, emb_dim, device=self.device)
        return noise - torch.mean(noise, dim=-2, keepdim=True)

    def _corrupt_x(self, x_1, t, token_mask, diffuse_mask):
        x_0 = self._centered_gaussian(*x_1.shape)
        x_t = (1 - t[..., None]) * x_0 + t[..., None] * x_1
        x_t = x_t * diffuse_mask[..., None] + x_1 * (~diffuse_mask[..., None])
        return x_t * token_mask[..., None]

    def corrupt_batch(self, x_1):
        B, N, D = x_1.shape
        # 采样时间步，形状 (B,) -> (B, 1) 供 DiT 使用，(B, 1, 1) 供广播用
        t_1d = self._sample_t(B)                  # (B,)
        t_broad = t_1d[:, None, None]              # (B, 1, 1)

        if self.corrupt:
            x_0 = self._centered_gaussian(B, N, D)  # (B, N, D) noise
            x_t = (1 - t_broad) * x_0 + t_broad * x_1
        else:
            x_0 = None
            x_t = x_1
            
        return x_1, x_t, x_0, t_1d # t (B, )

    def _x_vector_field(self, t, x_1, x_t):
        return (x_1 - x_t) / (1 - t).clamp(min=1e-8)

    def _x_euler_step(self, d_t, t, x_1, x_t):
        x_vf = self._x_vector_field(t, x_1, x_t)
        return x_t + x_vf * d_t

    def sample_with_classifier_free_guidance(
        self,
        batch_size,
        num_tokens,
        emb_dim,
        model,
        context,  # 接收来自 WB_fuse 的特征
        cfg_scale=4.0,
        timesteps=None,
        x_0=None,
        x_1=None,
        token_mask=None,
    ):
        """带有无分类器引导的生成采样。

        Args:
            batch_size:   int
            num_tokens:   int，序列长度 N
            emb_dim:      int，特征维度 D
            model:        DiT 实例，需实现 forward_with_cfg
            context:      (B, N, D)，来自 WB Fuser 的融合特征
            cfg_scale:    float，引导强度
            timesteps:int，采样步数（默认使用 self.timesteps）
            x_0:          (B, N, D)，初始噪声（None 则自动采样）
            x_1:          (B, N, D)，仅当 self.corrupt=False 时使用
            token_mask:   (B, N)，有效 token 掩码

        Returns:
            dict:
                - "tokens_traj": list of (B, N, D)，每步的 x_t 轨迹（仅条件半）
                - "clean_traj":  list of (B, N, D)，每步预测的干净特征（仅条件半）
        """
        if x_0 is None:
            x_0 = self._centered_gaussian(batch_size, num_tokens, emb_dim)
        if token_mask is None:
            token_mask = torch.ones(batch_size, num_tokens, device=self.device).bool()

        # 扩展至 (2B, ...) 以支持 CFG：前半为条件，后半为无条件
        x_0 = torch.cat([x_0, x_0], dim=0)
        context_null = torch.zeros_like(context)
        context_cfg = torch.cat([context, context_null], dim=0)    # (2B, N, D)
        token_mask_cfg = torch.cat([token_mask, token_mask], dim=0)  # (2B, N)

        if timesteps is None:
            timesteps = self.timesteps
        ts = torch.linspace(self.min_t, 1.0, timesteps, device=self.device)

        tokens_traj = [x_0]
        clean_traj = []
        x_sc = None

        for i, t_1 in enumerate(ts[:-1]):
            t_2 = ts[i + 1]
            x_t_1 = tokens_traj[-1]                                          # (2B, N, D)
            x = x_t_1 if self.corrupt else torch.cat([x_1, x_1], dim=0)
            t = torch.ones((2 * batch_size, 1), device=self.device) * t_1    # (2B, 1)
            d_t = t_2 - t_1

            with torch.no_grad():
                pred_x_1 = model.forward_with_cfg(
                    x, t, context_cfg, token_mask_cfg, cfg_scale, x_sc
                )  # (2B, N, D)

            # 只保留条件半（前半 B 个）进入轨迹记录
            cond_pred = pred_x_1.chunk(2, dim=0)[0]   # (B, N, D)
            clean_traj.append(cond_pred)

            if self.self_condition:
                x_sc = pred_x_1   # 保持 (2B, N, D) 以对齐下一步输入

            x_t_2 = self._x_euler_step(d_t, t_1, pred_x_1, x_t_1)
            tokens_traj.append(x_t_2)

        # 最后一步
        t_final = ts[-1]
        x_t_final = tokens_traj[-1]
        x = x_t_final if self.corrupt else torch.cat([x_1, x_1], dim=0)
        t = torch.ones((2 * batch_size, 1), device=self.device) * t_final
        with torch.no_grad():
            pred_x_1 = model.forward_with_cfg(
                x, t, context_cfg, token_mask_cfg, cfg_scale, x_sc
            )

        cond_pred_final = pred_x_1.chunk(2, dim=0)[0]
        clean_traj.append(cond_pred_final)
        tokens_traj.append(cond_pred_final)

        return {"tokens_traj": tokens_traj, "clean_traj": clean_traj}

    def sample_unconditional(
        self,
        batch_size,
        num_tokens,
        emb_dim,
        model,
        timesteps=None,
        x_0=None,
        x_1=None,
    ):
        """无条件采样：不输入 context，直接从噪声积分到预测样本。"""
        if x_0 is None:
            x_0 = self._centered_gaussian(batch_size, num_tokens, emb_dim)

        if timesteps is None:
            timesteps = self.timesteps
        ts = torch.linspace(self.min_t, 1.0, timesteps, device=self.device)

        tokens_traj = [x_0]
        clean_traj = []
        x_sc = None

        for i, t_1 in enumerate(ts[:-1]):
            t_2 = ts[i + 1]
            x_t_1 = tokens_traj[-1]
            x = x_t_1 if self.corrupt else x_1
            t = torch.ones((batch_size, 1), device=self.device) * t_1
            d_t = t_2 - t_1

            with torch.no_grad():
                pred_x_1 = model(x=x, t=t, x_sc=x_sc)

            clean_traj.append(pred_x_1)

            if self.self_condition:
                x_sc = pred_x_1

            x_t_2 = self._x_euler_step(d_t, t_1, pred_x_1, x_t_1)
            tokens_traj.append(x_t_2)

        t_final = ts[-1]
        x_t_final = tokens_traj[-1]
        x = x_t_final if self.corrupt else x_1
        t = torch.ones((batch_size, 1), device=self.device) * t_final

        with torch.no_grad():
            pred_x_1 = model(x=x, t=t, x_sc=x_sc)

        clean_traj.append(pred_x_1)
        tokens_traj.append(pred_x_1)

        return {"tokens_traj": tokens_traj, "clean_traj": clean_traj}