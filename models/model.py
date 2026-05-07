import torch
import torch.nn as nn
import torch.nn.functional as F

from models.latent_diffusion import LatentDiffusion
from models.encoder import Encoder
from models.wb_fuse import WassersteinBarycenterFuser
from models.decoder import Decoder, DecoderHead
from utils.common import kl_diag_gaussian, curvature_loss, peak_loss, sid_loss
from scipy.signal import find_peaks

class ADMCModel(nn.Module):
    def __init__(self, config, modality_names, spec_len, device):
        super().__init__()
        self.device = device
        self.config = config
        self.modality_names = modality_names
        self.spec_len = spec_len
        self.d_model = config.d_model

        # register_buffer 默认 requires_grad=False
        self.register_buffer('prior_mu', torch.tensor(config.prior_mu))
        self.register_buffer('prior_logvar', torch.tensor(config.prior_logvar))

        # Per-modality encoder configs.        
        self.encoders = nn.ModuleDict(
            {m: Encoder(config.encoder) for m in modality_names}
        )

        self.diffusion = LatentDiffusion(config.diffusion, d_x=config.d_model)
        
        self.wb_fuser = WassersteinBarycenterFuser()

        # z_hat (fused) -> per-modality feature tokens (M decoders)
        self.decoders = nn.ModuleDict(
            {m: Decoder(config.decoder, d_x=config.d_model) for m in modality_names}
        )

        # per-modality token -> reconstruct masked input spectra (M heads)
        self.decoder_heads = nn.ModuleDict(
            {m: DecoderHead(config.decoder, d_x=config.d_model) for m in modality_names}
        )

        # per-modality token -> peak logits over spectrum positions (M heads)
        self.peak_heads = nn.ModuleDict(
            {m: nn.Linear(config.d_model, spec_len) for m in modality_names}
        )

    def mask_inputs(self, batch, known_mask):
        """对缺失模态的原始输入清零。"""
        return {
            m: batch[m] * known_mask[:, i].to(batch[m].dtype).unsqueeze(-1)
            for i, m in enumerate(self.modality_names)
        }

    def encode(
        self,
        inputs,
        known_mask,
    ):
        """编码所有模态，缺失模态以先验（零均值、零 log 方差）填充。

        Args:
            inputs:     dict，modality -> (B, L)
            known_mask: (B, M) bool，True = 已知

        Returns:
            mus:     (B, M, N, D)
            logvars: (B, M, N, D)
        """
        B = known_mask.shape[0]
        mus_list, logvars_list = [], []

        for m_idx, m in enumerate(self.modality_names):
            present_mask = known_mask[:, m_idx].unsqueeze(-1).unsqueeze(-1).float()   # (B, 1, 1)
            # avoid encoding missing inputs and gradient computation on them
            mu_p, logvar_p = self.encoders[m](inputs[m], return_states=True)
            mu = mu_p * present_mask + self.prior_mu * (1.0 - present_mask)
            logvar = logvar_p * present_mask + self.prior_logvar * (1.0 - present_mask)

            mus_list.append(mu)
            logvars_list.append(logvar)

        return torch.stack(mus_list, dim=1), torch.stack(logvars_list, dim=1)

    def fuse(
        self,
        mus,      # (B, M, N, D)
        logvars,  # (B, M, N, D)
        mask,     # (B, M) bool
    ):
        """WB 融合 + 重参数化采样。

        Returns:
            z_fuse:      (B, N, D)
            fused_mu:    (B, N, D)
            fused_logvar:(B, N, D)
        """
        fused_mu, fused_logvar = self.wb_fuser(mus, logvars, mask)
        z_fuse = self.wb_fuser.sample(fused_mu, fused_logvar)
        return z_fuse, fused_mu, fused_logvar

    def decode_spectra(self, z_fuse):
        """z_fuse (B, N, D) -> seq_tokens & pred_spectra（各模态）。

        Returns:
            seq_tokens:   dict，modality -> (B, N, D)
            pred_spectra: dict，modality -> (B, L)
        """
        seq_tokens   = {m: self.decoders[m](z_fuse)         for m in self.modality_names}
        pred_spectra = {m: self.decoder_heads[m](seq_tokens[m]) for m in self.modality_names}
        return seq_tokens, pred_spectra

    def decoder_losses(
        self,
        seq_tokens,
        pred_spectra,
        batch,
        known_mask,
    ):
        """计算 spec / peak / curvature loss，只对缺失模态计算。

        Returns:
            spec_loss:  scalar
            peak_loss:  scalar
            curv_loss:  scalar
            cos_sims:   (M,)
        """
        B, M = known_mask.shape
        loss_w_cfg = self.config.loss_weight
        uv_spec_weight = float(getattr(loss_w_cfg, "uv_spec_weight", 1.0))
        other_spec_weight = float(getattr(loss_w_cfg, "other_spec_weight", 1.0))
        uv_curv_weight = float(getattr(loss_w_cfg, "uv_curv_weight", 100.0))
        other_curv_weight = float(getattr(loss_w_cfg, "other_curv_weight", 1.0))

        spec_losses, curv_losses = [], []
        spec_losses_uv, spec_losses_other = [], []
        curv_losses_uv, curv_losses_other = [], []
        cos_sims = torch.zeros(M, device=self.device, dtype=torch.float32)

        for m_idx, m in enumerate(self.modality_names):
            target       = batch[m].to(torch.float32)       # (B, L)
            pred         = pred_spectra[m]                   # (B, L)
            missing_flag = (~known_mask[:, m_idx]).float()          # (B,)
            missing_num  = missing_flag.sum()

            if missing_num == 0:
                continue

            # cosine similarity
            cos_sims[m_idx] = (
                F.cosine_similarity(pred + 1e-8, target + 1e-8, dim=-1) * missing_flag
            ).sum() / missing_num

            # 光谱重建
            # 峰位：序列均值池化 -> 线性分类
            # token_global = seq_tokens[m].mean(dim=1)  # (B, D)
            with torch.no_grad():
                peak_labels = torch.zeros_like(target)
                for b in range(B):
                    target_np = target[b].detach().cpu().numpy()
                    peak_height = 0.003 * float(target[b].max().item())
                    peaks, _ = find_peaks(target_np, height=peak_height, distance=5)
                    if len(peaks):
                        peak_labels[b, peaks] = 1.0

            curv_loss = curvature_loss(pred, target, peak_labels, m)

            if m == 'uv_spectrum':
                spec_loss = peak_loss(pred, target, peak_labels, m)
                curv_item = (curv_loss * missing_flag).sum() / missing_num
                spec_item = (spec_loss * missing_flag).sum() / missing_num
                curv_losses.append(curv_item * uv_curv_weight)
                spec_losses.append(spec_item * uv_spec_weight)
                curv_losses_uv.append(curv_item)
                spec_losses_uv.append(spec_item)
            else:
                spec_loss = peak_loss(pred, target, peak_labels, m)
                curv_item = (curv_loss * missing_flag).sum() / missing_num
                spec_item = (spec_loss * missing_flag).sum() / missing_num
                curv_losses.append(curv_item * other_curv_weight)
                spec_losses.append(spec_item * other_spec_weight)
                curv_losses_other.append(curv_item)
                spec_losses_other.append(spec_item)

        zero = torch.tensor(0.0, device=self.device)
        spec_total = torch.stack(spec_losses).mean() if spec_losses else zero
        curv_total = torch.stack(curv_losses).mean() if curv_losses else zero
        raw_spec_uv = torch.stack(spec_losses_uv).mean() if spec_losses_uv else zero
        raw_spec_other = torch.stack(spec_losses_other).mean() if spec_losses_other else zero
        raw_curv_uv = torch.stack(curv_losses_uv).mean() if curv_losses_uv else zero
        raw_curv_other = torch.stack(curv_losses_other).mean() if curv_losses_other else zero
        breakdown = {
            "spec_loss_uv": raw_spec_uv * uv_spec_weight,
            "spec_loss_other": raw_spec_other * other_spec_weight,
            "spec_loss_uv_raw": raw_spec_uv,
            "spec_loss_other_raw": raw_spec_other,
            "curv_loss_uv": raw_curv_uv * uv_curv_weight,
            "curv_loss_other": raw_curv_other * other_curv_weight,
            "curv_loss_uv_raw": raw_curv_uv,
            "curv_loss_other_raw": raw_curv_other,
        }
        return spec_total, curv_total, cos_sims, breakdown

    def forward(
        self,
        batch,
        known_mask,
        *,
        use_diffusion=False,
        use_decoder=True,
        kl_weight=None,
    ):
        """
        Args:
            batch:         dict，modality -> (B, L)
            known_mask:    (B, M) bool，True = 该模态已知
            use_diffusion: 是否走 LatentDiffusion 分支（Stage 2）
            use_decoder:   是否走 Decoder / DecoderHead 分支（Stage 1 & 3）
            kl_weight:     KL 正则化权重，None 则从 config 读取
        """
        assert use_diffusion or use_decoder, (
            "至少需要开启 use_diffusion 或 use_decoder 中的一个"
        )
        # Batch dict may contain non-tensor fields (e.g. missing_modalities as list).
        B = known_mask.shape[0] 
        M = known_mask.shape[1]

        w = self.config.loss_weight
        if kl_weight is None:
            kl_weight = w.kl_weight

        out = {}
        total_loss = torch.tensor(0.0, device=self.device)
        enc_batch = batch["enc_batch"]
        raw_batch = batch["raw_batch"]
        # ── 1. 编码已知模态 ───────────────────────────────────────────────
        masked_inputs = self.mask_inputs(enc_batch, known_mask)
        mus, logvars = self.encode(masked_inputs, known_mask)
        # mus / logvars: (B, M, N, D)

        # ── 2. WB 融合 -> z_fuse ──────────────────────────────────────────
        z_fuse, fused_mu, fused_logvar = self.fuse(mus, logvars, known_mask)
        # z_fuse: (B, N, D)

        # ── 3. Diffusion 分支（Stage 2）────────────────────────────────────
        if use_diffusion:
            # 构造全模态 teacher label（不需要梯度）
            with torch.no_grad():
                all_present = torch.ones(B, M, dtype=torch.bool, device=self.device)
                mus_full, logvars_full = self.encode(enc_batch, all_present)
                target_latent, target_mu, target_logvar = self.fuse(mus_full, logvars_full, all_present)

            diff_out = self.diffusion(
                target_latent=target_latent,      # (B, N, D)  label
                x_1=z_fuse,                 # (B, N, D)  cond token
            )
            diff_loss = diff_out["loss"]
            total_loss = total_loss + w.eps_weight * diff_loss
            z_fuse = diff_out["pred_x"]  # (B, N, D)  diffusion 预测的 latent

            out["diff_loss"] = diff_loss
            out["diff_out"]  = diff_out

        # ── 4. Decoder 分支（Stage 1 & 3）─────────────────────────────────
        if use_decoder:
            seq_tokens, pred_spectra = self.decode_spectra(z_fuse)

            spec_loss, curv_loss, cos_sims, loss_breakdown = self.decoder_losses(
                seq_tokens, pred_spectra, raw_batch, known_mask
            )

            total_loss = (
                total_loss
                + w.spectrum_missing_weight * spec_loss
                + w.curvature_weight       * curv_loss
            )
            out["spec_loss"] = spec_loss
            out["curv_loss"] = curv_loss
            out.update(loss_breakdown)
            out["cos_sims"]  = cos_sims

            # KL 正则：fused + 各 encoder 输出，等权平均后再乘 kl_weight。
            kl_terms = [
                kl_diag_gaussian(
                    fused_mu,       # (B, N, D)
                    fused_logvar,   # (B, N, D)
                    prior_mu=self.prior_mu,
                    prior_logvar=self.prior_logvar,
                )
            ]
            for m_idx in range(M):
                kl_terms.append(
                    kl_diag_gaussian(
                        mus[:, m_idx],      # (B, N, D)
                        logvars[:, m_idx],  # (B, N, D)
                        prior_mu=self.prior_mu,
                        prior_logvar=self.prior_logvar,
                    )
                )
            kl_loss = torch.stack(kl_terms).mean()
            total_loss = total_loss + kl_weight * kl_loss
            out["kl_loss"]   = kl_loss

        out["loss"] = total_loss
        return out


    @torch.no_grad()
    def predict_spectra(
        self,
        batch,
        known_mask,
        *,
        use_diffusion=False,
        cfg_scale=4.0,
        return_debug=False,
    ):
        """补全缺失光谱（推理专用，不计算梯度）。

        Args:
            use_diffusion: True -> Flow Matching 采样；False -> 直接用 WB 融合均值

        Returns:
            spec_pred:         (B, M, L)
            spec_missing_loss: scalar
            cos_sims:          (B, M)
            [debug]:           dict（return_debug=True 时）
        """
        B, M = known_mask.shape

        enc_batch = batch["enc_batch"]
        raw_batch = batch["raw_batch"]
        masked_inputs = self.mask_inputs(enc_batch, known_mask)
        mus, logvars  = self.encode(masked_inputs, known_mask)
        z_fuse, fused_mu, fused_logvar = self.fuse(mus, logvars, known_mask)
        if use_diffusion:
            z_pred = self.diffusion.sample(
                device=self.device,
                context_fused=z_fuse,
                cfg_scale=cfg_scale,
            )  # (B, N, D)
            z = z_pred
        else:
            z = fused_mu
        _, pred_spectra = self.decode_spectra(z)

        spectra_list, spec_losses = [], []
        cos_sims = torch.zeros(B, M, device=self.device, dtype=torch.float32)

        for m_idx, m in enumerate(self.modality_names):
            target = raw_batch[m].to(torch.float32)
            pred   = pred_spectra[m]           # (B, L)
            spectra_list.append(pred)

            missing_flag = (~known_mask[:, m_idx]).float()
            missing_num  = missing_flag.sum().clamp(min=1.0)
            if missing_flag.sum() == 0:
                continue

            cos_sims[:, m_idx] = (
                F.cosine_similarity(pred + 1e-8, target + 1e-8, dim=-1) * missing_flag
            )
            spec_losses.append(
                (F.smooth_l1_loss(pred, target, reduction="none") * missing_flag.unsqueeze(-1))
                .sum() / (missing_num * self.spec_len)
            )

        spec_pred         = torch.stack(spectra_list, dim=1)   # (B, M, L)
        spec_missing_loss = (
            torch.stack(spec_losses).sum() if spec_losses
            else torch.tensor(0.0, device=self.device)
        )

        if return_debug:
            return spec_pred, spec_missing_loss, cos_sims, {
                "mus": mus, "logvars": logvars, "z_fuse": z_fuse, "fused_mu": fused_mu, "fused_logvar": fused_logvar
            }
        return spec_pred, spec_missing_loss, cos_sims

    # ═══════════════════════════════════════════════════════════════════════ #
    # 阶段切换：set_stage() 负责所有 requires_grad，forward() 不做冻结
    # ═══════════════════════════════════════════════════════════════════════ #

    def set_stage(self, stage):
        """一键切换训练阶段，管理所有模块的 requires_grad。

        Stage 1  use_diffusion=False, use_decoder=True
                 可训练：Encoder, WB Fuser, Decoder, DecoderHead, peak_heads
                 冻结  ：LatentDiffusion

        Stage 2  use_diffusion=True,  use_decoder=False
                 可训练：LatentDiffusion
                 冻结  ：Encoder（默认；如需联合训练可之后手动 unfreeze_encoders()），
                         Decoder, DecoderHead, peak_heads

        Stage 3  use_diffusion=False, use_decoder=True
                 可训练：Decoder, DecoderHead, peak_heads
                 冻结  ：Encoder, WB Fuser, LatentDiffusion
        """
        if stage not in (1, 2, 3):
            raise ValueError(f"stage 必须是 1 / 2 / 3，当前值：{stage}")

        enc_grad  = (stage in (1, 2, 3))
        diff_grad = (stage == 2)
        dec_grad  = (stage in (1, 2, 3))

        for enc in self.encoders.values():
            for p in enc.parameters():
                p.requires_grad_(enc_grad)

        # WB Fuser 本身无可学习参数，但保持与 Encoder 一致的语义
        for p in self.wb_fuser.parameters():
            p.requires_grad_(enc_grad)

        for p in self.diffusion.parameters():
            p.requires_grad_(diff_grad)

        for m in self.modality_names:
            for p in self.decoders[m].parameters():
                p.requires_grad_(dec_grad)
            for p in self.decoder_heads[m].parameters():
                p.requires_grad_(dec_grad)
            for p in self.peak_heads[m].parameters():
                p.requires_grad_(dec_grad)

    def unfreeze_encoders(self):
        """Stage 2 联合训练时手动解冻 Encoder。"""
        for enc in self.encoders.values():
            for p in enc.parameters():
                p.requires_grad_(True)

