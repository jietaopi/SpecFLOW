import torch
import logging
import torch.nn.functional as F  

def sample_known_mask(*, batch_size, num_modalities, drop_prob, device):
    """Sample a (B,M) boolean mask where True means modality is KNOWN.

    Ensures each sample has at least 1 known modality.
    Samples may have zero missing modalities (i.e. all-known inputs are allowed).

    Returns: (B,M) bool tensor, True means modality is KNOWN.
    """
    if num_modalities <= 1:
        raise ValueError("num_modalities must be >= 2")
    prob = drop_prob
    if not (0.0 <= prob <= 1.0):
        raise ValueError("drop_prob must be in [0,1]")

    # Start with independent drops.
    known_mask = (torch.rand((batch_size, num_modalities), device=device) > prob)
    # Fix rows with all missing: force exactly one known.
    all_missing = (~known_mask).all(dim=1)
    if bool(all_missing.any().item()):
        idx = torch.nonzero(all_missing, as_tuple=False).squeeze(1)
        force_known = torch.randint(0, num_modalities, (idx.numel(),), device=device)
        known_mask[idx, force_known] = True

    return known_mask


def sample_known_mask_test(
    *,
    batch_size,
    num_modalities,
    device,
    missing_modalities=None,
):
    """Sample a (B,M) boolean mask with controlled missing-modality patterns.

    True means modality is KNOWN; False means modality is MISSING.

    Args:
        batch_size: Number of samples B.
        num_modalities: Number of modalities M.
        device: Torch device.
        missing_modalities: Iterable of modality indices to force as
            missing for every sample in the batch. If provided, random missing
            logic is skipped.
    """
    if num_modalities <= 1:
        raise ValueError("num_modalities must be >= 2")
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")

    if missing_modalities is not None:
        missing_modalities = list(missing_modalities)
        if len(missing_modalities) == 0:
            raise ValueError("missing_modalities cannot be empty")
        if len(set(missing_modalities)) != len(missing_modalities):
            raise ValueError("missing_modalities contains duplicated indices")
        if min(missing_modalities) < 0 or max(missing_modalities) >= num_modalities:
            raise ValueError("missing_modalities has out-of-range index")
        # if len(missing_modalities) >= num_modalities:
        #     raise ValueError("at least one modality must remain known")

        known_mask = torch.ones((batch_size, num_modalities), device=device, dtype=torch.bool)
        drop_idx = torch.tensor(missing_modalities, device=device, dtype=torch.long)
        known_mask[:, drop_idx] = False
        return known_mask


def compute_curvature_sigma(train_loader, modality_names, device, max_batches=100):
    logger = logging.getLogger(__name__)
    all_d2 = {m: [] for m in modality_names}
    for i, batch in enumerate(train_loader):
        raw_batch = batch["raw_batch"]
        if i >= max_batches:
            break
        for m in modality_names:
            spec = raw_batch[m].to(device).float()
            d2 = spec[:, 2:] - 2 * spec[:, 1:-1] + spec[:, :-2]
            all_d2[m].append(d2.reshape(-1))
    
    sigmas = {}
    for m in modality_names:
        all_vals = torch.cat(all_d2[m])
        sigmas[m] = all_vals.std().item()
        logger.info(f"curvature sigma [{m}]: {sigmas[m]:.6f}")
    sigmas["uv_spectrum"] = sigmas['uv_spectrum'] * 0.5

    return sigmas


def curvature_loss(pred, target, peak_labels, modality):
    """
    peak_labels: (B, L) 二值，1 表示峰位置
    """
    d2_pred = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]    # (B, L-2)
    d2_true = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]

    peak_region = peak_labels[:, 1:-1]  # 对齐到 L-2，(B, L-2)
    background = (1.0 - peak_region)    # 背景区域

    if modality == 'uv_spectrum':
        sigma_uv = torch.quantile(d2_true.abs(), 0.9, keepdim=True)
        # UV：全局平滑，峰区域和背景都惩罚高曲率
        violation = (d2_pred.abs() > sigma_uv)
        diff = pred[:, 1:] - pred[:, :-1]   # 一阶差分 (B, L-1)
        diff_loss = 0.5 * diff.abs().mean()

        return (d2_pred ** 2 * violation).sum(dim=-1) / (violation.sum(dim=-1) + 1e-8) + diff_loss # 只平均违反曲率约束的点

    else:
        # IR/Raman：只惩罚背景区域的高曲率，峰区域不惩罚
        # 背景应该平滑，峰附近允许突变
        bg_d2 = d2_pred * background           # 只看背景
        sigma = torch.quantile((d2_true * background).abs(), 0.6, keepdim=True)
        violation = (bg_d2.abs() > sigma)

        return (bg_d2 ** 2 * violation).sum(dim=-1) / (violation.sum(dim=-1) + 1e-8)  # 只平均违反曲率约束的点


def sid_loss(pred, target, eps=1e-8):
    """
    Spectral Information Divergence loss
    pred, target: (B, L) 光谱强度，非负
    
    注意：SID 要求输入严格为正，你的 decoder 用了 Softplus 激活，
    输出已经是正值，但 target 可能有 0，需要加 eps
    """
    pred_clamp = pred.clamp(min=eps)
    target_clamp = target.clamp(min=eps)
    # 归一化为概率分布
    pred_sum = pred_clamp.sum(dim=-1, keepdim=True).clamp(min=eps)
    target_sum = target_clamp.sum(dim=-1, keepdim=True).clamp(min=eps)
    
    p = pred_clamp / pred_sum + eps      # (B, L)
    q = target_clamp / target_sum + eps  # (B, L)
    
    # 对称 KL
    kl_pq = (p * torch.log(p / q)).sum(dim=-1)   # (B,)
    kl_qp = (q * torch.log(q / p)).sum(dim=-1)   # (B,)
    
    return kl_pq + kl_qp   # (B,)


def peak_loss(pred, target, peak_labels, modality):

    # 背景区域用 smooth_l1
    smooth_l1 = F.smooth_l1_loss(pred, target, reduction='none')
    if modality == 'uv_spectrum':
        # UV：峰区域和背景都用 smooth_l1
        return smooth_l1.mean(dim=-1)  # (B,)
    else:
        # 峰区域用 MSE，对大误差惩罚更重
        mse = F.mse_loss(pred, target, reduction='none')
        
        peak_mask = peak_labels.float()
        loss = mse * peak_mask + smooth_l1 * (1 - peak_mask)
        return loss.mean(dim=-1)  # (B,)


def kl_diag_gaussian(
    mu,
    logvar,
    *,
    prior_mu,
    prior_logvar,
):
    """KL( N(mu, exp(logvar)) || N(prior_mu, exp(prior_logvar)) ) averaged over batch.

    Supports latent tensors shaped (B, D) or (B, N, D), and more generally
    (B, ..., D). KL is summed over all non-batch dimensions, then averaged over
    batch.
    """

    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have the same shape")
    if mu.ndim < 2:
        raise ValueError("mu/logvar must have shape (B, D) or (B, ..., D)")

    var = torch.exp(logvar)
    prior_var = torch.exp(prior_logvar)
    # 0.5 * sum( log(prior_var/var) + (var + (mu-prior_mu)^2)/prior_var - 1 )
    kl = 0.5 * (
        (prior_logvar - logvar)
        + (var + (mu - prior_mu) ** 2) / prior_var
        - 1.0
    )
    reduce_dims = tuple(range(1, kl.ndim))
    return kl.sum(dim=reduce_dims).mean()