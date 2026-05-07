import os
import csv
import random
import time
import torch
import logging
import ast
from glob import glob
import numpy as np

def get_logger(name, log_dir=None, log_fn='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, log_fn))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir


def seed_all(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_checkpoint_path(folder, iteration=None):
    if iteration is not None:
        return os.path.join(folder, '%d.pt' % iteration), iteration

    ckpt_paths = glob(os.path.join(folder, '*.pt'))
    if len(ckpt_paths) == 0:
        raise FileNotFoundError(f"No checkpoint found in: {folder}")

    latest_path = os.path.join(folder, 'latest.pt')
    if os.path.exists(latest_path):
        # Prefer latest.pt when present; recover iteration from checkpoint payload.
        try:
            ckpt = torch.load(latest_path, map_location='cpu')
            latest_iter = int(ckpt.get('iteration'))
            return latest_path, latest_iter
        except Exception:
            pass

    all_iters = []
    for path in ckpt_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.isdigit():
            all_iters.append(int(stem))

    if len(all_iters) > 0:
        all_iters.sort()
        return os.path.join(folder, '%d.pt' % all_iters[-1]), all_iters[-1]

    if os.path.exists(latest_path):
        return latest_path, 1

    raise ValueError(f"No numeric checkpoint file found in: {folder}")


def move_to_device(batch, device):
    def move(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, dict):
            return {k: move(v) for k, v in x.items()}
        if isinstance(x, list):
            return [move(v) for v in x]
        if isinstance(x, tuple):
            return tuple(move(v) for v in x)
        return x

    return move(batch)


def next_batch(dataloader_iter, dataloader, device):
    try:
        batch = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)
    # move tensors to device
    batch = move_to_device(batch, device)
    return dataloader_iter, batch


def parse_label_list(text: str) -> torch.Tensor:
    if text is None:
        return torch.empty(0, dtype=torch.float32)
    text = str(text).strip()
    if text == "":
        return torch.empty(0, dtype=torch.float32)

    try:
        arr = ast.literal_eval(text)
        if isinstance(arr, (list, tuple)):
            return torch.tensor(arr, dtype=torch.float32)
    except Exception:
        pass

    cleaned = text.strip("[] ")
    parts = [p for p in cleaned.replace("\n", " ").replace("\t", " ").split(",") if p.strip()]
    if len(parts) <= 1:
        parts = [p for p in cleaned.split() if p.strip()]
    vals = [float(p) for p in parts]
    return torch.tensor(vals, dtype=torch.float32)


def parse_missing_modalities(text):
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts: raise ValueError("--missing_modalities cannot be empty")
    try: return [int(p) for p in parts]
    except ValueError as exc: raise ValueError("--missing_modalities must be comma-separated integers") from exc


def parse_spectrum(raw):
    """将 CSV 单元格中的 list 字符串解析为 float32 Tensor。
 
    支持的格式：
        "[0.1, 0.2, 0.3]"      Python list
        "0.1 0.2 0.3"          空格分隔
        ""                     空字符串 -> 返回空 Tensor
 
    Returns:
        torch.Tensor，shape (L,)，dtype float32；
        解析失败或为空时返回 shape (0,) 的空 Tensor。
    """
    logger = logging.getLogger(__name__)
    raw = raw.strip() if raw else ""
    if not raw:
        return torch.zeros(0, dtype=torch.float32)
 
    # 尝试 Python literal（最常见：列表格式）
    try:
        vals = ast.literal_eval(raw)
        if isinstance(vals, (list, tuple)):
            return torch.tensor(vals, dtype=torch.float32)
    except (ValueError, SyntaxError):
        pass
 
    # 回退：空格 / 逗号分隔的纯数字串
    try:
        vals = [float(x) for x in raw.replace(",", " ").split() if x]
        if vals:
            return torch.tensor(vals, dtype=torch.float32)
    except ValueError:
        pass
 
    logger.warning("无法解析光谱字段，视为缺失: %r", raw[:80])
    return torch.zeros(0, dtype=torch.float32)


def append_spec_pred_to_csv(out_path, *, smiles, spec_pred, modality_names, cos_sims):
    """Append a batch of predicted spectra to a CSV.

    CSV columns:
      - smiles
      - one column per modality name (e.g. ir_spectrum/uv_spectrum/raman_spectrum)

    Each modality cell stores a python-list string (length L) so it can be parsed
    later by utils.misc.parse_spectrum_list.
    """

    if spec_pred.ndim != 3:
        raise ValueError(f"spec_pred must be (B,M,L), got {spec_pred.shape}")

    B, M, _ = spec_pred.shape
    if len(smiles) != B:
        raise ValueError(f"len(smiles)={len(smiles)} must match B={B}")
    if len(modality_names) != M:
        raise ValueError(f"len(modality_names)={len(modality_names)} must match M={M}")

    os.makedirs(os.path.dirname(os.path.realpath(out_path)), exist_ok=True)
    file_exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    fieldnames = ["smiles"] + list(modality_names) + ["cos_sims"]

    spec_pred = spec_pred.detach().to("cpu", non_blocking=True)
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i in range(B):
            row = {"smiles": smiles[i]}
            for m_idx, modality in enumerate(modality_names):
                row[modality] = repr(spec_pred[i, m_idx].tolist())
            row["cos_sims"] = repr(cos_sims[i].tolist())
            writer.writerow(row)


def compute_multilabel_metrics(probs, labels, threshold=0.5):
    """
    probs: (N, K) Tensor, 模型输出的概率
    labels: (N, K) Tensor, 真实的 0/1 标签
    """
    # 1. 转换为二进制预测
    preds = (probs > threshold).float()
    labels = labels.float()

    # ---------------------------------------------------------
    # 准确率计算 (Accuracy)
    # ---------------------------------------------------------
    # 每一位预测正确的矩阵 (N, K)
    correct_matrix = (preds == labels).float()
    
    # K 个 label 各自的准确率 (Shape: [K])
    per_label_acc = correct_matrix.mean(dim=0)
    
    # 总的平均准确率 (所有元素的平均正确率)
    total_mean_acc = correct_matrix.mean()

    # ---------------------------------------------------------
    # F1-Score 计算
    # ---------------------------------------------------------
    # 计算 TP, FP, FN (Shape: [K])
    tp = torch.sum(preds * labels, dim=0)
    fp = torch.sum(preds * (1 - labels), dim=0)
    fn = torch.sum((1 - preds) * labels, dim=0)

    # 计算每个类别的 Precision 和 Recall
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    # 计算每个类位的 F1
    per_label_f1 = 2 * precision * recall / (precision + recall + 1e-10)

    # Macro-F1: 简单算术平均
    f1_macro = torch.mean(per_label_f1)
    
    # Micro-F1: 先汇总 TP/FP/FN 再计算
    tp_sum, fp_sum, fn_sum = tp.sum(), fp.sum(), fn.sum()
    micro_p = tp_sum / (tp_sum + fp_sum + 1e-10)
    micro_r = tp_sum / (tp_sum + fn_sum + 1e-10)
    f1_micro = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-10)

    return {
        "per_label_acc": per_label_acc,  # 每个类别的准确率
        "total_mean_acc": total_mean_acc, # 总体平均准确率
        "f1_macro": f1_macro,           # 宏观 F1
        "f1_micro": f1_micro,           # 微观 F1
        "per_label_f1": per_label_f1    # 每个类别的 F1
    }


def find_best_thresholds(all_probs, all_labels):
    """
    all_probs: (N, 17) Tensor
    all_labels: (N, 17) Tensor
    """
    best_thresholds = []
    
    # 遍历 17 个官能团
    for i in range(all_labels.shape[1]):
        probs = all_probs[:, i].numpy()
        labels = all_labels[:, i].numpy()
        
        # 如果该类在测试集中没有正样本，直接给个默认值 0.5
        if labels.sum() == 0:
            best_thresholds.append(0.5)
            continue
            
        best_f1 = 0
        best_t = 0.5
        
        # 在 0.01 到 0.99 之间搜索最佳阈值
        for t in np.arange(0.01, 1.0, 0.01):
            preds = (probs > t).astype(int)
            # 计算 F1，注意处理除零
            tp = (preds * labels).sum()
            fp = (preds * (1 - labels)).sum()
            fn = ((1 - preds) * labels).sum()
            
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-10)
            
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        
        best_thresholds.append(best_t)
        
    return torch.tensor(best_thresholds)