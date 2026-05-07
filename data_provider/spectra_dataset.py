import csv
import torch
from torch.utils.data import Dataset
from utils.misc import parse_spectrum
import logging

logger = logging.getLogger(__name__)

class SpectraDataset(Dataset):
    """从 CSV 文件读取多模态光谱数据。
 
    CSV header 必须包含：
        id, smiles, ir_spectrum, uv_spectrum, raman_spectrum
    （modality_names 对应其中的光谱列名）
 
    __getitem__ 返回 dict：
        {
            modality_name: torch.Tensor (L,) float32,  # 原始长度，由 Collater 对齐
            ...
            "smiles":              str,
            "present":             List[bool],  # 长度 M，True = 该模态有数据
        }
    """
 
    def __init__(self, path, modality_names):
        """
        Args:
            path:           CSV 文件路径
            modality_names: 光谱列名列表，如 ["ir_spectrum", "uv_spectrum", "raman_spectrum"]
        """
        super().__init__()
        self.csv_path      = path
        self.modality_names = modality_names
 
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            # 校验 header
            header = reader.fieldnames or []
            missing_cols = [m for m in modality_names if m not in header]
            if missing_cols:
                raise ValueError(
                    f"CSV {path} 缺少列: {missing_cols}，实际 header: {header}"
                )
            self.rows = list(reader)
 
        logger.info("SpectraDataset loaded: %s  (%d rows)", path, len(self.rows))
 
    def __len__(self):
        return len(self.rows)
 
    def __getitem__(self, idx):
        r   = self.rows[idx]
        out = {}
 
        present = []
        for modality in self.modality_names:
            spec = parse_spectrum(r.get(modality, ""))
            out[modality] = spec
            present.append(spec.numel() > 0)
 
        out["smiles"]  = r.get("smiles", "") or r.get("SMILES", "")
        out["present"] = present   # List[bool]，长度 M
 
        return out
 
 
# ─────────────────────────────────────────────────────────────────────────── #
# Collater
# ─────────────────────────────────────────────────────────────────────────── #
 
class SpectraCollater:
    """DataLoader 的 collate_fn。
 
    将一个 batch 的样本整理为：
        {
            modality_name: Tensor (B, spectrum_maxlen) float32,
            ...
            "smiles":      List[str],
            "known_mask":  Tensor (B, M) bool,
                           True  = 该样本该模态有真实光谱数据
                           False = 缺失，对应光谱已填零
        }
 
    known_mask 可直接传入 model.forward()，无需训练脚本二次处理。
    """
 
    def __init__(self, modality_names, spec_len):
        """
        Args:
            modality_names:  光谱列名列表，顺序必须与 model 中一致
            spectrum_maxlen: 固定光谱长度（= spec_len，与 Encoder 输入对齐）
        """
        self.modality_names  = modality_names
        self.spec_len = spec_len
 
    def __call__(self, batch):
        """
        Args:
            batch: List[dict]，每个 dict 来自 SpectraDataset.__getitem__
 
        Returns:
            dict，所有张量 shape 固定，可直接送入模型
        """
        out = {}
        raw_batch = {}
        enc_batch = {}
 
        # ── smiles ───────────────────────────────────────────────────────
        out["smiles"] = [b["smiles"] for b in batch]
 
        for m in self.modality_names:
            raw_spec = torch.stack([b[m] for b in batch], dim=0)
            # 原始浮点光谱用于重建监督（label）。
            raw_batch[m] = raw_spec
            # encoder 输入使用四舍五入后的版本。
            enc_batch[m] = torch.as_tensor(raw_spec).clamp(min=0, max=999).round().long()
        out["raw_batch"] = raw_batch
        out["enc_batch"] = enc_batch
        return out