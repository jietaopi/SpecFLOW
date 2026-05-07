import torch
import csv
from torch.utils.data import Dataset
from utils.misc import parse_spectrum_list, parse_label_list

class SpectraFGDataset(Dataset):
    """CSV -> spectra tensors + smiles + fg_labels (multi-label)."""

    def __init__(
        self,
        path,
        modality_names,
        spectrum_maxlen,
    ):
        super().__init__()
        self.csv_path = path
        self.modality_names = modality_names
        self.spectrum_maxlen = spectrum_maxlen

        rows = []
        with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        self.rows = rows


    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        data = {}
        missing = []
        for m_idx in self.modality_names:
            spec =parse_spectrum_list(r.get(m_idx, ""))
            if spec.numel() == 0:
                missing.append(m_idx)
            data[m_idx] = spec
        data['smiles'] = r.get("smiles", "") or r.get("SMILES", "")
        data['labels'] = parse_label_list(r.get("labels", ""))
        data["missing_modalities"] = missing
        return data
    

class SpectraFGCollater:
    def __init__(self, modality_names, spectrum_maxlen):
        self.modality_names = modality_names
        self.spectrum_maxlen = spectrum_maxlen

    def __call__(self, batch):

        data = {}
        # smiles as python list
        smiles = [b["smiles"] for b in batch]

        # labels: (B, K)
        labels = [b["labels"] for b in batch]
        labels = torch.stack(labels, dim=0).to(torch.float32)

        data["smiles"] = smiles
        data["labels"] = labels

        spec = []
        for m_idx, modality in enumerate(self.modality_names):
            spectradata = [b[modality] for b in batch]
            lengths = [spectrumdata.numel() for spectrumdata in spectradata]
            max_len = max(lengths) if lengths else 0
            if max_len <= 0:
                max_len = 1
            padding_list = []
            for b_idx, spectrumdata in enumerate(spectradata):
                if spectrumdata.numel() == 0:
                    padding_list.append(torch.zeros((max_len,), dtype=torch.float32))
                    continue
                if spectrumdata.numel() < max_len:
                    padding = torch.zeros((max_len - spectrumdata.numel(),), dtype=torch.float32)
                    padding_list.append(torch.cat([spectrumdata.to(torch.float32), padding], dim=0))
                else:
                    padding_list.append(spectrumdata.to(torch.float32))
            data[modality] = torch.stack(padding_list, dim=0)   # (B, Lmax)
            spec.append(data[modality])
        
        data["spec"] = torch.stack(spec, dim=1)  # (B, M, Lmax)
        return data