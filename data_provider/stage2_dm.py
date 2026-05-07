from torch.utils.data import DataLoader
from data_provider.spectra_fg_dataset import SpectraFGDataset, SpectraFGCollater

class Stage2DM:
    """CSV -> spectra tensors + smiles + fg_labels (multi-label)."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.modality_names = self.config.modality_names
        self.spectrum_maxlen = self.config.spectrum_maxlen
        
        self.train_path = self.config.dataset.train
        self.val_path = self.config.dataset.val
        self.test_path = self.config.dataset.test

        self.train_dataset = SpectraFGDataset(self.train_path, self.modality_names, self.spectrum_maxlen)
        self.val_dataset = SpectraFGDataset(self.val_path, self.modality_names, self.spectrum_maxlen)
        self.test_dataset = SpectraFGDataset(self.test_path, self.modality_names, self.spectrum_maxlen)


    def train_loader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=SpectraFGCollater(self.modality_names, self.spectrum_maxlen),
            drop_last=True,
        )
        return train_loader
    
    def val_loader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=SpectraFGCollater(self.modality_names, self.spectrum_maxlen),
            drop_last=False,
        )
        return val_loader
    
    def test_loader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=SpectraFGCollater(self.modality_names, self.spectrum_maxlen),
            drop_last=False,
        )
        return test_loader
    
        




