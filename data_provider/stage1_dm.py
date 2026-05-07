import logging
from torch.utils.data import DataLoader
from data_provider.spectra_dataset import SpectraDataset, SpectraCollater


logger = logging.getLogger(__name__)

class Stage1DM:
    """CSV -> spectra tensors + smiles + fg_labels (multi-label)."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.modality_names = config.modality_names
        self.spec_len = config.spec_len
        
        self.train_path = config.dataset.train
        self.val_path = config.dataset.val
        self.test_path = config.dataset.test

        logger.info("loading datasets...")
        self.train_dataset = SpectraDataset(self.train_path, self.modality_names)
        self.val_dataset = SpectraDataset(self.val_path, self.modality_names)
        self.test_dataset = SpectraDataset(self.test_path, self.modality_names)

        logger.info(
            "Dataset sizes — train: %d  val: %d  test: %d",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
        )

    def train_loader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=SpectraCollater(self.modality_names, self.spec_len),
            drop_last=True,
        )
        return train_loader
    
    def val_loader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=SpectraCollater(self.modality_names, self.spec_len),
            drop_last=False,
        )
        return val_loader

    def test_loader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=SpectraCollater(self.modality_names, self.spec_len),
            drop_last=False,
        )
        return test_loader
    
        




