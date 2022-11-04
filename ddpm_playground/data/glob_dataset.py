from pathlib import Path
from typing import Optional, Callable

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class GlobDataset(Dataset):
    def __init__(
        self,
        root_path: Path,
        pattern: str = "*.png",
        recursive: bool = False,
        repeat: int = 1,
        loader: Callable = Image.open,
        transform: Optional[Callable] = None,
    ):
        self.root_path = root_path
        self.pattern = pattern
        self.recursive = recursive
        self.repeat = repeat
        self.loader = loader
        self.transform = transform

        if recursive:
            glob_func = self.root_path.rglob
        else:
            glob_func = self.root_path.glob
        self.sample_paths = list(glob_func(pattern)) * self.repeat

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index: int):
        sample_path = self.sample_paths[index]
        sample = self.loader(sample_path)
        return self.transform(sample)


class DummyDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index: int):
        return 0


class GlobDatasetForGM(pl.LightningDataModule):
    def __init__(
        self,
        root_path: str,
        pattern: str = "*.jpg",
        recursive: bool = False,
        repeat: int = 1,
        image_size: int = 128,
        batch_size: int = 256,
        num_workers: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.root_path = Path(root_path)
        self.pattern = pattern
        self.recursive = recursive
        self.repeat = repeat
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        transform = T.Compose([
            T.Resize(self.image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(self.image_size),
            T.ToTensor()
        ])

        dataset = GlobDataset(
            root_path=self.root_path,
            pattern=self.pattern,
            recursive=self.recursive,
            repeat=self.repeat,
            loader=Image.open,
            transform=transform,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(DummyDataset(), num_workers=0)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(DummyDataset(), num_workers=0)
