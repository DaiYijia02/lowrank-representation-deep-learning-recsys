import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
import pandas
import numpy as np


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 split: str,
                 transform: Callable = None,
                 **kwargs):
        self.data_dir = Path(data_path)
        self.transforms = transform
        data = sorted([f for f in self.data_dir.iterdir() if str(f)[5] == "s"])
        label = sorted(
            [f for f in self.data_dir.iterdir() if str(f)[5] == "l"])
        self.data = data[:int(
            len(data) * 0.75)] if split == "train" else data[int(len(data) * 0.75):]
        self.label = label[:int(
            len(data) * 0.75)] if split == "train" else label[int(len(data) * 0.75):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = pandas.read_csv(self.data[idx], sep="\t", converters={'user': lambda x: x.strip(
            "[]"), 'docs_kaleness': lambda x: x.strip("[]").split(",")}, dtype={'user': float, 'docs_kaleness': float})
        label = pandas.read_csv(self.label[idx], sep="\t", converters={'next_user': lambda x: x.strip(
            "[]"), 'responses': lambda x: x.strip("[]").split(",")}, dtype={'next_user': float, 'responses': float})
        data_user = np.array(data['user'].values, dtype=np.float16)
        data_docs = np.asarray(np.array(
            [np.array(xi) for xi in data['docs_kaleness'].values]), dtype=np.float16)
        data_user = torch.Tensor(data_user)
        data_docs = torch.Tensor(data_docs)
        data = torch.cat((data_user.unsqueeze(1), data_docs), -1)
        label_response = np.asarray(
            np.array([np.array(xi) for xi in label['responses'].values]), dtype=np.float16)
        label_next_user = np.array(label['next_user'].values, dtype=np.float16)
        label_response = torch.Tensor(label_response)
        label_next_user = torch.Tensor(label_next_user)
        label = torch.cat((label_response, label_next_user.unsqueeze(1)), -1)

        if self.transforms is not None:
            data = self.transforms(data)

        return data, label


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 
    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:

        self.train_dataset = MyDataset(
            self.data_dir,
            split='train',
            transform=None
        )

        self.val_dataset = MyDataset(
            self.data_dir,
            split='test',
            transform=None
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=12,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
