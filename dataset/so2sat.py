import os
from typing import Union

import numpy as np
import torch
import random

from typing import List, Union
import h5py
import numpy as np
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader


class So2Sat(Dataset):
    """So2Sat"""

    normalize_mean: Union[List[float], None] = None
    normalize_std: Union[List[float], None] = None

    def __init__(
        self,
        path: str,
        transform,
        channels: List[int],
        split: str,  ## split: either train, valid, or test
        use_multi_views: bool = False,
    ) -> None:
        """Initialize the dataset."""
        super().__init__()

        self.channels = torch.tensor([c for c in channels])

        self.transform = transform
        ## read h5py file from `path`
        if split == "train":
            path = os.path.join(path, "training.h5")
        elif split == "valid":
            path = os.path.join(path, "validation.h5")
        elif split == "test":
            path = os.path.join(path, "testing.h5")
        else:
            raise ValueError(f"split must be either train, valid, or test, got {split}")

        self.file = h5py.File(path, "r")
        self.path = path
        self.use_multi_views = use_multi_views

    def __getitem__(self, index):
        ## https://github.com/zhu-xlab/So2Sat-LCZ42
        ## use both sen1 and sen2
        img_chw = np.concatenate(
            [
                self.file["sen1"][index].astype("float32"),
                self.file["sen2"][index].astype("float32"),
            ],
            axis=-1,
        )
        ## reorder the channels to c, h, w
        img_chw = np.transpose(img_chw, (2, 0, 1))

        label = self.file["label"][index].astype(int)
        img_chw = self.transform(img_chw)

        channels = self.channels.numpy()
        if self.use_multi_views:
            img_chw = img_chw[:, channels]
        else:
            if isinstance(img_chw, dict):
                for k in img_chw:
                    if k == "offsets":
                        continue
                    if isinstance(img_chw[k], list):
                        img_chw[k] = [img_chw_i[channels] for img_chw_i in img_chw[k]]
                    else:
                        img_chw[k] = img_chw[k][channels]
            else:
                img_chw = img_chw[channels]
                if isinstance(img_chw, np.ndarray):
                    img_chw = torch.tensor(img_chw.copy()).float()

        if sum(label) > 1:
            raise ValueError("More than one positive")

        for i, y in enumerate(label):
            if y == 1:
                label = i
                break
        out = {"image": img_chw, "channels": channels, "label": label}
        return out

    def __len__(self) -> int:
        return len(self.file["label"])

    @staticmethod
    def collate_fn(batch):
        """Filter out bad examples (None) within the batch."""
        batch = list(filter(lambda example: example is not None, batch))
        return default_collate(batch)


class So2SatAugmentation(object):
    def __init__(
        self,
        is_train: bool,
        means: list[float] = [0.4914, 0.4822, 0.4465],
        stds: list[float] = [0.2023, 0.1994, 0.2010],
        channel_mask=[],
    ):

        self.mean = np.array([m for m in means])[:, np.newaxis, np.newaxis]
        self.std = np.array([m for m in stds])[:, np.newaxis, np.newaxis]

        self.is_train = is_train
        self.channel_mask = list(channel_mask)

    def __call__(self, img) -> Union[list[torch.Tensor], torch.Tensor]:
        """
        Take a PIL image, generate its data augmented version
        """
        if img.shape[0] == len(self.mean):
            img = (img - self.mean) / self.std

        if self.is_train:
            # rotation
            r = random.randint(0, 3)
            img = np.rot90(img, r, (1, 2))

            # flip
            f = random.randint(0, 1)
            if f == 1:
                img = np.flip(img, 1)

            # flip
            f = random.randint(0, 1)
            if f == 1:
                img = np.flip(img, 2)

        if len(self.channel_mask) == 0:
            # do not mask channels
            return img
        else:
            # mask out the channels
            # NOTE: this channel mask index is relative / not absolute.
            # For instance, in JUMPCP where we have 8 channels.
            # If the data loader only sends over 3-channel images with channel 5, 6, 7.
            # The channel mask should be [0] if we want to mask out 5.
            img[self.channel_mask, :, :] = 0

            return img


def get_so2sat_dataloaders(
    root_dir: str,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    channels: dict[str, list[int]],
    normalization: dict[str, list[float]] | None,
    **kwargs,
) -> tuple[DataLoader, DataLoader, dict[str, DataLoader]]:

    if normalization is None:
        mean_data = None
        std_data = None
    else:
        mean_data = normalization["mean"]
        std_data = normalization["std"]

    transform_train = So2SatAugmentation(is_train=True, means=mean_data, stds=std_data, channel_mask=[])
    transform_eval = So2SatAugmentation(is_train=False, means=mean_data, stds=std_data, channel_mask=[])

    train_set = So2Sat(path=root_dir, split="train", transform=transform_train, channels=channels["train"])
    valid_set = So2Sat(path=root_dir, split="valid", transform=transform_eval, channels=channels["valid"])

    test_loaders = {}
    for test_name in channels.keys():
        if test_name.startswith("test"):
            test_set = So2Sat(path=root_dir, split="test", transform=transform_eval, channels=channels[test_name])
            test_loader_i = DataLoader(
                test_set,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False,
            )
            test_loaders[test_name] = test_loader_i

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        valid_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader, test_loaders
