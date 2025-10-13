from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import pandas as pd
import os


class CloudDataset(Dataset):
    def __init__(
        self,
        root_dir,
        channels: list[int],
        transform,
        split: str,
        use_multi_views: bool = False,
    ):
        """
        Cloud-38 dataset (4 channels: RGB-IR): https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset
        Image size: 384x384
        Num of channels: 4 (R, G, B, NIR)
        3 splits: train, valid, test
        """
        super().__init__()

        self.pytorch = True
        self.transform = transform
        self.channels = torch.tensor([c for c in channels])
        self.use_multi_views = use_multi_views
        ## `cloud38_split.csv` can be found in `assets/cloud38_split.csv`. We only use `38-Cloud_training` for this experiment
        self.df = pd.read_csv(os.path.join(root_dir, "38-Cloud_training", "cloud38_split.csv"))

        if split == "train":
            self.df = self.df[self.df["split"] == "train"]
        elif split == "valid":
            self.df = self.df[self.df["split"] == "valid"]
        elif split == "test":
            self.df = self.df[self.df["split"] == "test"]
        else:
            raise ValueError(f"split must be either train, valid, or test, got {split}")

        self.df = self.df.reset_index(drop=True)
        self.files = [self.combine_files(f) for f in self.df["path"]]

    def combine_files(self, r_file: Path):
        files = {
            "red": r_file,
            "green": r_file.replace("_red/red_", "_green/green_"),
            "blue": r_file.replace("_red/red_", "_blue/blue_"),
            "nir": r_file.replace("_red/red_", "_nir/nir_"),
            "gt": r_file.replace("_red/red_", "_gt/gt_"),
        }
        return files

    def __len__(self):
        return len(self.files)

    def open_as_array(self, idx, include_nir=False):

        raw_rgb = np.stack(
            [
                np.array(Image.open(self.files[idx]["red"])),
                np.array(Image.open(self.files[idx]["green"])),
                np.array(Image.open(self.files[idx]["blue"])),
            ],
            axis=2,
        )

        if include_nir:
            nir = np.expand_dims(np.array(Image.open(self.files[idx]["nir"])), 2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)

        ## shape of raw_rgb is (h, w, c)

        # normalize
        raw_rgb = raw_rgb / np.iinfo(raw_rgb.dtype).max
        return raw_rgb

    def open_mask(self, idx, add_dims=False):

        raw_mask = np.array(Image.open(self.files[idx]["gt"]))
        raw_mask = np.where(raw_mask == 255, 1, 0)

        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask

    def __getitem__(self, idx):

        x = self.open_as_array(idx, include_nir=True)
        label = self.open_mask(idx, add_dims=False)

        img_chw = x.transpose((2, 0, 1))
        combined = np.concatenate([img_chw, label[None, ...]], axis=0)
        combined = self.transform(combined)
        img_chw = combined[:-1]
        label = torch.tensor(combined[-1].copy(), dtype=torch.float)

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
        out = {"image": img_chw, "channels": channels, "label": label}
        return out

    def open_as_pil(self, idx):
        arr = 256 * self.open_as_array(idx)
        return Image.fromarray(arr.astype(np.uint8), "RGB")

    def __repr__(self):
        s = "Dataset class with {} files".format(self.__len__())

        return s
