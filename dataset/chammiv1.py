import os
from typing import Callable
import tarfile
from io import BytesIO
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import Tensor
from torch.utils.data import Dataset
import time
import skimage.io
import socket
from torch.utils.data import DataLoader
from torchvision import transforms
from functools import partial
from typing import Any

from dataset.dataset_utils import do_nothing


class SingleCellDataset(Dataset):
    """Single cell chunk."""

    def __init__(
        self,
        csv_path: str,
        chunk: str,
        root_dir: str,
        is_train: bool,
        target_labels: str = "label",
        transform: Callable | dict[str, Callable] | None = None,
        use_tar_file=False,
    ):
        """
        @param csv_path: Path to the csv file with metadata.
             e.g., "metadata/morphem70k_v2.csv".
        You should copy this file to the dataset folder to avoid modifying other config.

        Note: Allen was renamed to WTC-11 in the paper.
        @param chunk: "Allen" (i.e., WTC-11), "HPA", "CP", or "morphem70k"to use all 3 chunks
        @param root_dir: root_dir: Directory with all the images.
        @param is_train: True for training set, False for using all data
        @param target_labels: label column in the csv file
        @param transform: data transform to be applied on a sample.
        """
        # assert chunk in [
        #     "Allen",
        #     "HPA",
        #     "CP",
        #     "morphem70k",
        # ], "chunk must be one of 'Allen', 'HPA', 'CP', 'morphem70k'"

        ## Note that for rebuttal, chunk can be a single, or a combination of 2 datasets, or all 3 (i.e., 'morphem70k') separated by "_",
        # e.g., "Allen_HPA", or "Allen_CP", or "CP"
        self.chunk_names = chunk.split("_")

        self.use_tar_file = use_tar_file
        # print(f"------------ use_tar_file: {self.use_tar_file} ------------")

        self.tar_file = {}

        self.chunk = chunk
        self.is_train = is_train

        ## read csv file for the chunk
        self.metadata = pd.read_csv(csv_path)
        ## filter by chunk if chunk is not "morphem70k"
        if chunk == "Allen_HPA_CP":
            pass
        elif any(x in chunk for x in ["Allen", "HPA", "CP"]):
            if chunk in ["Allen", "HPA", "CP"]:
                self.metadata = self.metadata[self.metadata["chunk"] == chunk]
            else:  ## combination of 2 datasets
                chunk1, chunk2 = chunk.split("_")
                ## filter, keep only rows with chunk1 or chunk2
                self.metadata = self.metadata[self.metadata["chunk"].isin([chunk1, chunk2])]
        if is_train:
            self.metadata = self.metadata[self.metadata["train_test_split"] == "Train"]

        self.metadata = self.metadata.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.target_labels = target_labels

        ## classes on training set:

        if chunk == "Allen":
            self.train_classes_dict = {
                "M0": 0,
                "M1M2": 1,
                "M3": 2,
                "M4M5": 3,
                "M6M7_complete": 4,
                "M6M7_single": 5,
            }

        elif chunk == "HPA":
            self.train_classes_dict = {
                "golgi apparatus": 0,
                "microtubules": 1,
                "mitochondria": 2,
                "nuclear speckles": 3,
            }

        elif chunk == "CP":
            self.train_classes_dict = {
                "BRD-A29260609": 0,
                "BRD-K04185004": 1,
                "BRD-K21680192": 2,
                "DMSO": 3,
            }

        elif chunk == "morphem70k":  # all 3 chunks (i.e., "morphem70k")
            self.train_classes_dict = {
                "BRD-A29260609": 0,
                "BRD-K04185004": 1,
                "BRD-K21680192": 2,
                "DMSO": 3,
                "M0": 4,
                "M1M2": 5,
                "M3": 6,
                "M4M5": 7,
                "M6M7_complete": 8,
                "M6M7_single": 9,
                "golgi apparatus": 10,
                "microtubules": 11,
                "mitochondria": 12,
                "nuclear speckles": 13,
            }
        else:  ## rebuttal, chunk would be 1 or 2 datasets, separated by "_", e.g., "Allen_HPA", or "Allen_CP", or "CP"
            self.train_classes_dict = {}
            max_val = 0
            if "Allen" in chunk:
                self.train_classes_dict.update(
                    {
                        "M0": max_val,
                        "M1M2": max_val + 1,
                        "M3": max_val + 2,
                        "M4M5": max_val + 3,
                        "M6M7_complete": max_val + 4,
                        "M6M7_single": max_val + 5,
                    }
                )
                max_val = max_val + 6

            if "HPA" in chunk:
                self.train_classes_dict.update(
                    {
                        "golgi apparatus": max_val,
                        "microtubules": max_val + 1,
                        "mitochondria": max_val + 2,
                        "nuclear speckles": max_val + 3,
                    }
                )
                max_val = max_val + 4

            if "CP" in chunk:
                self.train_classes_dict.update(
                    {
                        "BRD-A29260609": max_val,
                        "BRD-K04185004": max_val + 1,
                        "BRD-K21680192": max_val + 2,
                        "DMSO": max_val + 3,
                    }
                )
                max_val = max_val + 4

        self.test_classes_dict = None  ## Not defined yet

    def __len__(self):
        return len(self.metadata)

    @staticmethod
    def _fold_channels(image: np.ndarray, channel_width: int, mode="ignore") -> Tensor:
        """
        Re-arrange channels from tape format to stack tensor
        @param image: (h, w * c)
        @param channel_width:
        @param mode:
        @return: Tensor, shape of  (c, h, w)  in the range [0.0, 1.0]
        """
        # convert to shape of (h, w, c),  (in the range [0, 255])
        output = np.reshape(image, (image.shape[0], channel_width, -1), order="F")

        if mode == "ignore":
            # Keep all channels
            pass
        elif mode == "drop":
            # Drop mask channel (last)
            output = output[:, :, 0:-1]
        elif mode == "apply":
            # Use last channel as a binary mask
            mask = output["image"][:, :, -1:]
            output = output[:, :, 0:-1] * mask
        output = torchvision.transforms.ToTensor()(output)

        return output

    def __getitem__(self, idx):
        channel_width = self.metadata.loc[idx, "channel_width"]
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.use_tar_file:
            image_path = self.metadata.loc[idx, "file_path"]
            folder, img_path = image_path.split("/", 1)
            img_path = "./" + img_path.replace("crops/", "", 1)
            tar_file_path = os.path.join(self.root_dir, folder, f"{folder}.tar")
            if self.tar_file.get(folder) is None:
                self.tar_file[folder] = tarfile.open(tar_file_path)

            image = skimage.io.imread(BytesIO(self.tar_file[folder].extractfile(img_path).read()))
        else:
            img_path = os.path.join(self.root_dir, self.metadata.loc[idx, "file_path"])
            image = skimage.io.imread(img_path)

        image = self._fold_channels(image, channel_width)

        if self.is_train:
            label = self.metadata.loc[idx, self.target_labels]
            label = self.train_classes_dict[label]
            label = torch.tensor(label)
        else:
            label = None  ## for now, we don't need labels for evaluation. It will be provided later in evaluation code.

        if self.chunk == "morphem70k" or self.chunk not in [
            "Allen",
            "HPA",
            "CP",
        ]:  ## using all 3 datasets
            chunk = self.metadata.loc[idx, "chunk"]
            if self.transform:
                image = self.transform[chunk](image)
            if self.is_train:
                data = {"chunk": chunk, "image": image, "label": label}
            else:
                data = {"chunk": chunk, "image": image}
        else:  ## using single chunk
            if self.transform:
                image = self.transform(image)
            if self.is_train:
                data = image, label
            else:
                data = image
        return data


def chammiv2_collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | list[list[int]] | int]:
    channel_ids_list = [item["channel_ids"] for item in batch]
    all_images = [item["image"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    num_images = len(all_images)
    num_channels_list = [len(channels) for channels in channel_ids_list]
    max_num_channels = max(num_channels_list)
    channel_masks = torch.arange(max_num_channels).unsqueeze(0) < torch.tensor(num_channels_list).unsqueeze(1)

    # Initialize a tensor for padding with zeros
    h, w = batch[0]["image"].shape[-2:]
    images = torch.zeros(num_images, max_num_channels, h, w, dtype=torch.float32)

    for i, (channel_ids, image) in enumerate(zip(channel_ids_list, all_images)):
        num_channels = len(channel_ids)
        images[i, :num_channels] = image

    return {
        "image": images,
        "label": labels,
        "max_channel_len": max_num_channels,
        "channel_ids_list": channel_ids_list,
        "valid_channel_masks": channel_masks,
    }


def morphem70k_collate_by_chunk(data):
    """
    group images by chunk

    data: list of dict, each has 3 keys: 'chunk', 'image', 'label'
    return : a dict
    {
        "Allen":
        {
            "image": [],
            "label": []
        },
        "HPA": {...},
        "CP": {...}
    }
    """
    out = {}
    ### create 3 keys: Allen, HPA, CP, each with a list of images and labels
    for d in data:
        if d["chunk"] not in out:
            out[d["chunk"]] = {"image": [], "label": []}
        out[d["chunk"]]["image"].append(d["image"])
        out[d["chunk"]]["label"].append(d["label"])

    for chunk in out:
        if len(out[chunk]["image"]) > 0:
            out[chunk]["image"] = torch.stack(out[chunk]["image"], dim=0)
            out[chunk]["label"] = torch.tensor(out[chunk]["label"])
    return out


def morphem70k_collate_with_padding(data: list[dict], channel_mapper: dict[str, list[int]]) -> dict:
    """
    data: list of dict, each has 3 keys: 'chunk', 'image', 'label'
    channel_mapper: a dict, mapping chunk name to channel ids
    return:
        {
            "image": images,
            "label": labels,
            "max_channel_len": max_num_channels,
            "channel_ids_list": channel_ids_list,
            "channel_mask": channel_masks,
        }

    """
    ## generate `channel_ids` for each image, depending on the chunk
    for i, item in enumerate(data):
        data[i]["channel_ids"] = channel_mapper[item["chunk"]]

    ## re-use the same collate function as chammiv2
    res = chammiv2_collate_fn(data)
    return res


def get_chammiv1_dataloaders(
    root_dir: str,
    csv_path: str,
    img_size: tuple[int, int],
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    training_chunk,
    normalization: dict[str, dict[str, list[float]]],
    collate_fn_name: str,
    channel_ids: dict[str, list[int]] | None = None,
    **kwargs,
) -> tuple[DataLoader, None, dict[str, DataLoader]]:

    train_loader, valid_loader, test_loaders = None, None, None
    chunk_names = {"Allen": None, "HPA": None, "CP": None}
    use_mean_std = normalization["use_mean_std"]
    ## get mean, std for each of the chunks
    for chunk_name in chunk_names:
        mean_data = normalization[chunk_name]["mean"]
        std_data = normalization[chunk_name]["std"]
        chunk_names[chunk_name] = (mean_data, std_data)

    #### make train dataloader
    transforms_train = {}
    for chunk_name in chunk_names:
        mean_data, std_data = chunk_names[chunk_name]
        transform_train_chunk = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                # transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                ## transforms.ToTensor(), input is already a Tensor
                transforms.Normalize(mean_data, std_data) if use_mean_std else do_nothing,  # self_normalize,
            ]
        )
        transforms_train[chunk_name] = transform_train_chunk

    train_set = SingleCellDataset(
        csv_path,
        chunk=training_chunk,  #  "morphem70k" if using all 3 chunks
        root_dir=root_dir,
        is_train=True,
        transform=transforms_train,
    )

    if collate_fn_name == "group_by_chunk":
        collate_fn = morphem70k_collate_by_chunk
    elif collate_fn_name == "padding":
        ## partial function to pass channel_mapper
        assert channel_ids is not None, "channel_ids must be provided for padding collate_fn"
        collate_fn = partial(morphem70k_collate_with_padding, channel_mapper=channel_ids)
    else:
        raise ValueError(f"collate_fn {collate_fn_name} is not supported")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    #### make test dataloaders
    test_loaders = {}
    for chunk_name in chunk_names:
        mean_data, std_data = chunk_names[chunk_name]

        transform_eval = transforms.Compose(
            [
                transforms.Resize(img_size, antialias=True),
                transforms.CenterCrop(img_size),
                transforms.Normalize(mean_data, std_data) if use_mean_std else do_nothing,  # self_normalize,
            ]
        )

        test_set = SingleCellDataset(
            csv_path,
            chunk_name,
            root_dir,
            is_train=False,
            transform=transform_eval,
        )
        ## IMPORTANT: set shuffle to False for test set. Otherwise, the order of the test set will be different when evaluating
        test_loader = DataLoader(
            test_set,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        test_loaders[chunk_name] = test_loader

    return train_loader, valid_loader, test_loaders  ## valid_loader is None
