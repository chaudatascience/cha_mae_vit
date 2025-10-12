import numpy as np
import pandas as pd
import torch
import os
from omegaconf import ListConfig
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
import random
from torch.utils.data import DataLoader
from typing import Union, Optional
import h5py


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.aug = A.GaussianBlur(sigma_limit=(self.radius_min, self.radius_max))

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return self.aug(images=[img])[0]


def RandomPadCrop(size):
    """
    Crops image to range of `scale` inputs and resize to `size`
    """
    return A.Compose(
        [
            A.PadIfNeeded(
                min_width=256,
                min_height=256,
                position="random",
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            A.RandomCrop(width=size, height=size),
        ]
    )


def RandomPadAndCropCenter(size):
    """
    Crops image to range of `scale` inputs and resize to `size`
    """
    return A.Compose(
        [
            A.PadIfNeeded(
                min_width=320,
                min_height=320,
                position="random",
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            A.CenterCrop(width=size, height=size),
            # A.ChannelDropout(p=0.2, channel_drop_range=(1, 3)),
        ]
    )


class CellAugmentation(object):
    def __init__(
        self,
        is_train: bool,
        global_resize: int = 224,
        means: list[float] = [0.4914, 0.4822, 0.4465],
        stds: list[float] = [0.2023, 0.1994, 0.2010],
        brightness: bool = False,
        use_coarse_dropout: bool = True,
        channel_mask=[],
    ):
        """
        MulticropAugmentation strategy, as developed by M. Caron
        https://arxiv.org/pdf/2006.09882.pdf.
        ASSUMES images are from the distribution N(0,I).
        global_crops_scale: List[float]
            List of (a, b) that defines the scale, sampled uniformly, at which
            to crop the image for the global crop. For instance, (.8, 1.0) will mean that each
            global crop will shrink the original image to be x ~ Uniform([.8, 1.])
            % of the original size.
        local_crops_scale: List[float]
            List of (a, b) that defines the scale, sampled uniformly, at which
            to crop the image for the local crop. For instance, (.6, .8) will mean that each
            local crop will shrink the original image to be x ~ Uniform([.6, .8])
            % of the original size.
        n_local_crops_per_image : int
            number of of local crops per image in the original pair.
            n_local_crops_per_image==0 implies just a single pair of
            reference images (global crops only), whereas n_local_crops_per_image>0
            (as in DINO) implies applying a local crop to each image n_local_crops_per_image
            times.
        global_resize: int
            After cropping image to be of global_crops_scale size of the original size,
            will resize to this value. 224 by default.
        local_resize: int
            After cropping image to be of local_crops_scale size of the original size,
            will resize to this value. 96 by default.
        """
        flip_rotate = A.OneOf(
            [
                A.HorizontalFlip(),
                A.VerticalFlip(),
                ## Note: `limit=90` doesn't rotate exactly 90 degrees,
                # but by an angle selected randomly from the range [-90, 90]
                A.Rotate(limit=90),
                A.Rotate(limit=180),
                A.Rotate(limit=270),
            ]
        )

        if brightness:
            print("Apply brightness change after flip and rotate")
            flip_rotate = A.Compose([flip_rotate, A.RandomBrightness()])

        normalize = A.Compose([A.Normalize(means, stds), ToTensorV2()])

        self.is_train = is_train
        self.normalize = normalize

        # global crop
        if use_coarse_dropout:
            coarse_dropout = A.CoarseDropout(max_holes=10, max_height=10, max_width=10)
        else:
            coarse_dropout = A.NoOp()

        self.global_transform1 = A.Compose(
            [
                RandomPadCrop(global_resize),
                flip_rotate,
                A.Defocus(radius=(1, 3)),
                coarse_dropout,
                normalize,
            ]
        )

        self.channel_mask = list(channel_mask)

    def __call__(self, image) -> Union[list[torch.Tensor], torch.Tensor]:
        """
        Take a PIL image, generate its data augmented version
        """
        img = np.asarray(image)  #####  HWC

        if self.is_train:
            img = self.global_transform1(image=img)["image"]
        else:
            img = self.normalize(image=img)["image"]

        if len(self.channel_mask) == 0:
            # do not mask channels
            return img
        else:
            # mask out the channels
            # NOTE: this channel mask index is relative / not absolute.
            # For instance, in JUMPCP where we have 8 channels.
            # If the data loader only sends over 3-channel images with channel 5, 6, 7.
            # The channel mask should be [0] if we want to mask out 5.
            img[self.channel_mask, :, :] = torch.zeros_like(img[self.channel_mask, :, :])

            return img


def load_meta_data(base_path: str):
    PLATE_TO_ID = {"BR00116991": 0, "BR00116993": 1, "BR00117000": 2}
    FIELD_TO_ID = dict(zip([str(i) for i in range(1, 10)], range(9)))
    WELL_TO_ID = {}
    for i in range(16):
        for j in range(1, 25):
            well_loc = f"{chr(ord('A') + i)}{j:02d}"
            WELL_TO_ID[well_loc] = len(WELL_TO_ID)

    WELL_TO_LBL = {}

    PLATE_MAP = {
        "compound": f"{base_path}/JUMP-Target-1_compound_platemap.tsv",
        "crispr": f"{base_path}/JUMP-Target-1_crispr_platemap.tsv",
        "orf": f"{base_path}/JUMP-Target-1_orf_platemap.tsv",
    }
    META_DATA = {
        "compound": f"{base_path}/JUMP-Target-1_compound_metadata.tsv",
        "crispr": f"{base_path}/JUMP-Target-1_crispr_metadata.tsv",
        "orf": f"{base_path}/JUMP-Target-1_orf_metadata.tsv",
    }

    for perturbation in PLATE_MAP.keys():
        df_platemap = pd.read_parquet(PLATE_MAP[perturbation])
        df_metadata = pd.read_parquet(META_DATA[perturbation])
        df = df_metadata.merge(df_platemap, how="inner", on="broad_sample")

        if perturbation == "compound":
            target_name = "target"
        else:
            target_name = "gene"

        codes, uniques = pd.factorize(df[target_name])
        codes += 1  # set none (neg control) to id 0
        assert min(codes) == 0
        # print(f"...{target_name} has {len(uniques)} unique values")
        WELL_TO_LBL[perturbation] = dict(zip(df["well_position"], codes))

    return PLATE_TO_ID, FIELD_TO_ID, WELL_TO_ID, WELL_TO_LBL


class JUMPCP(Dataset):
    """JUMPCP dataset"""

    normalize_mean: Union[list[float], None] = None
    normalize_std: Union[list[float], None] = None
    NUM_TOTAL_CHANNELS = 8

    def __init__(
        self,
        path: str,
        split: str,  # train, valid or test
        transform,
        channels: Union[list[int], None],
        use_hdf5: bool = True,
        channel_mask: bool = False,
        scale: float = 1,
        perturbation_list: ListConfig[str] = ["compound"],
        cyto_mask_path_list: ListConfig[str] = None,
    ) -> None:
        """Initialize the dataset."""
        self.root_dir = path + "/" if path[-1] != "/" else path
        self.use_hdf5 = use_hdf5

        if cyto_mask_path_list is None:
            cyto_mask_path_list = [os.path.join(self.root_dir, "jumpcp/BR00116991.pq")]
        # read the cyto mask df
        df = pd.concat([pd.read_parquet(path) for path in cyto_mask_path_list], ignore_index=True)
        df = self.get_split(df, split)

        self.data_path = list(df["path"])
        self.data_id = list(df["ID"])
        self.well_loc = list(df["well_loc"])

        self.file = h5py.File(os.path.join(self.root_dir, f"jumpcp/jumpcp_{split}.h5"), "r")
        assert len(perturbation_list) == 1
        self.perturbation_type = perturbation_list[0]

        if type(channels[0]) is str:
            # channel is separated by hyphen
            self.channels = torch.tensor([int(c) for c in channels[0].split("-")])
        else:
            self.channels = torch.tensor([c for c in channels])
        if scale is None and channel_mask:
            self.scale = float(self.NUM_TOTAL_CHANNELS) / len(self.channels)
        else:
            self.scale = scale  # scale the input to compensate for input channel masking

        if self.scale != 1:
            print(f"------ Scaling the input to compensate for channel masking, scale={self.scale} ------")

        # print(f"------ channels: {self.channels.numpy()} ------")

        self.transform = transform

        meta_data_path = os.path.join(self.root_dir, "jumpcp/platemap_and_metadata")
        self.plate2id, self.field2id, self.well2id, self.well2lbl = load_meta_data(meta_data_path)

        self.channel_mask = channel_mask

    def get_split(self, df, split_name, seed=0):
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
        m = len(df.index)
        train_end = int(0.6 * m)
        validate_end = int(0.2 * m) + train_end

        if split_name == "train":
            return df.iloc[perm[:train_end]]
        elif split_name == "valid":
            return df.iloc[perm[train_end:validate_end]]
        elif split_name == "test":
            return df.iloc[perm[validate_end:]]
        else:
            raise ValueError("Unknown split")

    def __getitem__(self, index):
        if self.well_loc[index] not in self.well2lbl[self.perturbation_type]:
            # this well is not labeled
            return None
        ## EDIT: use local img
        img_path = self.data_path[index].replace("s3://insitro-research-2023-context-vit/", self.root_dir)

        ## read npy img
        if self.use_hdf5:  ## one big file storing all images, to reduce number of files
            img_name = os.path.basename(img_path)
            img_chw = np.array(self.file[img_name])
        else:  ## each image is stored in a separate numpy file
            img_chw = np.load(img_path)

        if img_chw is None:
            return None

        img_hwc = img_chw.transpose(1, 2, 0)
        img_chw = self.transform(img_hwc)

        channels = self.channels.numpy()

        assert type(img_chw) is not list, "Only support jumpcp for supervised training"

        if self.scale != 1:
            # scale the image pixels to compensate for the masked channels
            # used in inference
            img_chw *= self.scale

        # mask out channels
        if self.channel_mask:
            # mask out unselected channels by setting their pixel values to 0
            unselected = [c for c in range(len(img_chw)) if c not in channels]
            img_chw[unselected] = 0
        else:
            img_chw = img_chw[channels]

        return {
            "image": img_chw,
            # "channels": channels,
            "label": self.well2lbl[self.perturbation_type][self.well_loc[index]],
        }

    def __len__(self) -> int:
        return len(self.data_path)

    @staticmethod
    def collate_fn(batch):
        """Filter out bad examples (None) within the batch."""
        batch = list(filter(lambda example: example is not None, batch))
        return default_collate(batch)


def get_jumpcp_dataloaders(
    root_dir: str,
    img_size: tuple[int, int],
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    channels: dict[str, list[int]],
    normalization: dict[str, list[float]] | None,
    use_hdf5: bool = True,
    **kwargs,
) -> tuple[DataLoader, DataLoader, dict[str, DataLoader]]:

    if normalization is None:
        mean_data = None
        std_data = None
    else:
        mean_data = normalization["mean"]
        std_data = normalization["std"]

    transform_train = CellAugmentation(is_train=True, means=mean_data, stds=std_data, global_resize=img_size[0])
    transform_eval = CellAugmentation(is_train=False, means=mean_data, stds=std_data, global_resize=img_size[0])

    train_set = JUMPCP(path=root_dir, split="train", transform=transform_train, channels=channels["train"], use_hdf5=use_hdf5)
    valid_set = JUMPCP(path=root_dir, split="valid", transform=transform_eval, channels=channels["valid"], use_hdf5=use_hdf5)

    test_loaders = {}
    for test_name in channels.keys():
        if test_name.startswith("test"):
            test_set = JUMPCP(path=root_dir, split="test", transform=transform_eval, channels=channels[test_name], use_hdf5=use_hdf5)
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
