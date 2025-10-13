import os

## add the parent directory to the path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import get_rank
import torch.distributed.nn
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_rank
import torch.nn as nn
from einops import rearrange


# Utility methods
def soft_dice_loss(input: Tensor, target: Tensor, smooth: float):
    loss = 1 - _soft_dice_coeff(input, target, smooth, focusing_param=1.0)
    return loss


def focal_dice_loss(input: Tensor, target: Tensor, smooth: float, focusing_param: float):
    loss = 1 - _soft_dice_coeff(input, target, smooth, focusing_param)
    return loss


def _soft_dice_coeff(input: Tensor, target: Tensor, smooth: float, focusing_param: float) -> Tensor:
    i = torch.sum(target)
    j = torch.sum(input)
    intersection = torch.sum(target * input)
    # formula is 2*(p_t*p_e)^gamma/(p_t+p_e)
    score = (2.0 * (intersection**focusing_param) + smooth) / (i + j + smooth)
    return score.mean()


def focal_binary_loss(input: Tensor, target: Tensor, focusing_param: float, eps: float = 1e-6):
    targets = target.view(-1)
    probs = input.view(-1)

    # Formula is
    # -(p_t*(1-p_e)^gamma)*log(p_e) for p_t=1; and
    # -(1-p_t)*(p_e)^gamma)*log(1-p_e) for p_t=0;
    # from : https://arxiv.org/pdf/1708.02002.pdf
    losses = -(targets * torch.pow((1.0 - probs), focusing_param) * torch.log(probs + eps)) - (
        (1.0 - targets) * torch.pow(probs, focusing_param) * torch.log(1.0 - probs + eps)
    )
    loss = torch.mean(losses)
    return loss


def soft_cldice(input, target, iter=100, smooth=1):
    score = cldice(input, target, iter, smooth)
    return 1 - score.mean()


class DiceBCELoss(nn.Module):
    def __init__(
        self,
        loss_type: str = "BCE+Dice",
        focusing_param_bce: float = 2,
        focusing_param_dice: float = 0.5,
    ):
        super(DiceBCELoss, self).__init__()
        self.loss_type = loss_type
        self.focusing_param_bce = focusing_param_bce
        self.focusing_param_dice = focusing_param_dice

        # Fixed parameters
        self.smooth = 1.0
        self.eps: float = 1e-6
        self.bce_loss: nn.Module = nn.BCELoss()

        self.loss_fn = self._get_loss_fn_for_type(loss_type)

    def _get_loss_fn_for_type(self, loss_type: str):
        loss_fns = {
            "BCE+Dice": self.bce_plus_dice,
        }
        return loss_fns[loss_type]

    def __call__(self, input: Tensor, target: Tensor):
        return self.loss_fn(input, target)

    def bce_plus_dice(self, input: Tensor, target: Tensor):
        a = self.bce_loss(input, target)
        b = soft_dice_loss(input, target, self.smooth)
        return a + b

    def bce_plus_focal_dice(self, input: Tensor, target: Tensor):
        a = self.bce_loss(input, target)
        b = focal_dice_loss(input, target, self.smooth, self.focusing_param_dice)
        return a + b

    def focal_bce_plus_dice(self, input: Tensor, target: Tensor):
        a = focal_binary_loss(input, target, self.focusing_param_bce)
        b = soft_dice_loss(input, target, self.smooth)
        return a + b

    def focal_bce_plus_focal_dice(self, input: Tensor, target: Tensor):
        a = focal_binary_loss(input, target, self.focusing_param_bce)
        b = focal_dice_loss(input, target, self.smooth, self.focusing_param_dice)
        return a + b

    def dice_cldice_loss(self, input: Tensor, target: Tensor):
        a = soft_dice_loss(input, target, self.smooth)
        b = soft_cldice(input, target)
        return a + b


def pairwise_distance_v2(proxies, x, squared=False):
    if squared:
        return (torch.cdist(x, proxies, p=2)) ** 2
    else:
        return torch.cdist(x, proxies, p=2)


def custom_cross_entropy_loss(predictions, targets, special_classes, weighted_for_rare_classes_loss: float = 1):
    """
    Custom cross-entropy loss function.

    Args:
        predictions (torch.Tensor): The raw model outputs (logits).
                                    Shape: (batch_size, num_classes)
        targets (torch.Tensor): The true class labels.
                                Shape: (batch_size)
        special_classes (list or tuple): A list of the special class indices.
                                        Example: [8, 9]
    """
    epsilon = 1e-9

    special_classes_tensor = torch.tensor(special_classes, device=targets.device)
    probs = F.softmax(predictions, dim=1)

    # mask
    is_special_target = torch.isin(targets, special_classes_tensor)  # True: special class, False: normal class
    is_normal_target = ~is_special_target

    #### Calculate loss for NORMAL samples
    normal_targets = targets[is_normal_target]
    normal_probs = probs[is_normal_target]

    if normal_targets.numel() > 0:
        # Standard cross-entropy
        prob_of_correct_class = normal_probs.gather(1, normal_targets.unsqueeze(1)).squeeze()
        loss_normal = -torch.log(prob_of_correct_class + epsilon)
    else:
        loss_normal = torch.tensor([], device=predictions.device)

    ####  Calculate loss for SPECIAL samples
    loss_special = torch.tensor([], device=predictions.device)
    if weighted_for_rare_classes_loss > 0:
        special_probs = probs[is_special_target]

        ## If there are any special samples, calculate their loss
        if special_probs.shape[0] > 0:
            # Take the maximum probability among special classes for each sample
            prob_of_special_group = special_probs[:, special_classes].max(dim=1)[0]
            loss_special = -torch.log(prob_of_special_group + epsilon)
            loss_special = loss_special * weighted_for_rare_classes_loss  # apply weighting

        # 5. Combine losses and take the mean
        total_loss = torch.cat((loss_normal, loss_special)).mean()
    else:
        total_loss = loss_normal.mean()
    return total_loss


def ortho_proj_loss_fn_v2(
    features,
    labels,
    gamma_s,
    gamma_d,
    reverse_pos_pairs: bool,
    use_square: bool,
    patch_mask: Tensor | None = None,
):
    """
    Compute Token Diversification Loss in DiChaViT paper
    features: shape (b, num_tokens, d)
    patch_mask: shape (b, num_tokens), mask out some patches when computing the loss: 1: visible, 0: masked+padding if any
    labels: shape (num_tokens)
    gamma_s, gamma_d: lambda_s and lambda_d in E.q (2) and (3) in DiChaViT paper
    reverse_pos_pairs: If true, we want each token to be orthogonal to all other tokens, regarless of their channels.
    """
    device = features.device
    #  features are normalized
    features = F.normalize(features, p=2, dim=-1)

    labels = labels[None, :, None]  # extend dims

    mask = torch.eq(labels, labels.transpose(-2, -1)).bool().to(device)
    eye = torch.eye(mask.shape[-2], mask.shape[-1]).bool().to(device).unsqueeze(0)

    mask_pos = mask.masked_fill(eye, 0).float()
    mask_neg = (~mask).float()
    if patch_mask is not None:
        patch_mask = patch_mask[:, :, None].bool()
        mask_pos = mask_pos * patch_mask
        mask_neg = mask_neg * patch_mask
    dot_prod = torch.matmul(features, features.transpose(-2, -1))

    mask_pos_sum = mask_pos.sum(dim=(-2, -1)) + 1e-6
    mask_neg_sum = mask_neg.sum(dim=(-2, -1)) + 1e-6

    pos_pairs_mean = (mask_pos * dot_prod).sum(dim=(-2, -1)) / mask_pos_sum
    neg_pairs_mean = (mask_neg * dot_prod).sum(dim=(-2, -1)) / mask_neg_sum

    if use_square:
        neg_pairs_mean = neg_pairs_mean**2

    if reverse_pos_pairs:
        if use_square:
            pos_pairs_mean = pos_pairs_mean**2
        loss = gamma_s * pos_pairs_mean + gamma_d * neg_pairs_mean
    else:
        loss = gamma_s * (1.0 - pos_pairs_mean) + gamma_d * neg_pairs_mean
    return loss.mean()


def compute_proxy_loss(proxies, img_emb, gt_imgs, scale: float | nn.Parameter) -> Tensor:
    """
    proxies: shape of (num_classes, dim)
    img_emb: shape of (num_imgs, dim)
    gt_imgs: shape of (num_imgs)
    """
    proxies_emb = scale * F.normalize(proxies, p=2, dim=-1)
    img_emb = scale * F.normalize(img_emb, p=2, dim=-1)

    img_dist = pairwise_distance_v2(proxies=proxies_emb, x=img_emb, squared=True)
    img_dist = img_dist * -1.0
    cross_entropy = nn.CrossEntropyLoss(reduction="mean")
    img_loss = cross_entropy(img_dist, gt_imgs)
    return img_loss


class FourierLoss(nn.Module):
    def __init__(
        self,
        use_l1_loss: bool = True,
        # num_multimodal_modalities: int = 1,  # set to 1 for vanilla MAE, 6 for channel-agnostic MAE
    ) -> None:
        """
        Recursion Pharmaceuticals 2024
        Fourier transform loss is only sound when using L1 or L2 loss to compare the frequency domains
        between the images / their radial histograms.

        We will always set `reduction="none"` and enforce that the computation of any reductions from the
        output of this loss be managed by the model under question.
        """
        super().__init__()
        self.loss = nn.L1Loss(reduction="none") if use_l1_loss else nn.MSELoss(reduction="none")
        # self.num_modalities = num_multimodal_modalities

    def forward(self, input: torch.Tensor, target: torch.Tensor, num_channels: int) -> torch.Tensor:
        # input = reconstructed image, target = original image
        # flattened images from MAE are (B, H*W, C), so, here we convert to B x C x H x W (note we assume H == W)
        flattened_images = len(input.shape) == len(target.shape) == 3
        if flattened_images:
            B, H_W, C = input.shape
            H_W = H_W // num_channels  ## self.num_modalities
            four_d_shape = (B, -1, int(H_W**0.5), int(H_W**0.5))  ## (B, C, H, W)

            input = input.view(*four_d_shape)
            target = target.view(*four_d_shape)
        else:
            B, C, h, w = input.shape
            H_W = h * w

        if len(input.shape) != len(target.shape) != 4:
            raise ValueError(f"Invalid input shape: got {input.shape} and {target.shape}.")

        fft_reconstructed = torch.fft.fft2(input)
        fft_original = torch.fft.fft2(target)

        magnitude_reconstructed = torch.abs(fft_reconstructed)
        magnitude_original = torch.abs(fft_original)

        loss_tensor: torch.Tensor = self.loss(magnitude_reconstructed, magnitude_original)

        # if (
        #     flattened_images and not self.num_bins
        # ):  # then output loss should be reshaped
        if flattened_images:
            loss_tensor = loss_tensor.reshape(B, -1, C)

        return loss_tensor


def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return -loss.mean()


def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def setup_ddp():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    batch_size = 1  # number of pairs per GPU
    feature_dim = 4

    # create synthetic features for 2B_local samples
    feats = torch.randn(batch_size * 2, feature_dim, device=device)

    loss_fn = SimCLRContrastiveLoss(temperature=0.1)
    loss = loss_fn(feats)


if __name__ == "__main__":
    # torch.set_printoptions(precision=2, sci_mode=False)
    # main()
    import torch
    import torch.nn.functional as F

    # Assuming the custom_cross_entropy_loss function is defined as above

    # Example inputs
    batch_size = 4
    num_classes = 10
    special_classes = [8, 9]  # Special class indices
    weighted_for_rare_classes_loss = 2.0  # Weight for special classes

    # Simulated model predictions (logits)
    predictions = torch.tensor(
        [
            [2.0, 1.0, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 1.5, 1.8],  # Normal class sample
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 2.0, 1.0],  # Special class sample
            [1.0, 2.0, 1.5, 0.5, 0.3, 0.2, 0.1, 0.4, 0.8, 0.9],  # Normal class sample
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0],  # Special class sample
        ],
        dtype=torch.float32,
    )

    # True labels
    targets = torch.tensor([2, 8, 3, 9], dtype=torch.long)  # 2, 3 are normal; 8, 9 are special

    # Calculate loss
    loss = custom_cross_entropy_loss(predictions, targets, special_classes, weighted_for_rare_classes_loss)
    print(f"Loss: {loss.item()}")
