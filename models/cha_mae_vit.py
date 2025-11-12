import math
from functools import partial
from typing import Optional

from torch import Tensor
import torch
import torch.nn as nn
from einops import rearrange, repeat
import time
import numpy as np
import random
from collections import OrderedDict
import timm

from .vit import Block
from .model_utils import trunc_normal_, maybe_flatten_images, MAB
from .mae_modules import ChAMAEViTDecoder, CAMAEDecoder
from .loss_func import compute_proxy_loss, FourierLoss, ortho_proj_loss_fn_v2


class PatchEmbedPerChannel(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        max_in_channels: int = 3,
        embed_dim: int = 768,
        use_channel_tokens: bool = True,
        channel_tokens_init: str = "orthogonal",
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * max_in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            1,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

        if use_channel_tokens:
            self.channel_tokens = nn.parameter.Parameter(torch.zeros(1, embed_dim, max_in_channels, 1, 1))
            if channel_tokens_init == "orthogonal":
                orthogonal_tensor = torch.empty(embed_dim, max_in_channels)
                nn.init.orthogonal_(orthogonal_tensor)  # produces orthogonal columns
                with torch.no_grad():
                    self.channel_tokens[:, :, :, 0, 0].copy_(orthogonal_tensor)
            elif channel_tokens_init == "random":
                trunc_normal_(self.channel_tokens, std=0.02)
            elif channel_tokens_init == "zero":
                trunc_normal_(self.channel_tokens, std=0.02) / 1000  ## close to zero
                # pass  ## already initialized to zero
            else:
                raise ValueError(f"Unknown channel_tokens_init: {channel_tokens_init}")
        else:
            self.channel_tokens = None

    def forward(
        self, x: Tensor, channel_ids_list: list[list[int]] | None, valid_channel_masks: Optional[Tensor] = None
    ):  ## return x, channel_token_patches
        """
        channel_ids: list of `batch_size` elements, each indicates channels of the img.  E.g., [[3,  5], [2]]
        valid_channel_masks: Attention mask (bool) with False at the end to indicate channel padding, e.g., [[True, True, False], [True, False, False]]
        """
        REGULAR_CASE = channel_ids_list is None and valid_channel_masks is None
        SAME_SUBSET_CHANNELS_FOR_ALL_IMG = channel_ids_list is not None and valid_channel_masks is None
        DIFFERENT_CHANNELS_FOR_EACH_IMG = channel_ids_list is not None and valid_channel_masks is not None

        device = x.device

        ## get channel tokens for this batch
        if self.channel_tokens is not None:
            if REGULAR_CASE:  ## Assume all images in the batch have the same channels, no masks.
                channel_tokens = self.channel_tokens
            elif SAME_SUBSET_CHANNELS_FOR_ALL_IMG:  ## E.g., each img has 8 channels, but only 5 channels are used for each image in the batch
                channel_ids = channel_ids_list[0]  # type: ignore
                channel_ids_tensor = torch.tensor(channel_ids, dtype=torch.long, device=device)
                channel_tokens = torch.index_select(self.channel_tokens, dim=2, index=channel_ids_tensor)
            elif DIFFERENT_CHANNELS_FOR_EACH_IMG:  ## General case, e.g., first image has 3 channels, second image has 5 channels, etc.
                ## get corresponding channel tokens for each image in the batch
                # 1. Flatten all indices and group size
                flat_idxs = [i for group in channel_ids_list for i in group]  # type: ignore
                flat_idxs_tensor = torch.tensor(flat_idxs, dtype=torch.long, device=device)
                group_sizes = [len(group) for group in channel_ids_list]  # type: ignore

                # 2. Gather once along the channel token's dim (dim=2)
                #    result shape = [B, d, sum(group_sizes), 1, 1]
                selected_flat = torch.index_select(self.channel_tokens, dim=2, index=flat_idxs_tensor)

                # 3. Split
                channel_tokens = list(torch.split(selected_flat, group_sizes, dim=2))

                # 4. padding to make channel_tokens the same size
                max_num_channels = max(group_sizes)
                dim = self.embed_dim
                channel_tokens = [
                    torch.cat([ct, torch.zeros(1, dim, max_num_channels - ct.shape[2], 1, 1, device=device)], dim=2) for ct in channel_tokens
                ]
                channel_tokens = torch.cat(channel_tokens, dim=0)  # B Cout Cin 1 1
            else:
                raise ValueError(f"Unknown case: channel_ids_list={channel_ids_list}, valid_channel_masks={valid_channel_masks}")
        else:
            channel_tokens = None

        # shared projection layer across channels
        x = self.proj(x.unsqueeze(1))  # B Cout Cin H W

        return x, channel_tokens  ### x: B Cout Cin H W, channel_tokens: 1 Cout Cin 1 1


class ChAMAEViT(nn.Module):
    """
    An implementation of NeurIPS'25 paper: ChA-MAEViT - Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning
    Paper: https://arxiv.org/pdf/2503.19331
    """

    def __init__(
        self,
        img_size=[224, 224],
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        attention_cls="regular",
        encoder_pooling="cls_token",
        diverse_patch_token_weight=0.0,
        diverse_patch_token_gamma_s=0.0,
        diverse_patch_token_gamma_d=0.0,
        diverse_channel_token_weight=0.0,
        diverse_channel_token_temperature=0.11,
        proxy_loss_lambda=1.0,
        proxy_temperature=0.11,
        proxy_orthogonal_init=True,
        num_proxies=0,
        use_proxy_head=False,
        cross_entropy_lambda=0.0,
        cross_entropy_temperature=10.0,
        mae_lambda=0.0,
        mae_loss_norm: Optional[str] = None,
        mae_recon_fourier_lambda: float = 0.0,
        mask_recon_fourier_loss: bool = True,
        num_memory_tokens: int = 0,
        training_sample=None,
        init_values: float | None = None,
        use_channel_tokens: bool = True,
        channel_tokens_init: str = "orthogonal",
        use_self_image_norm: bool = False,
        decoder: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = self.out_dim = embed_dim
        self.max_in_channels = in_chans
        self.training_sample = training_sample.upper() if training_sample is not None else None
        self.proxy_orthogonal_init = proxy_orthogonal_init
        self.patch_size = patch_size

        self.use_self_image_norm = use_self_image_norm
        if use_self_image_norm or mae_loss_norm == "instance_norm":
            self.image_norm = nn.InstanceNorm2d(None, affine=False, track_running_stats=False, eps=1e-5)
        self.patch_embed = PatchEmbedPerChannel(
            img_size=img_size,
            patch_size=patch_size,
            max_in_channels=self.max_in_channels,
            embed_dim=embed_dim,
            use_channel_tokens=use_channel_tokens,
            channel_tokens_init=channel_tokens_init,
        )
        self.num_patches_per_channel = self.patch_embed.num_patches // self.max_in_channels
        self.num_heads = num_heads
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.encoder_pooling = encoder_pooling
        if "attn_pooling" in self.encoder_pooling:
            self.pooling_query = nn.Parameter(torch.randn(1, embed_dim))
            trunc_normal_(self.pooling_query, std=0.02)
            self.attn_pooling = MAB(dim_Q=embed_dim, dim_K=embed_dim, dim_V=embed_dim, num_heads=num_heads)

        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.zeros(1, num_memory_tokens, embed_dim))

        self.num_extra_tokens = 1 + num_memory_tokens  # cls token + memory tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_per_channel + self.num_extra_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    attention_cls=attention_cls,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        ############ Define Losses ############
        self.use_proxy_loss = proxy_loss_lambda > 0
        self.proxy_loss_lambda = proxy_loss_lambda

        self.use_cross_entropy_loss = cross_entropy_lambda > 0
        self.cross_entropy_lambda = cross_entropy_lambda

        self.use_mae_loss = mae_lambda > 0
        self.mae_lambda = mae_lambda

        ## Diverse regularization losses (DiChaViT)
        self.diverse_patch_token_weight = diverse_patch_token_weight
        self.diverse_patch_token_gamma_s = diverse_patch_token_gamma_s
        self.diverse_patch_token_gamma_d = diverse_patch_token_gamma_d
        self.diverse_channel_token_weight = diverse_channel_token_weight

        if self.diverse_channel_token_weight > 0:
            self.channel_token_proxies = torch.nn.Parameter((torch.randn(in_chans, embed_dim) / 8))
            nn.init.orthogonal_(self.channel_token_proxies)

            self.proxy_diverse_channel_scale = np.sqrt(1.0 / diverse_channel_token_temperature)

        ## Cross entropy loss
        if self.use_cross_entropy_loss:
            self.cross_entropy_lambda = cross_entropy_lambda
            self.cross_entropy_temperature = cross_entropy_temperature
            self.compute_cross_entropy_loss = nn.CrossEntropyLoss()
            # Classifier head
            # self.cross_entropy_head = nn.Linear(embed_dim, num_classes)
            self.classifier_head = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, num_classes))

        ## Proxy loss
        self.proxy_scale = np.sqrt(1.0 / proxy_temperature)
        if self.use_proxy_loss:
            total_proxies = num_proxies

            if use_proxy_head:
                self.proxy_head = nn.Linear(embed_dim, embed_dim)
            else:
                self.proxy_head = nn.Identity()

            self.output_proxies = torch.nn.Parameter((torch.randn(total_proxies, embed_dim) / 8))
            if self.proxy_orthogonal_init:
                nn.init.orthogonal_(self.output_proxies)  ## initlaize orthogonally

        ##### MAE Loss
        ## dcp
        if decoder is not None:
            self.dcp_mask_ratio = decoder["dcp_mask_ratio"]
            self.dcp_p_patch = decoder["dcp_p_patch"]
            self.dcp_p_channel = decoder["dcp_p_channel"]
            self.dcp_at_least_mask_one_channel = decoder["dcp_at_least_mask_one_channel"]
        if self.use_mae_loss:
            self.mae_loss_norm = mae_loss_norm
            self.reconstruct_loss_fn = nn.MSELoss(reduction="none")
            self.recon_fourier_loss_fn = FourierLoss()
            self.mae_recon_fourier_lambda = mae_recon_fourier_lambda
            self.mask_recon_fourier_loss = mask_recon_fourier_loss

            self.decoder_dim = decoder.embed_dim if decoder.embed_dim is not None else embed_dim
            self.decoder_type = decoder["decoder_type"]
            if decoder.decoder_type == "chamaevit_decoder":
                self.decoder = ChAMAEViTDecoder(
                    depth=decoder["depth"],
                    embed_dim=self.decoder_dim,
                    mlp_ratio=decoder["mlp_ratio"],
                    norm_layer=partial(nn.LayerNorm, eps=decoder["norm_layer"]["eps"]),
                    num_heads=decoder["num_heads"],
                    qkv_bias=decoder["qkv_bias"],
                    attention_cls=attention_cls,
                )
            else:
                raise ValueError(f"Unknown decoder type: {decoder.decoder_type}")

            # projection layer between the encoder and decoder
            self.encoder_decoder_proj = nn.Linear(embed_dim, self.decoder_dim, bias=True)

            # linear layer from decoder embedding to input dims
            decoder_out_dim = patch_size**2
            self.decoder_pred = nn.Linear(
                self.decoder_dim,
                decoder_out_dim,
                bias=True,
            )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if num_memory_tokens > 0:
            trunc_normal_(self.memory_tokens, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, total_tokens, dim, w, h, nc):
        # number of auxilary dimensions before the patches
        num_extra_tokens = self.num_extra_tokens

        n_patches = total_tokens - num_extra_tokens
        n_patches_per_channel = self.pos_embed.shape[1] - num_extra_tokens
        if n_patches == n_patches_per_channel and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, :num_extra_tokens]
        patch_pos_embed = self.pos_embed[:, num_extra_tokens:]

        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(n_patches_per_channel)), int(math.sqrt(n_patches_per_channel)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(n_patches_per_channel), h0 / math.sqrt(n_patches_per_channel)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, 1, -1, dim)

        # create copies of the positional embeddings for each channel
        patch_pos_embed = patch_pos_embed.expand(1, nc, -1, dim).reshape(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def generate_patch_masks_from_channel_masks(self, valid_channel_masks: Tensor):
        """
        Generate attention masks (patch level) for the channel masks
        + `valid_channel_masks`: B, C. Attention mask (bool) with False at the end to indicate channel padding. Example input [[True, True, False], [True, False, False]]

        + output (`patch_masks`): Expand each channel mask into `num_patches_per_channel` patches, and add a mask for the cls token.
        E.g., Assume `num_patches_per_channel`=2, output for the example input above would be
                    [[True, True, True, True, True, False False], [True, True, True, False, False, False, False]]
        """
        patch_masks = repeat(valid_channel_masks, "b j -> b (j c)", c=self.num_patches_per_channel)

        ## add masks for the cls token and memory tokens if any
        extra_token_masks = torch.ones((patch_masks.shape[0], self.num_extra_tokens), dtype=patch_masks.dtype, device=patch_masks.device)
        patch_masks = torch.cat([extra_token_masks, patch_masks], dim=1)
        B, L = patch_masks.shape
        patch_masks = patch_masks.view(B, 1, 1, L).bool()
        return patch_masks

    def generate_patch_masks_from_patch_sampling(self, patch_masks: Tensor):
        ## add masks for the cls token and memory tokens if any
        extra_token_masks = torch.ones((patch_masks.shape[0], self.num_extra_tokens), dtype=patch_masks.dtype, device=patch_masks.device)
        patch_masks = torch.cat([extra_token_masks, patch_masks], dim=1)
        B, L = patch_masks.shape
        patch_masks = patch_masks.view(B, 1, 1, L).bool()
        return patch_masks

    def compute_patch_token_loss(self, x: Tensor, mask: Tensor | None) -> Tensor:
        """
        compute Token Diversification Loss in DiChaViT paper
        x: Tensor, shape (b d c h w), where, h*w = n_patches_per_channel
        mask: Tensor, shape (b, L), where L = c * h * w
        """
        n_patches_per_channel = x.shape[3] * x.shape[4]
        token_labels = torch.arange(x.shape[2]).repeat_interleave(n_patches_per_channel).to(x.device)
        x_reshaped = rearrange(x, "b d c h w -> b (c h w) d").clone()
        patch_token_loss = ortho_proj_loss_fn_v2(
            x_reshaped,
            labels=token_labels,
            gamma_s=self.diverse_patch_token_gamma_s,
            gamma_d=self.diverse_patch_token_gamma_d,
            reverse_pos_pairs=True,
            use_square=False,
            patch_mask=mask,
        )
        return patch_token_loss

    def compute_channel_token_loss(self, channel_ids_unique: Tensor) -> Tensor:
        """
        channel_tokens:  Cin, embed_dim=Cout
        compute Channel Diversification Loss in DiChaViT paper
        """
        ## create one hot ground truth for the channel
        channel_tokens_in_batch = self.patch_embed.channel_tokens[0, :, channel_ids_unique, 0, 0].T  # Cin, Cout

        proxies = self.channel_token_proxies
        channel_token_loss = compute_proxy_loss(proxies, channel_tokens_in_batch, channel_ids_unique, scale=self.proxy_diverse_channel_scale)

        return channel_token_loss

    def prepare_tokens(
        self,
        x: Tensor,
        channel_ids_list: list[list[int]] | None,
        valid_channel_masks: Optional[Tensor],
        total_tokens: int,
        tokens_to_keep: Optional[Tensor],
        channel_tokens_in_batch: list[list[int]],
        visibles_mask: Optional[Tensor] = None,
    ):
        B, nc, w, h = x.shape

        ## patchify and embed
        x, channel_tokens = self.patch_embed(x, channel_ids_list, valid_channel_masks)
        ## x: B Cout Cin H W, channel_tokens: 1 Cout Cin 1 1 or None

        ############################################################################################################
        #### Compute diverse loss
        diverse_losses = {
            "diverse_patchtoken_loss": torch.tensor(0.0, device=x.device),
            "diverse_channeltoken_loss": torch.tensor(0.0, device=x.device),
        }

        ## patch token loss
        if self.diverse_patch_token_weight > 0:
            diverse_losses["diverse_patchtoken_loss"] = self.compute_patch_token_loss(x, mask=visibles_mask)

        # ## channel token loss
        if self.diverse_channel_token_weight > 0:
            ## get set of all channel_ids_list:
            channel_ids_unique = list(set([i for group in channel_tokens_in_batch for i in group]))  # type: ignore
            channel_ids_unique = torch.tensor(channel_ids_unique, dtype=torch.long, device=x.device)
            diverse_losses["diverse_channeltoken_loss"] = self.compute_channel_token_loss(channel_ids_unique)
        #############################################################################################################

        # channel specific offsets
        if channel_tokens is not None:
            x += channel_tokens  # B Cout Cin H W

        # preparing the output sequence
        x = x.flatten(2)  # B Cout CinHW
        x = x.transpose(1, 2)  # B CinHW Cout

        ## add the [CLS] token and memory tokens if any, to the embed patch tokens
        extra_tokens = self.cls_token.expand(B, -1, -1)
        if self.num_memory_tokens > 0:
            memory_tokens = self.memory_tokens.expand(B, -1, -1)
            extra_tokens = torch.cat((extra_tokens, memory_tokens), dim=1)  # B, 1+M, D
        x = torch.cat((extra_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(total_tokens, self.embed_dim, w, h, nc)

        ## mask out some patches, resulting in fewer patches than x
        if tokens_to_keep is not None:
            tokens_to_keep = tokens_to_keep.unsqueeze(-1).repeat(1, 1, self.embed_dim)
            x_masked = torch.gather(x[:, self.num_extra_tokens :, :], dim=1, index=tokens_to_keep)
            ## add class token (and potentially meta tokens) back
            x = torch.cat([x[:, : self.num_extra_tokens, :], x_masked], dim=1)

        return self.pos_drop(x), channel_tokens, diverse_losses

    def _dcp_channel_masking(
        self,
        at_least_mask_one_channel: bool,
        batch_size: int,
        n_channels_origin: int,
        device: torch.device,
        valid_channel_masks: Tensor | None = None,
    ) -> Tensor:
        """
        Generates a random mask for channels.

        Args:
            at_least_mask_one_channel (bool): If True, ensures at least one channel is masked.
        Returns:
            Tensor: Boolean mask of shape (B, C) where True indicates a masked channel.
        """
        B, C = batch_size, n_channels_origin
        LARGE_VALUE = 1000000
        channel_masks = torch.zeros(B, C, dtype=torch.bool, device=device)

        # CASE 1: Each image in the batch may have a different number of valid channels
        if valid_channel_masks is not None:
            n_valid_channels = valid_channel_masks.sum(dim=1)

            ## For each image, sample a number 'k' of channels to mask
            rand_floats = torch.rand(B, device=device)
            nc_to_mask = (rand_floats * n_valid_channels.to(rand_floats.dtype)).to(torch.long)

            if at_least_mask_one_channel:
                nc_to_mask = torch.clamp(nc_to_mask, min=1)

            scores = torch.rand(B, C, device=device)
            scores.masked_fill_(~valid_channel_masks, LARGE_VALUE)
            sorted_indices = torch.argsort(scores, dim=1)

            # Create a mask to select the top 'k' indices for each row
            arange_mask = torch.arange(C, device=device).expand(B, -1)
            top_k_mask = arange_mask < nc_to_mask.unsqueeze(1)

            # Apply the top_k_mask to the final channel_masks
            channel_masks.scatter_(dim=1, index=sorted_indices, src=top_k_mask)

        # CASE 2: All images have the same number of channels
        else:
            nc_to_mask = torch.randint(1 if at_least_mask_one_channel else 0, C, (1,), device=device).item()

            if nc_to_mask > 0:
                indices = torch.randperm(C, device=device)[:nc_to_mask]
                channel_masks[:, indices] = True

        return channel_masks

    def _dcp_patch_masking(
        self,
        batch_size: int,
        n_patches: int,
        n_channels_origin: int,
        n_patches_per_channel: int,
        mask_ratio: float,
        device: torch.device,
    ) -> tuple[Tensor, int]:
        """
        Generates a standard patch mask based on a fixed mask ratio per channel.

        Returns:
            tuple[Tensor, int]: A tuple containing:
                - A noise tensor of shape (B, L) where 0 is keep and 1 is mask.
                - The number of patches kept per channel.
        """
        B, L, C, P = batch_size, n_patches, n_channels_origin, n_patches_per_channel

        # Calculate how many patches to keep per channel
        keep_per_channel = int((1 - mask_ratio) * P)

        # Generate random noise for each patch within each channel
        noise_per_channel = torch.rand(B, C, P, device=device)

        # Find the indices of the patches to keep (those with the smallest noise)
        keep_idxs_per_channel = torch.argsort(noise_per_channel, dim=-1)[:, :, :keep_per_channel]

        # Add channel offsets to get indices in the flattened (L) dimension
        channel_offsets = torch.arange(0, L, P, device=device).view(1, -1, 1)
        keep_idxs = keep_idxs_per_channel + channel_offsets
        keep_idxs = rearrange(keep_idxs, "b c p -> b (c p)")

        # Create the final noise tensor where 0 means keep (visible) and 1 means mask
        patch_level_masking = torch.ones(B, L, device=device, dtype=torch.bool)
        patch_level_masking.scatter_(dim=1, index=keep_idxs, value=0)

        return patch_level_masking, keep_per_channel

    def sample_dcp_strategy(self, p_patch: float = 0.5, p_channel: float = 0.5) -> str:
        """Determine the masking strategy based on probabilities."""
        rand_float = np.random.rand()
        if rand_float < p_patch:
            return "PATCH_MASKING_ONLY"
        elif rand_float < p_patch + p_channel:
            return "CHANNEL_MASKING_ONLY"
        else:
            return "BOTH_MASKING"

    def dynamic_channel_patch_sampling(
        self,
        at_least_mask_one_channel: bool,
        batch_size: int,
        n_patches: int,
        device: torch.device,
        n_patches_per_channel: int,
        n_channels_origin: int,
        mask_ratio: float,
        dcp_strategy: str,
        valid_channel_masks: Tensor | None = None,
    ):
        """ChA-MAEViT's Dynamic Channel and Patch Sampling"""
        B, L = batch_size, n_patches
        LARGE_VALUE_FOR_PADDING = 99
        VISIBLE_VALUE = 0  # small noise value for visible patches

        ## 1/ Determine sampling strategy
        assert dcp_strategy in ["PATCH_MASKING_ONLY", "CHANNEL_MASKING_ONLY", "BOTH_MASKING"]

        if dcp_strategy == "PATCH_MASKING_ONLY":
            patch_level_masks, keep_per_channel = self._dcp_patch_masking(B, L, n_channels_origin, n_patches_per_channel, mask_ratio, device)
            final_noise = patch_level_masks.float()  ##`patch_level_masks`:   0 -> visible (may contain padding channels, handle later), 1 -> masked
            channels_kept = torch.ones(B, n_channels_origin, dtype=torch.bool, device=device)  ## 1: keep, may contain padding channels
        elif dcp_strategy == "CHANNEL_MASKING_ONLY":
            channel_masks = self._dcp_channel_masking(
                at_least_mask_one_channel, B, n_channels_origin, device, valid_channel_masks
            )  ## 0: visible (and may contain padding channels), 1: masked
            channel_level_masks = repeat(channel_masks, "b c -> b (c n)", n=n_patches_per_channel)
            final_noise = channel_level_masks.float()
            channels_kept = ~channel_masks  ## may contain padding channels, as we haven't handled padding channels yet.
        else:  # BOTH_MASKING
            channel_masks = self._dcp_channel_masking(
                at_least_mask_one_channel, B, n_channels_origin, device, valid_channel_masks
            )  ## 0: visible (and may contain padding channels), 1: masked
            channel_level_masks = repeat(channel_masks, "b c -> b (c n)", n=n_patches_per_channel)

            patch_level_masks, keep_per_channel = self._dcp_patch_masking(
                B, L, n_channels_origin, n_patches_per_channel, mask_ratio, device
            )  ##`patch_noise`:   0 -> visible (may contain padding channels), 1 -> masked
            final_noise = patch_level_masks.float() + channel_level_masks.float()
            channels_kept = ~channel_masks  ## 1: keep, may contain padding channels

        patch_padding_masks = None
        ## Handle padding channels: if we have channel padding, make sure we do not sample from the padded channels
        if valid_channel_masks is not None:
            patch_padding_masks = repeat(valid_channel_masks, "b c -> b (c n)", n=n_patches_per_channel)  ## 0: padding
            final_noise.masked_fill_(~patch_padding_masks, LARGE_VALUE_FOR_PADDING)
            channels_kept = channels_kept & valid_channel_masks  ## channel kept to train (not masked and not padded)

        ####  Generate final indices and loss mask from the combined noise
        shuffled_tokens = torch.argsort(final_noise, dim=1)
        ind_restore = torch.argsort(shuffled_tokens, dim=1)

        if valid_channel_masks is not None:
            n_channels_kept_per_sample = channels_kept.sum(dim=1)
            if dcp_strategy == "CHANNEL_MASKING_ONLY":
                n_patches_keep_list = n_channels_kept_per_sample * n_patches_per_channel
            else:
                n_patches_keep_list = n_channels_kept_per_sample * keep_per_channel

            max_patches_keep = n_patches_keep_list.max().item()
            tokens_to_keep = torch.zeros(B, max_patches_keep, dtype=torch.long, device=device)
            mae_patch_masks = torch.zeros(B, max_patches_keep, dtype=torch.bool, device=device)

            for i, n_pk_i in enumerate(n_patches_keep_list):
                tokens_to_keep[i, :n_pk_i] = shuffled_tokens[i, :n_pk_i]
                mae_patch_masks[i, :n_pk_i] = True
            n_patches_keep = max_patches_keep
        else:
            n_channels_kept = channels_kept[0].sum().item()
            # print(f"dcp_strategy={dcp_strategy}, n_channels_kept: {n_channels_kept} out of {n_channels_origin}")
            if dcp_strategy == "CHANNEL_MASKING_ONLY":
                n_patches_keep = n_channels_kept * n_patches_per_channel
            else:
                n_patches_keep = n_channels_kept * keep_per_channel
            tokens_to_keep = shuffled_tokens[:, :n_patches_keep]
            mae_patch_masks = None

        mask = (LARGE_VALUE_FOR_PADDING > final_noise) & (final_noise > VISIBLE_VALUE)
        visibles = final_noise == VISIBLE_VALUE
        res = {
            "tokens_to_keep": tokens_to_keep,  ## token indices kept to train (visible patches)
            "channels_sampled": channels_kept,  ## channels kept to train  (1: visible channels)
            "mask": mask,  ## binary mask for computing reconstruction loss (1: masked, 0: visible & padded if any)
            "visibles": visibles,  ## binary mask for visible patches (1: visible, 0: masked & padded if any)
            "ind_restore": ind_restore,  ## indices to restore original order
            "mae_patch_masks": mae_patch_masks,  ## [optional] used when having padded patches: binary mask for attention calculation (1: visible patches, 0: padding)
        }
        return res

    def channel_dropout(
        self, x, channel_sample: str, channel_ids_list: list[list[int]] | None, valid_channel_masks: Optional[Tensor]
    ) -> dict[str, Tensor | list]:
        """
        Hiararchical Channel Sampling (HCS) strategy in ChannelViT paper
        """

        REGULAR_CASE = channel_ids_list is None and valid_channel_masks is None
        DIFFERENT_CHANNELS_FOR_EACH_IMG = channel_ids_list is not None and valid_channel_masks is not None

        res = {"x": None, "channel_ids_list": None, "channel_masks": None}

        Cin = x.shape[1]
        if REGULAR_CASE:  ##  all images have the same channels
            if channel_sample == "HCS":
                Cin_new = random.randint(1, Cin)
                channel_indices = torch.tensor(random.sample(range(Cin), k=Cin_new), device=x.device, dtype=torch.long)
                res["x"] = x[:, channel_indices]
                return res
            else:
                raise ValueError(f"Unknown channel sampling method: {channel_sample}")
        elif DIFFERENT_CHANNELS_FOR_EACH_IMG:  ## General Case, each image may have different channels
            #### E.g., first image has 3 channels, second image has 5 channels, etc.
            ## get Cin_new for each image in the batch
            channel_ids_list_new = []
            channel_indices_list_new = []
            for channel_ids in channel_ids_list:
                if channel_sample.startswith("HCS"):
                    Cin = len(channel_ids)
                    Cin_new = random.randint(1, Cin)
                    channel_indices_new = torch.tensor(random.sample(range(Cin), k=Cin_new), device=x.device, dtype=torch.long)
                    channel_indices_list_new.append(channel_indices_new)
                    channel_ids_new = [channel_ids[i] for i in channel_indices_new.tolist()]
                    channel_ids_list_new.append(channel_ids_new)

            ## get the new images and masks
            max_Cin_new = max([len(channel_ids) for channel_ids in channel_ids_list_new])
            images_new = torch.zeros((x.shape[0], max_Cin_new, x.shape[2], x.shape[3]), device=x.device)
            channel_masks_new = torch.zeros((x.shape[0], max_Cin_new), dtype=torch.bool, device=x.device)

            for i, channel_ids in enumerate(channel_ids_list_new):
                channel_indices = channel_indices_list_new[i]
                images_new[i, : len(channel_indices), :, :] = x[i, channel_indices, :, :]
                channel_masks_new[i, : len(channel_ids)] = True

            res = {
                "x": images_new,
                "channel_ids_list": channel_ids_list_new,
                "channel_masks": channel_masks_new,
            }
            return res
        else:
            raise ValueError(f"Unknown case: channel_ids_list={channel_ids_list}, valid_channel_masks={valid_channel_masks}")

    def _norm_pix_loss(self, target: Tensor):
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5
        return target

    def compute_mae_loss(
        self,
        reconstruction: Tensor,
        img: Tensor,
        mask: Tensor,
    ) -> dict[str, Tensor]:
        """Computes MAE loss"""
        mae_loss_dict = {}
        num_channels = img.shape[1]
        if self.mae_loss_norm == "norm_pix_loss":
            ## flat first, then norm per patch (based on MAE original paper)
            target_flattened = maybe_flatten_images(img, patch_size=self.patch_size, channel_agnostic=True)
            target_flattened = self._norm_pix_loss(target_flattened)
        elif self.mae_loss_norm == "instance_norm":  ## norm first, then flat (based on CAMAE paper)
            img = self.image_norm(img)
            target_flattened = maybe_flatten_images(img, patch_size=self.patch_size, channel_agnostic=True)
        elif self.mae_loss_norm is None:
            target_flattened = maybe_flatten_images(img, patch_size=self.patch_size, channel_agnostic=True)

        ## img: b c h p1 w p2
        ## target_flattened: b (c p1 p2) (h w), where p1=p2=#patches, h=w=patch_size. h*w is the #pixels in a patch (ie, patch dim)
        ## e.g., for jumpcp, target_flattened shape = torch.Size([128, 8*14*14, 16*16])
        ## Should be with MSE or MAE (L1) with reduction='none'
        loss = self.reconstruct_loss_fn(reconstruction, target_flattened)
        loss = loss.mean(dim=-1)  # average over embedding dim -> mean loss per patch (N,L)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on masked patches only
        mae_loss_dict["mae_img_loss"] = loss
        # compute fourier loss
        if self.mae_recon_fourier_lambda > 0:
            floss = self.recon_fourier_loss_fn(reconstruction, target_flattened, num_channels)  ## B, L, C
            if not self.mask_recon_fourier_loss:
                floss = floss.mean()
            else:
                floss = floss.mean(dim=-1)
                floss = (floss * mask).sum() / mask.sum()
        else:
            floss = torch.tensor(0.0, device=img.device)
        mae_loss_dict["mae_fourier_loss"] = floss

        return mae_loss_dict

    def forward_decoder(
        self,
        x_latent: Tensor,
        position_embeddings: Tensor,
        ind_restore: Tensor,
        channel_token_patches: Tensor | None = None,
        full_patch_masks: Tensor | None = None,
    ):
        num_extra_tokens = self.num_extra_tokens
        decoder_latent = self.encoder_decoder_proj(x_latent)

        if self.decoder_type == "chamaevit_decoder":
            decoder_tokens = self.decoder.forward_masked(
                decoder_latent,
                ind_restore=ind_restore,
                pos_embeddings=position_embeddings,
                channel_token_patches=channel_token_patches,
                num_extra_tokens=num_extra_tokens,
                full_patch_masks=full_patch_masks,
            )  # decoder.embed_dim
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")

        predicted_reconstruction = self.decoder_pred(decoder_tokens)  # linear projection to input
        reconstruction = predicted_reconstruction[:, num_extra_tokens:, :]  # drop class token and memory tokens

        return reconstruction

    def forward(
        self,
        x: Tensor,
        channel_ids_list: list[list[int]] | None = None,
        valid_channel_masks: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
    ):

        if self.use_self_image_norm:
            x = self.image_norm(x)

        x_origin = x.clone()  ## for reconstruction loss if any
        b, c, w, h = x.shape

        n_patches = self.num_patches_per_channel * c
        total_tokens = n_patches + self.num_extra_tokens

        ### channel dropout, used in ChannelViT
        if self.training and self.training_sample in ["HCS", "HCS_SYMMETRIC"]:
            dropout_res = self.channel_dropout(
                x, channel_sample=self.training_sample, channel_ids_list=channel_ids_list, valid_channel_masks=valid_channel_masks
            )
            x, channel_ids_list, valid_channel_masks = dropout_res["x"], dropout_res["channel_ids_list"], dropout_res["channel_masks"]

        ### generate patch masks from channel masks if any
        if valid_channel_masks is not None:
            full_patch_masks = self.generate_patch_masks_from_channel_masks(valid_channel_masks)
        else:
            full_patch_masks = None

        ### sampling for MAEs
        sample_strategy = self.training_sample
        tokens_to_keep = sample_res = dcp_strategy = None
        if self.training and sample_strategy == "DCP":  ## dynamic channel and patch sampling, used in ChA-MAEViT
            dcp_strategy = self.sample_dcp_strategy(p_patch=self.dcp_p_patch, p_channel=self.dcp_p_channel)
            sample_res = self.dynamic_channel_patch_sampling(
                at_least_mask_one_channel=self.dcp_at_least_mask_one_channel,
                batch_size=b,
                n_patches=c * self.num_patches_per_channel,
                device=x.device,
                n_patches_per_channel=self.num_patches_per_channel,
                n_channels_origin=c,
                mask_ratio=self.dcp_mask_ratio,
                valid_channel_masks=valid_channel_masks,
                dcp_strategy=dcp_strategy,
            )
            tokens_to_keep = sample_res["tokens_to_keep"]
            mae_patch_masks = sample_res["mae_patch_masks"]

        ############ patch projection + add pos emb + add cls token and memory tokens
        if channel_ids_list is None:
            if sample_res is not None:
                channel_ids = [list(filter(lambda c_idx: sample_res["channels_sampled"][0][c_idx], range(c)))]
            else:
                channel_ids = [list(range(c))]
        else:
            channel_ids = channel_ids_list
        x, channel_tokens, diverse_losses = self.prepare_tokens(
            x,
            channel_ids_list,
            valid_channel_masks,
            total_tokens=total_tokens,
            tokens_to_keep=tokens_to_keep,
            channel_tokens_in_batch=channel_ids,
            visibles_mask=sample_res["visibles"] if sample_res is not None else None,
        )
        forward_patch_masks = full_patch_masks
        ## if training, and each image has different channels
        if self.training and valid_channel_masks is not None:
            if sample_strategy == "DCP":
                ## if we do patch sampling, we have the patch masks for visible patches
                forward_patch_masks = self.generate_patch_masks_from_patch_sampling(mae_patch_masks)
                ## other cases, we use the full patch masks (if any)
        for blk in self.blocks:
            x = blk(x, mask=forward_patch_masks)
        x = self.norm(x)

        if self.encoder_pooling == "cls_token":
            out = x[:, 0].clone()
        elif self.encoder_pooling.startswith("cls_attn_pooling"):
            cls_token = x[:, 0].clone()
            patch_tokens = x[:, self.num_extra_tokens :]
            attn_pooling_out = self.attn_pooling(Q=self.pooling_query, X=patch_tokens).squeeze(1)  # b, d
            if self.encoder_pooling == "cls_attn_pooling_sigmoid":  ## use sigmoid to gate the cls token
                attn_pooling_out = torch.sigmoid(attn_pooling_out)
                out = cls_token * attn_pooling_out
            else:
                out = cls_token + attn_pooling_out
        else:
            raise ValueError(f"Unknown encoder pooling: {self.encoder_pooling}")

        ############## Pass into a decoder if any (e.g., for MAE reconstruction)
        if self.training and self.use_mae_loss:
            ### if all images have the same channels, and we do channel masking with no channels get masked, then no need to do reconstruction
            skip_decoder = valid_channel_masks is None and dcp_strategy == "CHANNEL_MASKING_ONLY" and all(sample_res["channels_sampled"][0])
            if not skip_decoder:
                if channel_tokens is not None:
                    channel_token_patches = repeat(channel_tokens, "b d c 1 1 -> b d c p", p=self.num_patches_per_channel)
                    channel_token_patches = rearrange(channel_token_patches, "b d c p -> b (c p) d")  ## b, c*p1*p2, d
                else:
                    channel_token_patches = None
                reconstruction = self.forward_decoder(
                    x_latent=x,
                    position_embeddings=self.interpolate_pos_encoding(total_tokens, self.embed_dim, w, h, c),
                    ind_restore=sample_res["ind_restore"],
                    channel_token_patches=channel_token_patches,
                    full_patch_masks=full_patch_masks,
                )
            else:
                pass
                # print("Skip decoder as no channels are masked!")

        ############## return predictions + losses if any
        if hasattr(self, "classifier_head"):
            ## for classification tasks, we return the logits
            out_logits = self.classifier_head(out)
            res = {"output": out_logits}
        else:
            res = {"output": out}

        ############## compute losses during training
        if self.training:
            ## compute proxy loss
            if self.use_proxy_loss:
                out_proxy = self.proxy_head(out)

                proxy_loss = compute_proxy_loss(
                    proxies=self.output_proxies,
                    img_emb=out_proxy,
                    gt_imgs=y,
                    scale=self.proxy_scale,
                )
                res["proxy_loss"] = proxy_loss
            else:
                res["proxy_loss"] = torch.tensor(0.0)

            ## compute cross entropy loss if any
            if self.use_cross_entropy_loss:
                ce_loss = self.compute_cross_entropy_loss(out_logits / self.cross_entropy_temperature, y)
                res["ce_loss"] = ce_loss
            else:
                res["ce_loss"] = torch.tensor(0.0)

            ## compute MAE loss
            if self.use_mae_loss and not skip_decoder:
                ######## reconstruct img
                ## reconstruction: (b, c * n_patches, patch_size**2)
                ## x_origin: (b, c, h, w),    (b, d, c, n_pathes)
                recon_loss_dict = self.compute_mae_loss(reconstruction, x_origin.float(), sample_res["mask"])
                lambda_f = self.mae_recon_fourier_lambda
                recon_loss_dict["mae_loss"] = (1 - lambda_f) * recon_loss_dict["mae_img_loss"] + lambda_f * recon_loss_dict["mae_fourier_loss"]
                res.update(recon_loss_dict)
            else:
                for k in ["mae_img_loss", "mae_fourier_loss", "mae_loss"]:
                    res[k] = torch.tensor(0.0)

            ## Diverse regularization losses
            res["diverse_channeltoken_loss"] = diverse_losses["diverse_channeltoken_loss"]
            res["diverse_patchtoken_loss"] = diverse_losses["diverse_patchtoken_loss"]

            ## final loss
            res["loss"] = (
                self.proxy_loss_lambda * res["proxy_loss"]
                + self.mae_lambda * res["mae_loss"]
                + self.diverse_patch_token_weight * res["diverse_patchtoken_loss"]
                + self.diverse_channel_token_weight * res["diverse_channeltoken_loss"]
                + self.cross_entropy_lambda * res["ce_loss"]
            )
        return res


def get_multi_channel_vit(model_size: str, patch_size=16, **kwargs):
    if model_size == "tiny":
        model = ChAMAEViT(
            patch_size=patch_size,
            embed_dim=192,
            num_heads=3,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )
    elif model_size == "small":
        model = ChAMAEViT(
            patch_size=patch_size,
            embed_dim=384,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )
    elif model_size == "base":
        model = ChAMAEViT(
            patch_size=patch_size,
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model name: {model_size}")
    return model
