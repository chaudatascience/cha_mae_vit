import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ViTSegmentationHead(nn.Module):
    def __init__(self, embed_dim: int, patch_size: int = 16, img_size: int = 384, num_classes: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_classes = num_classes

        self.h_patches = img_size // patch_size  # 384 / 16 = 24
        self.w_patches = img_size // patch_size  # 384 / 16 = 24

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1),  # 24x24 -> 48x48
            nn.GroupNorm(8, embed_dim // 2),
            nn.ReLU(),
        )

        self.conv_fusion1 = nn.Sequential(
            nn.Conv2d(embed_dim // 2 + embed_dim, embed_dim // 2, kernel_size=3, padding=1), nn.GroupNorm(8, embed_dim // 2), nn.ReLU()
        )

        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=4, stride=2, padding=1),  # 48x48 -> 96x96
            nn.GroupNorm(8, embed_dim // 4),
            nn.ReLU(),
        )
        self.conv_fusion2 = nn.Sequential(
            nn.Conv2d(embed_dim // 4 + embed_dim, embed_dim // 4, kernel_size=3, padding=1), nn.GroupNorm(8, embed_dim // 4), nn.ReLU()
        )

        # Decoder for 96x96 features
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=4, stride=2, padding=1),  # 96x96 -> 192x192
            nn.GroupNorm(8, embed_dim // 8),
            nn.ReLU(),
        )
        self.conv_fusion3 = nn.Sequential(
            nn.Conv2d(embed_dim // 8 + embed_dim, embed_dim // 8, kernel_size=3, padding=1), nn.GroupNorm(8, embed_dim // 8), nn.ReLU()
        )

        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim // 8, embed_dim // 16, kernel_size=4, stride=2, padding=1),  # 192x192 -> 384x384
            nn.GroupNorm(8, embed_dim // 16),
            nn.ReLU(),
        )
        self.conv_fusion4 = nn.Sequential(
            nn.Conv2d(embed_dim // 16 + embed_dim, embed_dim // 16, kernel_size=3, padding=1), nn.GroupNorm(8, embed_dim // 16), nn.ReLU()
        )

        # Final output convolution
        self.final_conv = nn.Conv2d(embed_dim // 16, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]):
        # Ensure features are a list and have the expected number of elements (e.g., 4)
        if not isinstance(features, list) or len(features) < 1:
            raise ValueError("Input 'features' must be a list of tensors from ViT layers.")

        # Helper to reshape and permute ViT outputs
        def _process_vit_feature(x_tensor: torch.Tensor, target_H: int, target_W: int):
            B, L, d = x_tensor.shape

            # Remove CLS token if present
            if L == (self.h_patches * self.w_patches) + 1:
                x_tensor = x_tensor[:, 1:]
                L = x_tensor.shape[1]  # Update L after removal

            if L != self.h_patches * self.w_patches:
                raise ValueError(
                    f"Feature sequence length ({L}) does not match expected "
                    f"spatial patches ({self.h_patches * self.w_patches}) after CLS token removal."
                )

            x_spatial = x_tensor.view(B, self.h_patches, self.w_patches, d)
            x_spatial = x_spatial.permute(0, 3, 1, 2)

            if x_spatial.shape[2] != target_H or x_spatial.shape[3] != target_W:
                x_spatial = F.interpolate(x_spatial, size=(target_H, target_W), mode="bilinear", align_corners=False)

            return x_spatial

        features_new = []
        for feat in features:
            new_feat = rearrange(feat, "b (c l) d -> b c l d", l=self.h_patches * self.w_patches).mean(dim=1)  ## (b, l, d)
            features_new.append(new_feat)
        features = features_new

        x = _process_vit_feature(features[-1], self.h_patches, self.w_patches)  # 24x24

        x = self.upconv1(x)

        if len(features) >= 2:
            skip1 = _process_vit_feature(features[-2], self.h_patches, self.w_patches)  # 24x24
            skip1_upsampled = F.interpolate(skip1, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip1_upsampled], dim=1)  # Concatenate channels
            x = self.conv_fusion1(x)  # Reduce channels and apply convolution

        x = self.upconv2(x)

        if len(features) >= 3:
            skip2 = _process_vit_feature(features[-3], self.h_patches, self.w_patches)  # 24x24
            skip2_upsampled = F.interpolate(skip2, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip2_upsampled], dim=1)
            x = self.conv_fusion2(x)

        x = self.upconv3(x)

        if len(features) >= 4:
            skip3 = _process_vit_feature(features[-4], self.h_patches, self.w_patches)  # 24x24
            skip3_upsampled = F.interpolate(skip3, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip3_upsampled], dim=1)
            x = self.conv_fusion3(x)

        x = self.upconv4(x)

        if len(features) >= 4:  # This might be where you add the earliest features
            pass  # Already handled above as skip3, if you have exactly 4.

        # Final output convolution
        x = self.final_conv(x)

        # Apply sigmoid for binary segmentation
        return torch.sigmoid(x).squeeze(1)
