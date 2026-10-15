"""U-Net, ported from get_unet and conv2d_block in the original
src/train/*/*/models.py (docs/paper_notes.md §6)."""

import torch
from torch import nn

# Keras BatchNorm defaults (PyTorch's are eps=1e-5, momentum=0.1). Keras's
# momentum=0.99 is PyTorch's momentum=0.01: the two define it in reverse.
BATCHNORM_EPS = 0.001
BATCHNORM_MOMENTUM = 0.01

# Weight init follows the original (D18): He normal for the block convolutions,
# Glorot uniform (Keras's default) for the others, zero biases.
# `type: ignore` on zeros_: PyTorch types bias as optional.


def conv_block(in_channels, out_channels, batchnorm=True):
    """2 x (3x3 conv -> BatchNorm -> ReLU). Keeps height and width."""
    layers = []
    for layer_in_channels in [in_channels, out_channels]:
        conv = nn.Conv2d(layer_in_channels, out_channels, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
        nn.init.zeros_(conv.bias)  # type: ignore
        layers.append(conv)
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels, eps=BATCHNORM_EPS, momentum=BATCHNORM_MOMENTUM))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Upsample(nn.Module):
    """[N, C, H, W] -> [N, C', 2H, 2W]: Keras Conv2DTranspose(3x3, stride 2, padding="same").

    The transposed conv outputs 2H+1 x 2W+1; Keras keeps the first 2H x 2W.
    The usual PyTorch equivalent (padding=1, output_padding=1) keeps the last
    ones instead, a one-pixel shift (see test_upsample_matches_tensorflow).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)  # type: ignore

    def forward(self, image):
        return self.conv(image)[:, :, :-1, :-1]


class UNet(nn.Module):
    """in_channels   3 (B7, B6, B2) or 10
    base_filters  16 = U-Net-Light, 64 = U-Net; doubled at each level down
    dropout       0.1 in the original
    batchnorm     False for experiment T7

    Returns logits [N, 1, H, W]. The original ends with a sigmoid; here it is
    left to the loss (BCEWithLogitsLoss) and to prediction, for numerical
    stability.
    """

    def __init__(self, in_channels, base_filters=16, dropout=0.1, batchnorm=True):
        super().__init__()
        f = base_filters

        self.down1 = conv_block(in_channels, f, batchnorm)
        self.down2 = conv_block(f, f * 2, batchnorm)
        self.down3 = conv_block(f * 2, f * 4, batchnorm)
        self.down4 = conv_block(f * 4, f * 8, batchnorm)
        self.bottom = conv_block(f * 8, f * 16, batchnorm)

        self.halve = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)

        # Each up block takes the upsampled channels plus the skip connection's.
        self.double4 = Upsample(f * 16, f * 8)
        self.up4 = conv_block(f * 8 + f * 8, f * 8, batchnorm)
        self.double3 = Upsample(f * 8, f * 4)
        self.up3 = conv_block(f * 4 + f * 4, f * 4, batchnorm)
        self.double2 = Upsample(f * 4, f * 2)
        self.up2 = conv_block(f * 2 + f * 2, f * 2, batchnorm)
        self.double1 = Upsample(f * 2, f)
        self.up1 = conv_block(f + f, f, batchnorm)

        self.output = nn.Conv2d(f, 1, kernel_size=1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)  # type: ignore

    def forward(self, image):
        """image [N, in_channels, H, W], H and W divisible by 16 -> logits [N, 1, H, W].

        Comments give the original's variable names.
        """
        down1 = self.down1(image)  # c1
        down2 = self.down2(self.dropout(self.halve(down1)))  # c2
        down3 = self.down3(self.dropout(self.halve(down2)))  # c3
        down4 = self.down4(self.dropout(self.halve(down3)))  # c4
        bottom = self.bottom(self.dropout(self.halve(down4)))  # c5

        # Concatenation order matches the original (upsampled first), which
        # matters when loading its weights.
        doubled = self.double4(bottom)
        joined = torch.cat([doubled, down4], dim=1)  # u6
        up4 = self.up4(self.dropout(joined))  # c6

        doubled = self.double3(up4)
        joined = torch.cat([doubled, down3], dim=1)  # u7
        up3 = self.up3(self.dropout(joined))  # c7

        doubled = self.double2(up3)
        joined = torch.cat([doubled, down2], dim=1)  # u8
        up2 = self.up2(self.dropout(joined))  # c8

        doubled = self.double1(up2)
        joined = torch.cat([doubled, down1], dim=1)  # u9
        up1 = self.up1(self.dropout(joined))  # c9

        return self.output(up1)
