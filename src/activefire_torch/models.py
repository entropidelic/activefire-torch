"""The U-Net from the original repo, rebuilt in PyTorch.

Copied from get_unet and conv2d_block in the original
src/train/voting/unet_16f_2conv_762/models.py (all 15 copies are identical,
docs/paper_notes.md §6).
"""

import torch
from torch import nn

# Keras and PyTorch use different default settings for BatchNorm. We use
# Keras's, so our model behaves like the original:
#   eps        a tiny number added before dividing by the spread, so it never
#              divides by zero. Keras: 0.001, PyTorch: 0.00001.
#   momentum   how fast the running average (used at test time) follows
#              each new batch. Keras keeps 99% of the old value
#              (momentum=0.99); PyTorch counts the other way round and calls
#              the new batch's 1% "momentum=0.01". Same rule, written two ways.
BATCHNORM_EPS = 0.001
BATCHNORM_MOMENTUM = 0.01

# Starting weights (docs/DECISIONS.md D18). Before training, every weight gets
# a random value; biases start at 0, as in Keras. The original uses two kinds
# of random start:
#   - the convolutions inside each block: "he_normal", which picks the spread
#     so the values keep about the same size as they pass through many ReLU
#     layers. PyTorch calls it kaiming_normal_ (He and Kaiming are the same
#     person). Keras cuts off the rare values far from 0; we don't, which
#     can't change the results in any way that matters.
#   - every other layer (the up layers and the output): Keras's default,
#     "glorot_uniform". PyTorch calls it xavier_uniform_ (also one person,
#     Xavier Glorot).
# The "type: ignore" next to each zeros_ is for the type checker: PyTorch says
# a bias may be None, which only happens for layers made with bias=False.


def conv_block(in_channels, out_channels, batchnorm=True):
    """Two rounds of: 3x3 convolution -> BatchNorm -> ReLU.

    in_channels   how many channels come in (e.g. 3 for the B7-B6-B2 image)
    out_channels  how many channels (filters) come out
    batchnorm     False removes the BatchNorm layers (experiment T7)

    The height and width stay the same: padding=1 adds a 1-pixel border, so
    a 3x3 filter can also be centered on edge pixels (Keras: padding="same").
    """
    layers = []
    for layer_in_channels in [in_channels, out_channels]:
        # The first convolution changes the number of channels; the second
        # one keeps it.
        conv = nn.Conv2d(layer_in_channels, out_channels, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
        nn.init.zeros_(conv.bias)  # type: ignore
        layers.append(conv)
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels, eps=BATCHNORM_EPS, momentum=BATCHNORM_MOMENTUM))
        layers.append(nn.ReLU())
    # nn.Sequential runs the layers one after the other.
    return nn.Sequential(*layers)


def up_layer(in_channels, out_channels):
    """A layer that doubles the height and width (a 3x3 transposed convolution
    with stride 2, Keras: Conv2DTranspose(..., (3, 3), strides=(2, 2), padding="same")).

    Use it through upsample(), which also removes one extra row and column.
    """
    layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)  # type: ignore
    return layer


def upsample(layer, image):
    """Apply an up_layer: [N, C, H, W] -> [N, C', 2H, 2W].

    The layer itself gives 2H + 1 rows and 2W + 1 columns. Keras's "same"
    version keeps the first 2H and 2W, so we drop the last row and column.
    (The common PyTorch shortcut, padding=1 with output_padding=1, also gives
    2H x 2W but drops the first row and column instead, which shifts the
    result by one pixel compared with the original. Checked numerically against
    TensorFlow's definition before writing this.)
    """
    return layer(image)[:, :, :-1, :-1]


class UNet(nn.Module):
    """The original get_unet, layer by layer.

    in_channels   3 for B7-B6-B2, 10 for all bands
    base_filters  filters in the first block; each step down doubles them.
                  16 = U-Net-Light, 64 = U-Net (paper Table 1)
    dropout       share of values set to 0 at random during training (0.1 in
                  the original)
    batchnorm     False removes every BatchNorm layer (experiment T7)

    Sizes for a 256 x 256 input and base_filters=16 (channels, height x width):

        down1   16, 256      ->      up1   16, 256  -> output 1, 256
        down2   32, 128      ->      up2   32, 128
        down3   64,  64      ->      up3   64,  64
        down4  128,  32      ->      up4  128,  32
                     bottom  256, 16

    The model returns one number per pixel (a "logit"): above 0 means fire is
    more likely than not. The original ends with a sigmoid that turns it into
    a probability from 0 to 1; we apply the sigmoid in the loss and at
    prediction time instead, because PyTorch's loss is more accurate that way.
    """

    def __init__(self, in_channels, base_filters=16, dropout=0.1, batchnorm=True):
        super().__init__()
        f = base_filters  # short, because it appears in every line below

        # Going down: each block finds patterns, then the image is halved.
        self.down1 = conv_block(in_channels, f, batchnorm)
        self.down2 = conv_block(f, f * 2, batchnorm)
        self.down3 = conv_block(f * 2, f * 4, batchnorm)
        self.down4 = conv_block(f * 4, f * 8, batchnorm)
        self.bottom = conv_block(f * 8, f * 16, batchnorm)

        # Keeps the largest of each 2 x 2 group of pixels: halves height and width.
        self.halve = nn.MaxPool2d(2)
        # Has no learned numbers, so one layer can be used everywhere.
        self.dropout = nn.Dropout(dropout)

        # Going up: double the image, then join it with the block from the
        # way down that has the same size (the "skip connection"). Joining
        # adds up the channels of both, e.g. f * 8 + f * 8 for up4.
        self.double4 = up_layer(f * 16, f * 8)
        self.up4 = conv_block(f * 8 + f * 8, f * 8, batchnorm)
        self.double3 = up_layer(f * 8, f * 4)
        self.up3 = conv_block(f * 4 + f * 4, f * 4, batchnorm)
        self.double2 = up_layer(f * 4, f * 2)
        self.up2 = conv_block(f * 2 + f * 2, f * 2, batchnorm)
        self.double1 = up_layer(f * 2, f)
        self.up1 = conv_block(f + f, f, batchnorm)

        # A 1x1 convolution: combines the f channels of each pixel into one number.
        self.output = nn.Conv2d(f, 1, kernel_size=1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)  # type: ignore

    def forward(self, image):
        """image [N, in_channels, H, W] -> logits [N, 1, H, W].

        H and W must be divisible by 16 (they are halved 4 times).
        The comments give the names in the original code (c1 ... c9, u6 ... u9).
        """
        # Going down.
        down1 = self.down1(image)  # c1
        down2 = self.down2(self.dropout(self.halve(down1)))  # p1 -> c2
        down3 = self.down3(self.dropout(self.halve(down2)))  # p2 -> c3
        down4 = self.down4(self.dropout(self.halve(down3)))  # p3 -> c4
        bottom = self.bottom(self.dropout(self.halve(down4)))  # p4 -> c5

        # Going up, one level at a time:
        #   1. double the height and width
        #   2. join it with the result of the same size from the way down.
        #      torch.cat joins along dim 1 (the channels), in the same order
        #      as the original: the doubled image first.
        #   3. dropout, then a block
        doubled = upsample(self.double4, bottom)  # u6
        joined = torch.cat([doubled, down4], dim=1)
        up4 = self.up4(self.dropout(joined))  # c6

        doubled = upsample(self.double3, up4)  # u7
        joined = torch.cat([doubled, down3], dim=1)
        up3 = self.up3(self.dropout(joined))  # c7

        doubled = upsample(self.double2, up3)  # u8
        joined = torch.cat([doubled, down2], dim=1)
        up2 = self.up2(self.dropout(joined))  # c8

        doubled = upsample(self.double1, up2)  # u9
        joined = torch.cat([doubled, down1], dim=1)
        up1 = self.up1(self.dropout(joined))  # c9

        return self.output(up1)
