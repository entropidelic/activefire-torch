"""U-Net, ported from get_unet and conv2d_block in the original
src/train/*/*/models.py (docs/paper_notes.md §6)."""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

# Keras BatchNorm defaults (PyTorch's are eps=1e-5, momentum=0.1). Keras's
# momentum=0.99 is PyTorch's momentum=0.01: the two define it in reverse.
BATCHNORM_EPS = 0.001
BATCHNORM_MOMENTUM = 0.01

# Weight init follows the original (D18): He normal for the block convolutions,
# Glorot uniform (Keras's default) for the others, zero biases.
# `type: ignore` on zeros_: PyTorch types bias as optional.


def conv_block(in_channels: int, out_channels: int, batchnorm: bool = True) -> nn.Sequential:
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
    ones instead, a one-pixel shift (see test_converted_upsample_matches_keras).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)  # type: ignore

    def forward(self, image: torch.Tensor) -> torch.Tensor:
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

    def __init__(self, in_channels: int, base_filters: int = 16, dropout: float = 0.1, batchnorm: bool = True) -> None:
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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
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


def keras_kernel_to_torch(kernel: np.ndarray) -> torch.Tensor:
    """Keras conv kernel (height, width, in, out) -> PyTorch Conv2d weight (out, in, height, width).

    The same reordering turns a Keras transposed-conv kernel (height, width,
    out, in) into the PyTorch ConvTranspose2d layout (in, out, height, width).
    """
    return torch.from_numpy(np.ascontiguousarray(kernel.transpose(3, 2, 0, 1)))


def load_keras_weights(model: UNet, path: str | Path) -> None:
    """Copy the authors' trained weights (an .h5 file from their download_weights.py)
    into model.

    Keras and PyTorch both list layers in creation order, which is the same in
    get_unet and UNet, so layers are paired by position. Each pair's layer type
    and every shape are checked.
    """
    torch_layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d))]

    keras_layers = []
    with h5py.File(path, "r") as file:
        for layer_name in np.array(file.attrs["layer_names"]):
            group = file[layer_name]
            assert isinstance(group, h5py.Group)
            # e.g. b"conv2d_1/kernel:0" -> "kernel"
            weights = {
                weight_name.decode().split("/")[-1].removesuffix(":0"): np.array(group[weight_name])
                for weight_name in np.array(group.attrs["weight_names"])
            }
            if weights:  # activation, pooling, dropout and concatenate layers have none
                keras_layers.append((layer_name.decode(), weights))

    if len(keras_layers) != len(torch_layers):
        raise ValueError(f"{path} has {len(keras_layers)} layers with weights, the model {len(torch_layers)}")

    keras_type_by_torch_type = {
        nn.Conv2d: "conv2d",
        nn.ConvTranspose2d: "conv2d_transpose",
        nn.BatchNorm2d: "batch_normalization",
    }
    for torch_layer, (keras_name, weights) in zip(torch_layers, keras_layers):
        expected = keras_type_by_torch_type[type(torch_layer)]
        if keras_name.rsplit("_", 1)[0] != expected:  # "conv2d_transpose_1" -> "conv2d_transpose"
            raise ValueError(f"Expected a {expected} layer, found {keras_name}")

        if isinstance(torch_layer, nn.BatchNorm2d):
            pairs = [
                (torch_layer.weight, weights["gamma"]),
                (torch_layer.bias, weights["beta"]),
                (torch_layer.running_mean, weights["moving_mean"]),
                (torch_layer.running_var, weights["moving_variance"]),
            ]
        else:
            pairs = [
                (torch_layer.weight, keras_kernel_to_torch(weights["kernel"])),
                (torch_layer.bias, weights["bias"]),
            ]

        for target, source in pairs:
            source = torch.as_tensor(source)
            # copy_ would silently broadcast a wrong shape, so check first.
            if target.shape != source.shape:
                raise ValueError(f"{keras_name}: shape {tuple(source.shape)} does not fit {tuple(target.shape)}")
            with torch.no_grad():
                target.copy_(source)
