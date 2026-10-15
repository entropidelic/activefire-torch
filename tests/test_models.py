from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from activefire_torch.models import UNet, Upsample, conv_block, keras_kernel_to_torch, load_keras_weights


def test_conv_block_changes_channels_but_not_size():
    block = conv_block(in_channels=3, out_channels=16)
    assert block(torch.rand(2, 3, 256, 256)).shape == (2, 16, 256, 256)


def test_conv_block_ends_with_relu():
    block = conv_block(in_channels=3, out_channels=16)
    assert (block(torch.randn(2, 3, 32, 32)) >= 0).all()


def test_conv_block_without_batchnorm():
    block = conv_block(in_channels=3, out_channels=16, batchnorm=False)
    layer_types = [type(layer).__name__ for layer in block]
    assert layer_types == ["Conv2d", "ReLU", "Conv2d", "ReLU"]


def test_conv_block_parameter_count():
    block = conv_block(in_channels=3, out_channels=16)
    first_conv = 3 * 3 * 3 * 16 + 16  # weights + biases
    second_conv = 3 * 3 * 16 * 16 + 16
    batchnorms = 2 * (2 * 16)  # scale and shift per channel
    assert sum(p.numel() for p in block.parameters()) == first_conv + second_conv + batchnorms


def test_unet_output_has_one_channel_and_the_input_size():
    for in_channels in [3, 10]:
        model = UNet(in_channels=in_channels, base_filters=16)
        assert model(torch.rand(2, in_channels, 256, 256)).shape == (2, 1, 256, 256)


def test_unet_without_batchnorm():
    model = UNet(in_channels=3, batchnorm=False)
    layer_types = {type(layer).__name__ for layer in model.modules()}
    assert "BatchNorm2d" not in layer_types
    assert model(torch.rand(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_starting_weights():
    torch.manual_seed(0)
    model = UNet(in_channels=3, base_filters=16)

    # He normal: std = sqrt(2 / fan_in). up1's first conv: 32 channels, 3x3.
    conv = model.up1[0]
    assert isinstance(conv, torch.nn.Conv2d)
    expected = (2 / (32 * 3 * 3)) ** 0.5
    assert abs(conv.weight.std().item() - expected) < 0.05 * expected

    # Glorot uniform: |w| <= sqrt(6 / (fan_in + fan_out)).
    double1_limit = (6 / (32 * 3 * 3 + 16 * 3 * 3)) ** 0.5  # 32 -> 16 channels, 3x3
    assert model.double1.conv.weight.abs().max().item() <= double1_limit
    output_limit = (6 / (16 + 1)) ** 0.5  # 16 -> 1 channel, 1x1
    assert model.output.weight.abs().max().item() <= output_limit

    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            assert (layer.bias == 0).all()


# Paper Table 1 (p.10). Trainable parameters only: BatchNorm running statistics
# are buffers in PyTorch and non-trainable in Keras, so neither count includes them.
@pytest.mark.parametrize(
    "name, in_channels, base_filters, paper_count",
    [
        ("U-Net-Light (3c)", 3, 16, 2_161_649),
        ("U-Net (3c)", 3, 64, 34_525_121),
        ("U-Net (10c)", 10, 64, 34_529_153),
    ],
)
def test_parameter_count_matches_the_paper(name, in_channels, base_filters, paper_count):
    model = UNet(in_channels=in_channels, base_filters=base_filters)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == paper_count, name


def keras_conv2d(image, kernel, stride, pad_before, pad_after):
    """TensorFlow's conv2d written out from its definition, on channels-last
    image [N, H, W, in] with a Keras kernel [k, k, in, out]:
    output[y, x] = sum over dy, dx of padded[y * stride + dy, x * stride + dx] @ kernel[dy, dx].
    Uses no PyTorch convolution, so it checks our weight layout independently."""
    padded = F.pad(image, (0, 0, pad_before, pad_after, pad_before, pad_after))  # channels, width, height
    size = kernel.shape[0]
    out_height = (padded.shape[1] - size) // stride + 1
    out_width = (padded.shape[2] - size) // stride + 1
    output = torch.zeros(image.shape[0], out_height, out_width, kernel.shape[3])
    for dy in range(size):
        for dx in range(size):
            rows = slice(dy, dy + stride * (out_height - 1) + 1, stride)
            columns = slice(dx, dx + stride * (out_width - 1) + 1, stride)
            output = output + padded[:, rows, columns, :] @ kernel[dy, dx]
    return output


def test_converted_conv_matches_keras():
    torch.manual_seed(0)
    kernel = torch.randn(3, 3, 4, 5)  # Keras layout: height, width, in, out
    bias = torch.randn(5)
    image = torch.randn(1, 6, 6, 4)  # channels-last, as in Keras
    keras_result = keras_conv2d(image, kernel, stride=1, pad_before=1, pad_after=1) + bias

    conv = torch.nn.Conv2d(4, 5, kernel_size=3, padding=1)
    with torch.no_grad():
        conv.weight.copy_(keras_kernel_to_torch(kernel.numpy()))
        conv.bias.copy_(bias)  # type: ignore
    ours = conv(image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    assert torch.allclose(ours, keras_result, atol=1e-5)


def test_converted_upsample_matches_keras():
    # TensorFlow defines conv2d_transpose(padding="SAME") as the gradient of
    # conv2d(padding="SAME") with the same kernel. For a 3x3 kernel with
    # stride 2, that conv2d pads 0 before and 1 after.
    torch.manual_seed(0)
    kernel = torch.randn(3, 3, 2, 5)  # Keras transposed layout: height, width, out, in
    bias = torch.randn(2)
    small = torch.randn(1, 4, 4, 5)

    big = torch.zeros(1, 8, 8, 2, requires_grad=True)
    keras_conv2d(big, kernel, stride=2, pad_before=0, pad_after=1).backward(small)
    assert big.grad is not None
    keras_result = big.grad + bias

    layer = Upsample(in_channels=5, out_channels=2)
    with torch.no_grad():
        layer.conv.weight.copy_(keras_kernel_to_torch(kernel.numpy()))
        layer.conv.bias.copy_(bias)  # type: ignore
    ours = layer(small.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    assert torch.allclose(ours, keras_result, atol=1e-5)


AUTHORS_WEIGHTS = Path("data/raw/weights/voting/voting/unet_16f_2conv_762/model_unet_Voting_final_weights.h5")


@pytest.mark.skipif(not AUTHORS_WEIGHTS.exists(), reason="authors' weights not downloaded")
def test_load_authors_weights():
    model = UNet(in_channels=3, base_filters=16)
    load_keras_weights(model, AUTHORS_WEIGHTS)
    first_conv = model.down1[0]
    assert isinstance(first_conv, torch.nn.Conv2d)
    assert first_conv.weight.abs().sum() > 0

    with pytest.raises(ValueError, match="does not fit"):
        load_keras_weights(UNet(in_channels=10, base_filters=16), AUTHORS_WEIGHTS)
