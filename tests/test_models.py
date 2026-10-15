import pytest
import torch
import torch.nn.functional as F

from activefire_torch.models import UNet, Upsample, conv_block


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


def test_upsample_matches_tensorflow():
    # TensorFlow defines conv2d_transpose(padding="SAME") as the gradient of
    # conv2d(padding="SAME"), which for a 3x3 kernel with stride 2 pads 0 before
    # and 1 after. Compute that gradient and compare.
    torch.manual_seed(0)
    layer = Upsample(in_channels=5, out_channels=2)
    small = torch.randn(1, 5, 4, 4)

    big = torch.zeros(1, 2, 8, 8, requires_grad=True)
    F.conv2d(F.pad(big, (0, 1, 0, 1)), layer.conv.weight, stride=2).backward(small)
    assert big.grad is not None and layer.conv.bias is not None
    tensorflow_result = big.grad + layer.conv.bias.view(1, -1, 1, 1)

    assert torch.allclose(layer(small), tensorflow_result, atol=1e-6)


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
