import pytest
import torch
import torch.nn.functional as F

from activefire_torch.models import UNet, conv_block, up_layer, upsample


def test_conv_block_changes_channels_but_not_size():
    block = conv_block(in_channels=3, out_channels=16)
    images = torch.rand(2, 3, 256, 256)  # 2 images, 3 bands, 256 x 256
    assert block(images).shape == (2, 16, 256, 256)


def test_conv_block_output_is_never_negative():
    # The last layer is a ReLU, which turns negative numbers into 0.
    block = conv_block(in_channels=3, out_channels=16)
    assert (block(torch.randn(2, 3, 32, 32)) >= 0).all()


def test_conv_block_without_batchnorm():
    block = conv_block(in_channels=3, out_channels=16, batchnorm=False)
    layer_types = [type(layer).__name__ for layer in block]
    assert layer_types == ["Conv2d", "ReLU", "Conv2d", "ReLU"]


def test_conv_block_parameter_count():
    # Each 3x3 convolution has 3 * 3 * in_channels weights per filter, plus one
    # bias per filter. Each BatchNorm has 2 learned numbers per channel.
    block = conv_block(in_channels=3, out_channels=16)
    first_conv = 3 * 3 * 3 * 16 + 16  # 448
    second_conv = 3 * 3 * 16 * 16 + 16  # 2,320
    batchnorms = 2 * (2 * 16)  # 64
    assert sum(p.numel() for p in block.parameters()) == first_conv + second_conv + batchnorms


def test_upsample_matches_tensorflow():
    # TensorFlow defines its transposed convolution (Keras Conv2DTranspose,
    # padding="same") as "the gradient of a normal convolution". For a 3x3
    # filter with stride 2, that normal convolution adds 0 pixels before and
    # 1 pixel after the image. Build it, let PyTorch compute its gradient, and
    # compare with our upsample().
    torch.manual_seed(0)
    layer = up_layer(in_channels=5, out_channels=2)
    small = torch.randn(1, 5, 4, 4)

    big = torch.zeros(1, 2, 8, 8, requires_grad=True)
    padded = F.pad(big, (0, 1, 0, 1))  # 0 before, 1 after, for width and height
    F.conv2d(padded, layer.weight, stride=2).backward(small)
    assert big.grad is not None and layer.bias is not None
    tensorflow_result = big.grad + layer.bias.view(1, -1, 1, 1)  # plus one bias per channel

    assert torch.allclose(upsample(layer, small), tensorflow_result, atol=1e-6)


def test_unet_output_has_one_channel_and_the_input_size():
    for in_channels in [3, 10]:
        model = UNet(in_channels=in_channels, base_filters=16)
        images = torch.rand(2, in_channels, 256, 256)
        assert model(images).shape == (2, 1, 256, 256)


def test_unet_without_batchnorm():
    model = UNet(in_channels=3, batchnorm=False)
    layer_types = {type(layer).__name__ for layer in model.modules()}
    assert "BatchNorm2d" not in layer_types
    assert model(torch.rand(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_starting_weights():
    torch.manual_seed(0)
    model = UNet(in_channels=3, base_filters=16)

    # He: spread sqrt(2 / inputs per filter). For the first convolution of
    # up1 (32 channels in, 3x3) that is sqrt(2 / 288) = 0.083.
    conv = model.up1[0]
    assert isinstance(conv, torch.nn.Conv2d)
    expected = (2 / (32 * 3 * 3)) ** 0.5
    assert abs(conv.weight.std().item() - expected) < 0.05 * expected

    # Glorot uniform: every value within +-sqrt(6 / (inputs + outputs)),
    # counting inputs and outputs per position of the filter.
    double1_limit = (6 / (32 * 3 * 3 + 16 * 3 * 3)) ** 0.5  # 32 -> 16 channels, 3x3
    assert model.double1.weight.abs().max().item() <= double1_limit
    output_limit = (6 / (16 + 1)) ** 0.5  # 16 -> 1 channel, 1x1
    assert model.output.weight.abs().max().item() <= output_limit

    # Every bias starts at 0.
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            assert (layer.bias == 0).all()


# Trainable parameters in the paper's Table 1 (p.10). The BatchNorm running
# averages are not trainable: Keras leaves them out of this count, and PyTorch
# keeps them as "buffers", not parameters, so they are left out here too.
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
