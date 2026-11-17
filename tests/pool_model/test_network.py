import pytest
import torch

from swim.pool_model.network import UNet


def test_scores_for_every_input_pixel():
    net = UNet(classes=2, pretrained=False).eval()
    with torch.no_grad():
        assert net(torch.zeros(1, 3, 64, 128)).shape == (1, 2, 64, 128)


def test_input_sides_must_be_divisible_by_32():
    with pytest.raises(ValueError, match="divisible by 32"):
        UNet(classes=2, pretrained=False)(torch.zeros(1, 3, 72, 128))


def test_encoder_and_decoder_parameters_are_all_parameters():
    net = UNet(classes=2, pretrained=False)
    split = {id(p) for p in net.encoder_parameters()} | {id(p) for p in net.decoder_parameters()}
    assert split == {id(p) for p in net.parameters()}
