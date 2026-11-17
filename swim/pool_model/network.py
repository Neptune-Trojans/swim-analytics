"""The pool segmentation network: a U-Net on an ImageNet-pretrained ResNet encoder (research/pool-model-training.md).

The encoder halves the image 5 times, so the input's sides must be divisible by 32. The decoder brings the features
back up, joined with the encoder's at each size (the skip connections), to one score per class for every input pixel:
full resolution, so thin things (lane ropes, later) are not lost.
"""

import torch
import torch.nn.functional as F
from torch import nn

# What the pretrained encoder expects: RGB in 0..1, normalized with ImageNet's mean and spread
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

ENCODERS = ("resnet18", "resnet34")
DECODER = (256, 128, 64, 32, 16)  # channels of the decoder's blocks, from 1/16 back up to full size


class UNet(nn.Module):
    def __init__(self, classes: int, encoder: str = "resnet18", pretrained: bool = True):
        super().__init__()
        import torchvision

        if encoder not in ENCODERS:
            raise ValueError(f"Unknown encoder '{encoder}' (one of: {', '.join(ENCODERS)})")
        resnet = getattr(torchvision.models, encoder)(weights="IMAGENET1K_V1" if pretrained else None)
        # Encoder: features at 1/2, 1/4, 1/8, 1/16, 1/32 of the input (64, 64, 128, 256, 512 channels)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.stages = nn.ModuleList([nn.Sequential(resnet.maxpool, resnet.layer1), resnet.layer2, resnet.layer3, resnet.layer4])
        skips = (256, 128, 64, 64, 0)  # encoder channels joined at each decoder block (none at full size)
        blocks, channels = [], 512
        for out, skip in zip(DECODER, skips):
            blocks.append(_block(channels + skip, out))
            channels = out
        self.decoder = nn.ModuleList(blocks)
        self.head = nn.Conv2d(channels, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Images (N, 3, H, W), normalized -> scores (N, classes, H, W)."""
        if x.shape[-2] % 32 or x.shape[-1] % 32:
            raise ValueError(f"Input sides must be divisible by 32, got {tuple(x.shape[-2:])}")
        features = [self.stem(x)]
        for stage in self.stages:
            features.append(stage(features[-1]))
        y = features.pop()
        for block in self.decoder:
            y = F.interpolate(y, scale_factor=2, mode="nearest")
            if features:
                y = torch.cat([y, features.pop()], 1)
            y = block(y)
        return self.head(y)

    def encoder_parameters(self) -> list[nn.Parameter]:
        return [*self.stem.parameters(), *self.stages.parameters()]

    def decoder_parameters(self) -> list[nn.Parameter]:
        return [*self.decoder.parameters(), *self.head.parameters()]


def _block(channels_in: int, channels_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False), nn.BatchNorm2d(channels_out), nn.ReLU(inplace=True),
        nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False), nn.BatchNorm2d(channels_out), nn.ReLU(inplace=True),
    )
