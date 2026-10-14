"""Optional upstream alignment only. NOT a replacement for STEGO's three losses.

The caller must transform this coordinate image with exactly the image geometry.
No hard spatial negatives, InfoNCE, or mask-derived correspondences are invented.
"""
import torch
from torch.nn import functional as F
from reference_loss import norm, sample


def coordinate_image(height, width):
    y, x = torch.meshgrid(torch.linspace(-1, 1, height),
                          torch.linspace(-1, 1, width), indexing='ij')
    return torch.stack((x, y))


def alignment(code, augmented_code, transformed_coordinates):
    # sample() intentionally transposes the two spatial axes, as in pinned source.
    # Compensate explicitly when presenting a coordinate image in conventional HW.
    coordinates = F.interpolate(transformed_coordinates,
                                 size=augmented_code.shape[-2:], mode='bilinear',
                                 align_corners=True).permute(0, 3, 2, 1)
    return -(norm(sample(code, coordinates)) * norm(augmented_code)).sum(1).mean()
