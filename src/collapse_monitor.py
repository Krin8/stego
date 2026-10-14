"""Label-free validation statistics and consecutive-check collapse stopping."""
import torch
import torch.nn.functional as F


class CollapseMonitor:
    def __init__(self, clusters):
        self.usage = torch.zeros(clusters, dtype=torch.float64)
        self.pixel_sum = self.pixel_square_sum = self.image_sum = None
        self.image_square_sum = 0.0
        self.pixels = self.images = 0

    @torch.no_grad()
    def update(self, code, cluster_ids):
        unit = F.normalize(code.detach(), dim=1).to(device='cpu', dtype=torch.float64)
        pixels = unit.permute(1, 0, 2, 3).flatten(1)
        means = unit.flatten(2).mean(2)
        if self.pixel_sum is None:
            self.pixel_sum = torch.zeros_like(pixels.sum(1))
            self.pixel_square_sum = torch.zeros_like(self.pixel_sum)
            self.image_sum = torch.zeros_like(self.pixel_sum)
        self.pixel_sum += pixels.sum(1)
        self.pixel_square_sum += pixels.square().sum(1)
        self.image_sum += means.sum(0)
        self.image_square_sum += means.square().sum().item()
        self.pixels += pixels.shape[1]
        self.images += len(means)
        self.usage += torch.bincount(cluster_ids.detach().cpu().flatten(), minlength=len(self.usage))

    def compute(self):
        if self.images < 2:
            raise ValueError('Collapse monitoring requires at least two validation images')
        variance = self.pixel_square_sum / self.pixels - (self.pixel_sum / self.pixels).square()
        cosine = (self.image_sum.square().sum().item() - self.image_square_sum) / (self.images * (self.images - 1))
        fractions = self.usage / self.usage.sum().clamp_min(1)
        entropy = -(fractions * fractions.clamp_min(1e-12).log()).sum().item()
        return dict(code_std=variance.clamp_min(0).sqrt().mean().item(),
                    cross_image_cosine=cosine, cluster_usage=fractions.tolist(),
                    cluster_entropy=entropy, images=self.images)


class CollapseStopper:
    def __init__(self, cosine=0.99, std=0.01, patience=3):
        self.cosine, self.std, self.patience = cosine, std, patience
        self.bad_checks = 0

    def check(self, stats):
        bad = stats['cross_image_cosine'] > self.cosine or stats['code_std'] < self.std
        self.bad_checks = self.bad_checks + 1 if bad else 0
        return self.bad_checks >= self.patience
