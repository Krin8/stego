"""STEGO head: pinned modules.py:73-81,108-118, adapted to 768 channels."""
import torch
from torch import nn
from torch.nn import functional as F
from reference_loss import ContrastiveCorrelationLoss


class StegoHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout2d(p=.1)
        self.cluster1 = nn.Sequential(nn.Conv2d(768, 70, 1))
        self.cluster2 = nn.Sequential(nn.Conv2d(768, 768, 1), nn.ReLU(),
                                      nn.Conv2d(768, 70, 1))

    def forward(self, features):
        assert features.ndim == 4 and features.shape[1] == 768
        code = self.cluster1(self.dropout(features))
        code = code + self.cluster2(self.dropout(features))
        # The reference draws an independent channel-dropout mask for the signal.
        return self.dropout(features), code

    def embed(self, features):
        assert not self.training, 'Inference requires eval()'
        return F.normalize(self(features)[1], dim=1)


def objective(criterion, cfg, features, neighbors, code, neighbor_code):
    assert features.shape[0] > 1, 'Random-image term requires batch > 1'
    intra, _, inter, _, negative, _ = criterion(
        features, neighbors, None, None, code, neighbor_code)
    negative = negative.mean()
    total = cfg.correspondence_weight * (
        cfg.pos_intra_weight * intra + cfg.pos_inter_weight * inter
        + cfg.neg_inter_weight * negative)
    return total, dict(pos_intra=intra, pos_inter=inter, neg_inter=negative)
