"""STEGO head: pinned modules.py:73-81,108-118, adapted to 768 channels."""
import torch
from torch import nn
from torch.nn import functional as F
from reference_loss import ContrastiveCorrelationLoss


class StegoHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout2d(p=.1)
        self.learned_head = nn.Sequential(
            nn.Conv2d(768, 768, 1),
            nn.ReLU(),
            nn.Conv2d(768, 768, 1)
        )
        # Initialize last layer near zero
        with torch.no_grad():
            self.learned_head[2].weight.normal_(mean=0.0, std=0.01)
            self.learned_head[2].bias.fill_(0.0)

    def forward(self, features):
        assert features.ndim == 4 and features.shape[1] == 768
        correction = self.learned_head(self.dropout(features))
        code = features + correction
        # The reference draws an independent channel-dropout mask for the signal.
        return self.dropout(features), code

    def embed(self, features):
        assert not self.training, 'Inference requires eval()'
        return F.normalize(self(features)[1], dim=1)


def objective(criterion, cfg, features, neighbors, code, neighbor_code, salience=None, neighbor_salience=None):
    assert features.shape[0] > 1, 'Random-image term requires batch > 1'
    intra, _, inter, _, negative, _ = criterion(
        features, neighbors, salience, neighbor_salience, code, neighbor_code)
    negative = negative.mean()
    total = cfg.correspondence_weight * (
        cfg.pos_intra_weight * intra + cfg.pos_inter_weight * inter
        + cfg.neg_inter_weight * negative)
    return total, dict(pos_intra=intra, pos_inter=inter, neg_inter=negative)
