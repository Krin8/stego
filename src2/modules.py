"""P1 training head and explicitly distinct optimized inference embedding."""
import torch
from torch import nn
from torch.nn import functional as F
from .utils import resolve_path, sha256


class StegoHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout2d(p=.1)
        self.cluster1 = nn.Sequential(nn.Conv2d(768, 70, 1))
        self.cluster2 = nn.Sequential(nn.Conv2d(768, 768, 1), nn.ReLU(), nn.Conv2d(768, 70, 1))

    def forward(self, features):
        code = self.cluster1(self.dropout(features))
        code = code + self.cluster2(self.dropout(features))
        return self.dropout(features), code

    def embed_full(self, features):
        if self.training:
            raise ValueError('Inference requires eval mode')
        return F.normalize(self(features)[1], dim=1)

    def embed(self, features):
        if self.training:
            raise ValueError('Inference requires eval mode')
        # Deliberate optimized-P1 variant; do not silently restore full-head/L2.
        return self.cluster2(features)


def load_head(cfg):
    path = resolve_path(cfg, cfg['paths']['checkpoint'])
    actual = sha256(path)
    expected = cfg['inference'].get('checkpoint_sha256')
    if expected and actual != expected:
        raise ValueError('Head checkpoint SHA256 mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = StegoHead().eval().requires_grad_(False)
    loaded = model.load_state_dict(state['state_dict'], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError(str(loaded))
    return model, actual


def objective(criterion, cfg, features, partner_features, code, partner_code):
    a, _, b, _, c, _ = criterion(features, partner_features, None, None, code, partner_code)
    c = c.mean()
    total = cfg.correspondence_weight * (cfg.pos_intra_weight*a + cfg.pos_inter_weight*b + cfg.neg_inter_weight*c)
    return total, dict(pos_intra=a, pos_inter=b, neg_inter=c)
