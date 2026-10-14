"""Pinned upstream STEGO loss; only TorchScript decorator omitted (CPU eager)."""
import torch
from torch import nn
from torch.nn import functional as F

def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)


def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


def sample_nonzero_locations(t, target_size):
    nonzeros = torch.nonzero(t)
    coords = torch.zeros(target_size, dtype=nonzeros.dtype, device=nonzeros.device)
    n = target_size[1] * target_size[2]
    for i in range(t.shape[0]):
        selected_nonzeros = nonzeros[nonzeros[:, 0] == i]
        if selected_nonzeros.shape[0] == 0:
            selected_coords = torch.randint(t.shape[1], size=(n, 2), device=nonzeros.device)
        else:
            selected_coords = selected_nonzeros[torch.randint(len(selected_nonzeros), size=(n,)), 1:]
        coords[i, :, :, :] = selected_coords.reshape(target_size[1], target_size[2], 2)
    coords = coords.to(torch.float32) / t.shape[1]
    coords = coords * 2 - 1
    return torch.flip(coords, dims=[-1])


class ContrastiveCorrelationLoss(nn.Module):

    def __init__(self, cfg, ):
        super(ContrastiveCorrelationLoss, self).__init__()
        self.cfg = cfg

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = tensor_correlation(norm(f1), norm(f2))

            if self.cfg.pointwise:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))

        if self.cfg.zero_clamp:
            min_val = 0.0
        else:
            min_val = -9999.0

        if self.cfg.stabalize:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        else:
            loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def forward(self,
                orig_feats: torch.Tensor, orig_feats_pos: torch.Tensor,
                orig_salience: torch.Tensor, orig_salience_pos: torch.Tensor,
                orig_code: torch.Tensor, orig_code_pos: torch.Tensor,
                ):

        coord_shape = [orig_feats.shape[0], self.cfg.feature_samples, self.cfg.feature_samples, 2]

        if self.cfg.use_salience:
            coords1_nonzero = sample_nonzero_locations(orig_salience, coord_shape)
            coords2_nonzero = sample_nonzero_locations(orig_salience_pos, coord_shape)
            coords1_reg = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            coords2_reg = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            mask = (torch.rand(coord_shape[:-1], device=orig_feats.device) > .1).unsqueeze(-1).to(torch.float32)
            coords1 = coords1_nonzero * mask + coords1_reg * (1 - mask)
            coords2 = coords2_nonzero * mask + coords2_reg * (1 - mask)
        else:
            coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = sample(orig_feats, coords1)
        code = sample(orig_code, coords1)

        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)

        pos_intra_loss, pos_intra_cd = self.helper(
            feats, feats, code, code, self.cfg.pos_intra_shift)
        pos_inter_loss, pos_inter_cd = self.helper(
            feats, feats_pos, code, code_pos, self.cfg.pos_inter_shift)

        neg_losses = []
        neg_cds = []
        for i in range(self.cfg.neg_samples):
            perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
            feats_neg = sample(orig_feats[perm_neg], coords2)
            code_neg = sample(orig_code[perm_neg], coords2)
            neg_inter_loss, neg_inter_cd = self.helper(
                feats, feats_neg, code, code_neg, self.cfg.neg_inter_shift)
            neg_losses.append(neg_inter_loss)
            neg_cds.append(neg_inter_cd)
        neg_inter_loss = torch.cat(neg_losses, axis=0)
        neg_inter_cd = torch.cat(neg_cds, axis=0)

        return (pos_intra_loss.mean(),
                pos_intra_cd,
                pos_inter_loss.mean(),
                pos_inter_cd,
                neg_inter_loss,
                neg_inter_cd)


