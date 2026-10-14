"""P1: body-frame co-located positives; original self/random terms unchanged."""
import torch
from reference_loss import ContrastiveCorrelationLoss, sample, super_perm


class ColocatedCorrelationLoss(ContrastiveCorrelationLoss):
    def __init__(self, cfg, valid):
        super().__init__(cfg)
        self.valid = valid

    def partner(self, features, partner_features, codes, partner_codes):
        # The inherited helper centers the FULL spatial matrix first. Only then
        # select co-located entries; centering just the diagonal is not STEGO.
        losses, correlations = self.helper(
            features, partner_features, codes, partner_codes,
            self.cfg.pos_inter_shift)
        diagonal = losses.flatten(1, 2).flatten(2, 3).diagonal(dim1=1, dim2=2)
        return diagonal.mean(), correlations

    def forward(self, orig_feats, orig_feats_pos, orig_salience,
                orig_salience_pos, orig_code, orig_code_pos):
        assert not self.cfg.use_salience
        assert orig_feats.shape[0] > 1
        valid = self.valid.to(device=orig_feats.device, dtype=torch.bool)
        assert valid.shape == (orig_feats.shape[0],)
        shape = [orig_feats.shape[0], self.cfg.feature_samples,
                 self.cfg.feature_samples, 2]
        coords1 = torch.rand(shape, device=orig_feats.device) * 2 - 1
        # Preserve the reference RNG stream and random-pair sampling.
        coords2 = torch.rand(shape, device=orig_feats.device) * 2 - 1
        features, codes = sample(orig_feats, coords1), sample(orig_code, coords1)
        positive_features = sample(orig_feats_pos, coords1)
        positive_codes = sample(orig_code_pos, coords1)
        intra, intra_cd = self.helper(features, features, codes, codes,
                                      self.cfg.pos_intra_shift)
        if valid.any():
            partner, partner_cd = self.partner(
                features[valid], positive_features[valid],
                codes[valid], positive_codes[valid])
        else:
            partner = codes.sum() * 0.
            partner_cd = codes.new_empty((0,))
        negative, negative_cd = [], []
        for _ in range(self.cfg.neg_samples):
            perm = super_perm(orig_feats.shape[0], orig_feats.device)
            loss, cd = self.helper(
                features, sample(orig_feats[perm], coords2), codes,
                sample(orig_code[perm], coords2), self.cfg.neg_inter_shift)
            negative.append(loss)
            negative_cd.append(cd)
        return (intra.mean(), intra_cd, partner, partner_cd,
                torch.cat(negative), torch.cat(negative_cd))


def adjacent_partners(records, anchor_indices, rng):
    """Uniform +/-1, +/-2 draw; invalid boundary draw skips partner term only."""
    groups = {}
    for i, record in enumerate(records):
        key = (record['patient_id'], record['series_uid'])
        groups.setdefault(key, []).append(i)
    locations = {}
    for ids in groups.values():
        ids.sort(key=lambda i: records[i]['physical_position'])
        assert len({records[i]['physical_position'] for i in ids}) == len(ids)
        for rank, i in enumerate(ids):
            locations[i] = (ids, rank)
    offsets = rng.choice([-2, -1, 1, 2], size=len(anchor_indices))
    partners, valid = [], []
    for index, offset in zip(anchor_indices, offsets):
        ids, rank = locations[int(index)]
        available = 0 <= rank + int(offset) < len(ids)
        partners.append(ids[rank + int(offset)] if available else int(index))
        valid.append(available)
    return partners, valid, offsets.tolist()
