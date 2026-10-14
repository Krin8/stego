"""Synthetic before/after checks and backward-only verification; no updates."""
from common import *

cfg = SimpleNamespace(**{**CFG, 'feature_samples': 3, 'pos_inter_shift': 1.04})
torch.manual_seed(87)
f, fp = torch.randn(3, 8, 5, 5), torch.randn(3, 8, 5, 5)
z, zp = torch.randn(3, 4, 5, 5), torch.randn(3, 4, 5, 5)
valid = torch.tensor([True, False, True])
original = ContrastiveCorrelationLoss(cfg)
p1 = ColocatedCorrelationLoss(cfg, valid)
torch.manual_seed(98)
before = original(f, fp, None, None, z, zp)
torch.manual_seed(98)
after = p1(f, fp, None, None, z, zp)
assert torch.equal(before[0], after[0])
assert torch.equal(before[4], after[4])
assert not torch.equal(before[2], after[2])

# Independently recover the aligned full helper matrix, then select diagonal.
torch.manual_seed(98)
coords = torch.rand(3, 3, 3, 2) * 2 - 1
fa, fb = sample(f, coords)[valid], sample(fp, coords)[valid]
za, zb = sample(z, coords)[valid], sample(zp, coords)[valid]
full, _ = original.helper(fa, fb, za, zb, cfg.pos_inter_shift)
expected = torch.stack([full[:, a, b, a, b] for a in range(3) for b in range(3)]).mean()
assert torch.allclose(after[2], expected, atol=1e-7, rtol=0.)
# Invalid partners cannot influence centering or loss.
fp_changed, zp_changed = fp.clone(), zp.clone()
fp_changed[1], zp_changed[1] = 900., -900.
torch.manual_seed(98)
changed = p1(f, fp_changed, None, None, z, zp_changed)
assert torch.equal(changed[2], after[2])
torch.manual_seed(98)
empty = ColocatedCorrelationLoss(cfg, torch.zeros(3, dtype=torch.bool))(f, fp, None, None, z, zp)
assert float(empty[2]) == 0.

with torch.enable_grad():
    fa, fb = f.clone().requires_grad_(), fp.clone().requires_grad_()
    za, zb = z.clone().requires_grad_(), zp.clone().requires_grad_()
    total, _ = objective(p1, cfg, fa, fb, za, zb)
    total.backward()
    assert fa.grad is None and fb.grad is None
    assert za.grad is not None and zb.grad is not None
    assert torch.isfinite(za.grad).all() and torch.isfinite(zb.grad).all()
    assert za.grad.abs().sum() > 0 and zb.grad.abs().sum() > 0
    assert zb.grad[1].abs().sum() == 0

records = [dict(patient_id=p, series_uid=p, physical_position=x)
           for p in ['a', 'b'] for x in [2., 0., 1.]]
ids = list(range(6)) * 30
partners, available, offsets = adjacent_partners(records, ids, np.random.default_rng(1))
for a, b, ok, offset in zip(ids, partners, available, offsets):
    assert records[a]['patient_id'] == records[b]['patient_id']
    if ok:
        assert records[b]['physical_position'] - records[a]['physical_position'] == offset
    else:
        assert b == a
assert set(offsets) == {-2, -1, 1, 2} and not all(available)
dump('implementation_checks.json', dict(source='check_loss.py', labels_used=False,
    original_inter=float(before[2]), p1_inter=float(after[2]), expected_diagonal=float(expected),
    self_and_random_bit_identical=True, valid_only_centering=True, empty_partner_zero=True,
    feature_gradients_absent=True, code_gradients_finite_nonzero=True,
    invalid_partner_gradient_zero=True, physical_order_boundary_checks=True,
    optimizer_updates=0, status='PASS'))
ledger('implementation_access.json')
print('PASS: unchanged self/random; aligned centered diagonal; boundaries; backward only')
