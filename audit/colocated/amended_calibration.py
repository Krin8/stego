"""User-fixed five-target grid; full-loss-only gate; annotations remain closed."""
from common import *

TARGETS = [.2, .4, .6, .8, .9]
SEED = 20261003
protocol = read(OUT / 'protocol.json')
anchors = np.array(protocol['anchors_local'])
partners = np.array(protocol['partners_local'])
valid = np.array(protocol['valid'])
assert len(anchors) == 200
centers = torch.from_numpy(np.load(ROOT / 'audit/final_pipeline/inductive_seed0_centers.npy', allow_pickle=False))
tree = ast.parse((ROOT / 'audit/preflight/diagnose.py').read_text())
node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'codes')
exec(compile(ast.Module(body=[node], type_ignores=[]), 'preflight codes', 'exec'), globals())
dump('amended_protocol.json', dict(recorded_utc='2026-10-03T00:21:00Z',
    targets=TARGETS, seed=SEED, source_protocol_sha256=sha(OUT / 'protocol.json'),
    gate='constant > random > K8 in full-loss mean AND all batches including final remainder',
    selection='largest mean random minus K8 full loss among admissible targets',
    labels_used=False, previous_failed_attempt_preserved=True,
    source_sha256=sha(Path(__file__))))

class Capture(ColocatedCorrelationLoss):
    def partner(self, fa, fb, za, zb):
        ones = torch.ones_like(za[:, :1])
        ll, _ = self.helper(fa, fb, ones, ones, 0.)
        self.affinity = (-ll).flatten(1, 2).flatten(2, 3).diagonal(dim1=1, dim2=2).numpy().copy()
        return super().partner(fa, fb, za, zb)

affinities = []
for start in range(0, 200, 16):
    f, fp = feature_batch(TRAIN[anchors[start:start+16]]), feature_batch(TRAIN[partners[start:start+16]])
    crit = Capture(SimpleNamespace(**CFG), torch.tensor(valid[start:start+16]))
    torch.manual_seed(SEED + start)
    objective(crit, crit.cfg, f, fp, codes(f, 'constant'), codes(fp, 'constant'))
    affinities.append(crit.affinity.ravel())
affinity = np.concatenate(affinities)
shifts = [float(np.quantile(affinity, 1-rate)) for rate in TARGETS]
rows = []
for start in range(0, 200, 16):
    f, fp = feature_batch(TRAIN[anchors[start:start+16]]), feature_batch(TRAIN[partners[start:start+16]])
    torch.manual_seed(SEED + start + 10)
    pairs = {kind: (codes(f, kind), codes(fp, kind)) for kind in ['constant', 'random', 'cluster']}
    for rate, shift in zip(TARGETS, shifts):
        cfg = SimpleNamespace(**{**CFG, 'pos_inter_shift': shift})
        criterion = ColocatedCorrelationLoss(cfg, torch.tensor(valid[start:start+16]))
        losses = {}
        for kind, (z, zp) in pairs.items():
            torch.manual_seed(SEED + start)
            total, parts = objective(criterion, cfg, f, fp, z, zp)
            losses[kind] = dict(total=float(total), **{k: float(v) for k,v in parts.items()})
        passed = losses['constant']['total'] > losses['random']['total'] > losses['cluster']['total']
        rows.append(dict(target=rate, shift=shift, batch=start//16+1, slices=len(f), losses=losses, passed=passed))
    print('BATCH', start//16+1, flush=True)
summary = []
for rate, shift in zip(TARGETS, shifts):
    rr = [r for r in rows if r['target'] == rate]
    losses = {kind: {term: sum(r['slices'] * r['losses'][kind][term] for r in rr)/200
        for term in ['total', 'pos_intra', 'pos_inter', 'neg_inter']} for kind in ['constant', 'random', 'cluster']}
    mean_pass = losses['constant']['total'] > losses['random']['total'] > losses['cluster']['total']
    summary.append(dict(target=rate, shift=shift, attraction=float((affinity>shift).mean()),
        losses=losses, mean_pass=mean_pass, batches_passed=sum(r['passed'] for r in rr),
        failed_batches=[r['batch'] for r in rr if not r['passed']],
        advantage=losses['random']['total']-losses['cluster']['total'],
        admissible=mean_pass and all(r['passed'] for r in rr)))
selected = max((r for r in summary if r['admissible']), key=lambda r:r['advantage'], default=None)
# Reproduce the original 20% numeric values; revised gate must not change them.
old = read(OUT/'verification.json')
assert shifts[0] == old['shift']
for kind in ['constant', 'random', 'cluster']:
    for term in ['total', 'pos_inter']:
        assert abs(summary[0]['losses'][kind][term]-old['summary']['P1'][kind][term]) < 1e-10
dump('amended_calibration.json', dict(source='amended_calibration.py', labels_used=False,
    summary=summary, batches=rows, selected_target=selected['target'] if selected else None,
    selected_shift=selected['shift'] if selected else None, original_20pct_reproduced=True,
    seconds=time.monotonic()-START))
ledger('amended_access.json')
print(json.dumps(dict(summary=summary, selected_target=selected['target'] if selected else None), indent=2), flush=True)
