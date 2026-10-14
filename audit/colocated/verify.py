"""Predeclared P1 shift and strict term/full-loss admissibility, no training."""
from common import *

SEED = 20261003
sample_record = read(ROOT / 'audit/preflight/sample.json')
anchors = np.array(sample_record['local_indices'])
assert len(anchors) == 200 and len({RR[TRAIN[i]]['patient'] for i in anchors}) == 16
metadata = read(ROOT / 'audit/scratch/crosspatient/train_records.json')
lookup = {r['sample_id']: r for r in metadata}
records = [lookup[RR[i]['sample_id']] for i in TRAIN]
partners, valid, offsets = adjacent_partners(records, anchors, np.random.default_rng(SEED + 1))
knn = read(ROOT / 'audit/preflight/pair_plan.json')['positive_local_indices']
centers = torch.from_numpy(np.load(ROOT / 'audit/final_pipeline/inductive_seed0_centers.npy', allow_pickle=False))
# Reuse the exact preflight constant/random/K8 code construction, no label mapping.
tree = ast.parse((ROOT / 'audit/preflight/diagnose.py').read_text())
node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'codes')
exec(compile(ast.Module(body=[node], type_ignores=[]), 'preflight codes', 'exec'), globals())
loss_keys = ['feature_samples', 'neg_samples', 'pointwise', 'zero_clamp', 'stabalize',
    'use_salience', 'pos_intra_shift', 'pos_inter_shift', 'neg_inter_shift',
    'pos_intra_weight', 'pos_inter_weight', 'neg_inter_weight', 'correspondence_weight',
    'aug_alignment_weight', 'num_neighbors']
dump('protocol.json', dict(seed=SEED, labels_used=False, threads=2,
    loss_cfg={k: CFG[k] for k in loss_keys},
    requested_training=dict(seeds=[0, 1, 2], stage1_batches=150, stage2_batches=500,
        batch_size=16, optimizer='Adam', lr=.0005, scheduler=None, dropout=.1,
        head='Conv(768,70)+Conv(768,768)->ReLU->Conv(768,70)',
        monitor_every=50, healthy_cosine_less_than=.99, healthy_std_greater_than=.01,
        screen_requires_final_cosine_below_initial=True, checkpoint_selection='final only',
        restart='from scratch if interrupted; no resume', alignment=False),
    anchors_local=anchors.tolist(), partners_local=partners, valid=valid, offsets=offsets,
    control_partners_local=knn, target_attraction=.2,
    calibration='linear 0.8 quantile of all valid sampled centered diagonal affinities',
    code_construction='unchanged codes() AST from audit/preflight/diagnose.py',
    diagnostic_dropout=False, sample_coordinates='11x11; inherited bilinear sampler',
    alignment='same normalized coordinate in independent saved body boxes; physical ordering within patient/series; no registration',
    boundary_rule='uniform choice among -2,-1,+1,+2; unavailable draw omitted from partner only',
    gate='P1 partner AND P1 full loss: constant > random > K8; otherwise stop',
    sources={str(p.relative_to(ROOT)): sha(p) for p in [
        ROOT / 'experiments/stego_colocated/loss.py', ROOT / 'experiments/stego_biomedclip/reference_loss.py',
        ROOT / 'audit/preflight/sample.json', ROOT / 'audit/preflight/pair_plan.json',
        ROOT / 'audit/final_pipeline/inductive_seed0_centers.npy',
        ROOT / 'src/configs/chaos_splits_v2.json']}))


class Capture(ColocatedCorrelationLoss):
    def partner(self, fa, fb, za, zb):
        ones = torch.ones_like(za[:, :1])
        losses, _ = self.helper(fa, fb, ones, ones, 0.)
        self.affinity = (-losses).flatten(1, 2).flatten(2, 3).diagonal(dim1=1, dim2=2).numpy().copy()
        return super().partner(fa, fb, za, zb)


affinities = []
for start in range(0, 200, 16):
    f = feature_batch(TRAIN[anchors[start:start + 16]])
    fp = feature_batch(TRAIN[np.array(partners[start:start + 16])])
    crit = Capture(SimpleNamespace(**CFG), torch.tensor(valid[start:start + 16]))
    torch.manual_seed(SEED + start)
    objective(crit, crit.cfg, f, fp, codes(f, 'constant'), codes(fp, 'constant'))
    affinities.append(crit.affinity.ravel())
    print('CALIBRATE', start, flush=True)
affinity = np.concatenate(affinities)
shift = float(np.quantile(affinity, .8))
attraction = float((affinity > shift).mean())
dump('calibration.json', dict(shift=shift, attraction=attraction, pairs=len(affinity),
    valid_samples=sum(valid), skipped_samples=len(valid) - sum(valid),
    affinity_mean=float(affinity.mean()), source='verify.py', labels_used=False))

rows = []
for start in range(0, 200, 16):
    f = feature_batch(TRAIN[anchors[start:start + 16]])
    for arm, pi, partner_shift in [('C448', knn, .785981), ('P1', partners, shift)]:
        fp = feature_batch(TRAIN[np.array(pi[start:start + 16])])
        cfg = SimpleNamespace(**{**CFG, 'pos_inter_shift': partner_shift})
        criterion = (ColocatedCorrelationLoss(cfg, torch.tensor(valid[start:start + 16]))
                     if arm == 'P1' else ContrastiveCorrelationLoss(cfg))
        torch.manual_seed(SEED + start + 10)
        pairs = {kind: (codes(f, kind), codes(fp, kind)) for kind in ['constant', 'random', 'cluster']}
        losses = {}
        for kind, (z, zp) in pairs.items():
            torch.manual_seed(SEED + start)
            total, parts = objective(criterion, cfg, f, fp, z, zp)
            losses[kind] = dict(total=float(total), **{k: float(v) for k, v in parts.items()})
        rows.append(dict(arm=arm, start=start, slices=len(f),
                         valid=sum(valid[start:start + 16]) if arm == 'P1' else len(f), losses=losses))
    print('LOSSES', start, flush=True)
summary = {}
for arm in ['C448', 'P1']:
    rr = [r for r in rows if r['arm'] == arm]
    summary[arm] = {kind: {term: sum(r['slices'] * r['losses'][kind][term] for r in rr) / 200
        for term in ['total', 'pos_intra', 'pos_inter', 'neg_inter']} for kind in ['constant', 'random', 'cluster']}
    summary[arm]['partner_valid_pair_weighted'] = {kind: sum(r['valid'] * r['losses'][kind]['pos_inter'] for r in rr) / sum(r['valid'] for r in rr)
        for kind in ['constant', 'random', 'cluster']}
p = summary['P1']
checks = {term: p['constant'][term] > p['random'][term] > p['cluster'][term]
          for term in ['pos_inter', 'total']}
checks['attraction_about_20pct'] = abs(attraction - .2) < .001
dump('verification.json', dict(source='verify.py', labels_used=False, shift=shift,
    attraction=attraction, rows=rows, summary=summary, checks=checks,
    admissible=all(checks.values()), seconds=time.monotonic() - START))
ledger('verify_access.json')
print(json.dumps(dict(shift=shift, attraction=attraction, summary=summary, checks=checks), indent=2), flush=True)
