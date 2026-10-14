"""Frozen-cache STEGO training; no labels or validation images are read."""
import os
os.environ['STEGO_STAGE'] = 'training'
from runtime import *
from model import StegoHead, objective
from reference_loss import ContrastiveCorrelationLoss
import argparse

p = argparse.ArgumentParser()
p.add_argument('--seed', required=True, type=int, choices=[0, 1, 2])
a = p.parse_args()
c = read(HERE/'configs/locked.json')
assert c['status'] == 'LOCKED'
assert sha(HERE/'configs/locked.json') == (HERE/'configs/locked.sha256').read_text().strip()
assert not (HERE/'predictions/stageA_manifest.json').exists()
check = read(HERE/'logs/preflight.json')
assert check['passed'] and check['backbone_trainable'] == 0
for name, digest in check['source_sha256'].items():
    assert sha(HERE/name) == digest, 'Code changed after preflight: '+name
seed_all(a.seed)
cfg = SimpleNamespace(**c)
rr = records()
ids = [i for i, r in enumerate(rr) if r['split'] == 'train']
assert len(ids) == 1869
features = np.load(OLD/'B2_448_features.npy', mmap_mode='r')
neighbors = np.load(HERE/'logs/train_neighbors.npy', allow_pickle=False)
assert neighbors.shape == (len(ids), 30)
assert np.all(neighbors != np.arange(len(ids))[:, None])
directory = HERE/f'checkpoints/seed{a.seed}'
directory.mkdir(exist_ok=False)
model = StegoHead()
criterion = ContrastiveCorrelationLoss(cfg)
opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
rng = np.random.default_rng(a.seed)
monitor_rng = np.random.default_rng(c['monitor_seed'])
patients = sorted({rr[i]['patient'] for i in ids}, key=int)
monitor = [int(monitor_rng.choice([j for j, i in enumerate(ids)
            if rr[i]['patient'] == patient])) for patient in patients]
monitor_pos = [int(neighbors[j, monitor_rng.integers(7)]) for j in monitor]

def batch(rows):
    return torch.from_numpy(np.stack([features[ids[int(j)]] for j in rows])).reshape(
        len(rows), 28, 28, 768).permute(0, 3, 1, 2).contiguous()

fixed = batch(monitor), batch(monitor_pos)
best = float('inf')
best_step = None
bad = 0
start = time.monotonic()
order = rng.permutation(len(ids))
cursor = 0
log = HERE/f'logs/seed{a.seed}_train.jsonl'
for step in range(1, cfg.batches+1):
    if cursor+cfg.batch_size > len(order):
        order = rng.permutation(len(ids)); cursor = 0
    selected = order[cursor:cursor+cfg.batch_size]; cursor += cfg.batch_size
    positive = [neighbors[j, rng.integers(cfg.num_neighbors)] for j in selected]
    model.train()
    f, code = model(batch(selected)); fp, cp = model(batch(positive))
    loss, components = objective(criterion, cfg, f, fp, code, cp)
    assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    row = dict(batch=step, loss=float(loss.detach()),
               **{k:float(v.detach()) for k,v in components.items()},
               seconds=time.monotonic()-start)
    if step % cfg.monitor_every == 0 or step == cfg.batches:
        model.eval()
        with torch.random.fork_rng(), torch.no_grad():
            torch.manual_seed(cfg.monitor_seed)
            fm, zm = model(fixed[0]); fp, zp = model(fixed[1])
            value, parts = objective(criterion, cfg, fm, fp, zm, zp)
            z = F.normalize(zm, dim=1)
            std = float(z.flatten(2).std(2).mean())
            # Mean cosine across ALL cross-image patch pairs, not cosine of GAPs.
            means = z.flatten(2).mean(2); cos = means@means.T
            cosine = float(cos[~torch.eye(len(means), dtype=torch.bool)].mean())
        bad = bad+1 if cosine > .99 or std < .01 else 0
        row.update(monitor_loss=float(value), code_std=std,
                   cross_image_cosine=cosine, consecutive_bad_checks=bad,
                   collapse_detected=bad>=3)
        state = dict(state_dict=model.state_dict(), seed=a.seed, step=step,
                     criterion=float(value), config_sha256=sha(HERE/'configs/locked.json'),
                     source_sha256=check['source_sha256'],
                     backbone_sha256=check['backbone_sha256'], train_patients=patients,
                     monitor_indices=[ids[j] for j in monitor])
        if float(value) < best:
            best = float(value); best_step = step; torch.save(state, directory/'best.pt')
        torch.save(state, directory/'last.pt')
        print(json.dumps(row), flush=True)
    with log.open('a') as stream: stream.write(json.dumps(row)+'\n')
dump(f'logs/seed{a.seed}_complete.json', dict(seed=a.seed, batches=cfg.batches,
     best_step=best_step, best_loss=best, labels_opened=False,
     validation_images_used=0, train_patients=patients, seconds=time.monotonic()-start))
ledger(f'logs/seed{a.seed}_training_access.json')
