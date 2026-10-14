"""Clone the prior bounded runner; change only arm-specific partner dispatch."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'experiments/stego_calibrated'; NEW=ROOT/'experiments/stego_colocated'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources={}
for name in ['runtime.py','model.py','reference_loss.py','state.py','stages.py','seal_all.py','evaluate.py']:
    dest=NEW/name;assert not dest.exists();dest.write_bytes((OLD/name).read_bytes());sources[name]=digest(dest)
s=(OLD/'train.py').read_text()
def replace(a,b):
    global s
    assert a in s,a
    s=s.replace(a,b,1)
replace('from state import capture','from state import capture\nfrom loss import ColocatedCorrelationLoss,adjacent_partners')
replace("assert c['candidate']=='D448'", "assert c['candidate'] in ['C448','P1']")
replace("seed_all(a.seed);model=StegoHead();cfg=SimpleNamespace(**c);criterion=ContrastiveCorrelationLoss(cfg)",
"""seed_all(a.seed);model=StegoHead();cfg=SimpleNamespace(**c)
    colocated=c['candidate']=='P1'
    criterion=ColocatedCorrelationLoss(cfg,torch.ones(16,dtype=torch.bool)) if colocated else ContrastiveCorrelationLoss(cfg)
    if colocated:
        metadata=read(ROOT/'audit/scratch/crosspatient/train_records.json');lookup={r['sample_id']:r for r in metadata}
        ordered_records=[lookup[rr[i]['sample_id']] for i in ids]
    def partners(selected,generator):
        if colocated:
            positive,valid,_=adjacent_partners(ordered_records,selected,generator)
            return positive,torch.tensor(valid)
        return [int(neighbors[j,generator.integers(7)]) for j in selected],None""")
replace("positive=[int(neighbors[j,mrng.integers(7)]) for j in monitor];fixed=batch(monitor),batch(positive)",
        "positive,monitor_valid=partners(monitor,mrng);fixed=batch(monitor),batch(positive)")
replace('model.eval()\n        with torch.random.fork_rng()',
        'model.eval()\n        if colocated:criterion.valid=monitor_valid\n        with torch.random.fork_rng()')
replace('positive=[neighbors[j,rng.integers(7)] for j in selected]',
        'positive,batch_valid=partners(selected,rng)\n        if colocated:criterion.valid=batch_valid')
assert not (NEW/'train.py').exists();(NEW/'train.py').write_text(s)
s=(OLD/'check_state.py').read_text().replace("PACKAGE/'D448/stage1/configs/locked.json'","PACKAGE/'P1/stage1/configs/locked.json'")
s=s.replace('from state import capture,restore','from state import capture,restore\nfrom loss import ColocatedCorrelationLoss')
s=s.replace('criterion=ContrastiveCorrelationLoss(cfg);','criterion=ColocatedCorrelationLoss(cfg,torch.ones(2,dtype=torch.bool));')
assert not (NEW/'check_state.py').exists();(NEW/'check_state.py').write_text(s)
result=json.loads((ROOT/'audit/colocated/amended_calibration.json').read_text())
assert result['selected_target'] is not None
selected=next(x for x in result['summary'] if x['target']==result['selected_target']);assert selected['admissible']
(NEW/'logs').mkdir(exist_ok=True)
for arm,shift in [('C448',.785981),('P1',result['selected_shift'])]:
    for stage,budget in [('stage1',150),('stage2',500)]:
        folder=NEW/arm/stage
        for directory in ['configs','logs','checkpoints','predictions','metrics']:(folder/directory).mkdir(parents=True,exist_ok=True)
        cfg=json.loads((OLD/'D448'/stage/'configs/locked.json').read_text())
        cfg.update(candidate=arm,batches=budget,pos_intra_shift=.794418,pos_inter_shift=shift,
            neg_inter_shift=.797651,checkpoint_selection='final state only',
            source_policy='user-amended grid; label-free full-loss selection' if arm=='P1' else 'fixed D448 shifts; control called C448 by user',
            source_commit='e17d2d7',calibration_sha256=digest(ROOT/'audit/colocated/amended_calibration.json'))
        cfg['selected_attraction_target']=result['selected_target'] if arm=='P1' else None
        path=folder/'configs/locked.json';assert not path.exists();path.write_text(json.dumps(cfg,indent=2)+'\n')
        (folder/'configs/locked.sha256').write_text(digest(path)+'\n')
(ROOT/'audit/colocated/runner_reuse.json').write_text(json.dumps(dict(
    unchanged_copies=sources,adapted=['train.py','check_state.py'],
    differences='candidate assertion; P1 partner selection and criterion; same control RNG and loss; P1 state test',
    config_differences='requested rounded shifts, candidate identity, stage budgets, metadata'),indent=2)+'\n')
print('Prepared bounded C448/P1 stages; no training launched')
