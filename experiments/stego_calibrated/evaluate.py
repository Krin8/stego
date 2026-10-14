"""Scoring only, after every selected run's predictions are globally sealed."""
import os
os.environ['STEGO_STAGE']='evaluation'
from runtime import *
import runtime as rt
from datetime import datetime,timezone

global_seal=read(PACKAGE/'all_predictions_sealed.json')
assert sha(PACKAGE/'all_predictions_sealed.json')==(PACKAGE/'all_predictions_sealed.sha256').read_text().strip()
for path,digest in global_seal['outputs'].items():assert sha(PACKAGE/path)==digest,path
for path,digest in global_seal['source_sha256'].items():assert sha(PACKAGE/path)==digest,path
rt.reuse(ROOT/'audit/scratch/forensics/common.py',['scores']);rt.reuse(OLD/'pipeline.py',['patient_stats'])
scores=rt.scores;patient_stats=rt.patient_stats
assert not (HERE/'metrics/results.json').exists()
dump('metrics/label_access_start.json',dict(utc=datetime.now(timezone.utc).isoformat(),global_seal_sha256=sha(PACKAGE/'all_predictions_sealed.json')))
# All annotation access is below the global seal verification. Materialize validation rows only.
y={r:np.load(OLD/f'scoring_masks_{"native" if r==512 else "224"}.npy',mmap_mode='r') for r in [224,512]}
rr=records();results={}
for seed in [0,1,2]:
    out=HERE/f'predictions/seed{seed}';m=read(out/'prediction_manifest.json');indices=m['indices']
    assert all(rr[i]['split']=='val' for i in indices)
    patients=sorted({rr[i]['patient'] for i in indices},key=int);run=dict(seed=seed,selected=m['selected'],rules={})
    for rule in ['N1','N2','N3']:
        run['rules'][rule]={}
        for kind in ['hard','F1']:
            masks=np.load(out/f'{rule}_{kind}.npy',mmap_mode='r');scales={}
            for resolution in [224,512]:
                conf={p:np.zeros((2,2),np.int64) for p in patients};per={p:[] for p in patients}
                for j,i in enumerate(indices):
                    pred=masks[j] if resolution==512 else resize224(masks[j])
                    co=np.bincount((np.asarray(pred,np.int64)*2+np.asarray(y[resolution][i],np.int64)).ravel(),minlength=4).reshape(2,2)
                    conf[rr[i]['patient']]+=co;per[rr[i]['patient']].append(dict(index=i,**scores(co,[1])))
                ps={p:dict(slices=len(per[p]),pooled=scores(conf[p],[1]),mean_slice={metric:float(np.mean([s[metric] for s in per[p]])) for metric in ['iou','dice']},slice_scores=per[p]) for p in patients}
                summary={agg:{metric:patient_stats([v[agg][metric] for v in ps.values()]) for metric in ['iou','dice']} for agg in ['pooled','mean_slice']}
                scales[str(resolution)]=dict(patients=ps,summary=summary)
            run['rules'][rule][kind]=scales
    results[str(seed)]=run
primary=[100*results[str(s)]['rules']['N3']['F1']['224']['summary']['pooled']['iou']['mean'] for s in [0,1,2]]
mean=float(np.mean(primary));worst=min(primary)
outcome='success' if mean>=80.03 and worst>=77.96 else ('partial' if mean>77.96 else 'failure')
dump('metrics/results.json',dict(results=results,primary_rule='N3',primary_scale=224,primary_aggregation='patient-pooled then patient mean then seed mean',
  primary_seed_iou=primary,mean_seed_iou=mean,worst_seed_iou=worst,outcome=outcome,
  criterion='Success: mean >=80.03 AND no seed <77.96; partial: mean >77.96 but not success; otherwise failure',
  global_seal_sha256=sha(PACKAGE/'all_predictions_sealed.json'),label_use='validation scoring only; no label mapping or selection'))
ledger('logs/evaluation_access.json');print('Stage B complete:',outcome,primary,flush=True)
