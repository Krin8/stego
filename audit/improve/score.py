"""Strict LOPO hard-cluster scoring, with forced background outside crop."""
from pipeline import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
tag=f'{a.variant}_s{a.seed}';meta=metadata();records=meta['records']
assert (OUT/'ceiling.json').is_file(),'Scoring masks are not finalized'
grids=np.load(OUT/f'{tag}_grids.npy',mmap_mode='r')
labels={res:np.load(OUT/f'scoring_masks_{"native" if res==512 else 224}.npy',mmap_mode='r') for res in [224,512]}
confs={res:{p:np.zeros((9,2),np.int64) for p in meta['patient_counts']} for res in [224,512]}
for i,r in enumerate(records):
 native=native_prediction(grids[i],r,a.variant).astype(np.uint8)
 # B0 goes directly 14->224 to reproduce the locked baseline exactly.
 small=resize224(grids[i]) if a.variant=='B0' else resize224(native)
 for res,pred in [(224,small),(512,native)]:confs[res][r['patient']]+=confusion(pred,labels[res][i],9)
result=dict(script=__file__,variant=a.variant,seed=a.seed,
 labels_used='Other19 patients for mapping; heldout patient only for scoring, independently at each scoring resolution.',
 results={str(res):results_from_conf(cs) for res,cs in confs.items()},elapsed_seconds=elapsed())
if a.variant=='B0':
 old=json.loads((FINAL/f'k8_s{a.seed}_scores.json').read_text())
 error=max(abs(result['results']['224']['folds'][patient][m][metric]-old['lopo'][patient][m][metric])
  for patient in old['lopo'] for m in ['optimal','majority'] for metric in ['iou','dice'])
 result['baseline_max_patient_metric_difference']=error
 assert error<1e-5,('B0 baseline mismatch',error)
dump(f'{tag}_scores.json',result);print(tag,result['results']['224']['summary'],flush=True)
