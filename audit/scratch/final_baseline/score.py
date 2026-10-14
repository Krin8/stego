"""LOPO mapping and scoring only; no cluster fitting or feature selection."""
from common import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--k',type=int,choices=[8,16],required=True)
p.add_argument('--seed',type=int,choices=[0,1,2],required=True);a=p.parse_args()
tag=f'k{a.k}_s{a.seed}'
assert not (OUT/(tag+'_scores.json')).exists(),'Refusing to overwrite completed scores'
meta=json.loads((OUT/'verification.json').read_text())
grids=np.load(OUT/(tag+'_grids.npy'),mmap_mode='r')
patients=sorted(set(meta['patients']),key=int)
confs={patient:np.zeros((a.k,2),dtype=np.int64) for patient in patients}
inv=json.loads((FORE/'inventory.json').read_text())
model,ck=load(inv['seed0']['path']);offset=0
val_labels=np.load(FORE/'val_labels.npy',mmap_mode='r')
for split in ['train','val']:
 ds=dataset(model.cfg,split)
 assert [r['sample_id'] for r in ds.dataset.records]==meta['ordered_sample_ids'][offset:offset+len(ds)]
 for batch in DataLoader(ds,batch_size=16,shuffle=False,num_workers=0):
  idx=batch['ind'].numpy();y=batch['label'].numpy()
  assert np.isin(y,[0,1]).all(),'Unexpected or ignored labels'
  if split=='val':assert np.array_equal(y,val_labels[idx])
  pr=F.interpolate(torch.from_numpy(np.array(grids[offset+idx])).float()[:,None],
                   size=y.shape[-2:],mode='nearest')[:,0].numpy().astype(np.uint8)
  for j,i in enumerate(idx):
   patient=meta['patients'][offset+int(i)]
   confs[patient]+=confusion(pr[j],y[j],a.k)
 offset+=len(ds)
assert offset==2341
total=sum(confs.values());assert int(total.sum())==2341*224*224
result=dict(script=__file__,k=a.k,seed=a.seed,
 labels_used='Other 19 patients for mapping; held-out patient only for scoring. All 20 patient folds.',
 patient_confusions={patient:c.tolist() for patient,c in confs.items()},lopo={})
for patient,c in confs.items():
 fitted=mapping_any_k(total-c)
 if a.k==8:assert fitted==mapping_report(total-c)
 others=[q for q in patients if q!=patient];assert len(others)==19 and patient not in others
 result['lopo'][patient]=dict(slice_count=meta['patient_counts'][patient],
  mapping_fit_patients=others,mapping_fit_pixels=int((total-c).sum()),
  heldout_pixels=int(c.sum()),liver_pixels=int(c[:,1].sum()),
  optimal=scores(c,fitted['optimal']['clusters']),majority=scores(c,fitted['majority']['clusters']))
 print(tag,'patient',patient,'optimal',result['lopo'][patient]['optimal']['iou'],flush=True)
def stats(v):
 v=np.asarray(v,dtype=float)
 return dict(mean=float(v.mean()),std=float(v.std(ddof=1)),min=float(v.min()),max=float(v.max()),median=float(np.median(v)))
result['summary']={mapping:{metric:stats([r[mapping][metric] for r in result['lopo'].values()])
 for metric in ['iou','dice']} for mapping in ['optimal','majority']}
result['elapsed_seconds']=elapsed();dump(tag+'_scores.json',result)
note('LOPO evaluation completed: `'+tag+'_scores.json` from `score.py --k '+str(a.k)+' --seed '+str(a.seed)+'`. '
     'Mapping labels: other 19 patients only; scoring labels: held-out patient only. '+json.dumps(result['summary']))
print(json.dumps(result['summary']),flush=True)
