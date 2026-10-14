"""Cached-weight refinements, reference reproduction, and per-patient diagnostics."""
from common import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[0,1,2],required=True);a=p.parse_args()
assert not (OUT/f'seed{a.seed}.json').exists(),'Refusing to overwrite completed run'
meta=metadata();records=meta['records'];patients=sorted(meta['patient_counts'],key=int)
original=json.loads((OLD/f'B2_448_s{a.seed}_scores.json').read_text())
grids=np.load(OLD/f'B2_448_s{a.seed}_grids.npy',mmap_mode='r')
weights=np.load(OLD/f'B4_s{a.seed}_weights.npy',mmap_mode='r')
assert json.loads((OLD/f'B4_s{a.seed}_weights.json').read_text())['parent']=='B2_448'
truth=np.load(OLD/'scoring_masks_native.npy',mmap_mode='r');small=np.load(OLD/'scoring_masks_224.npy',mmap_mode='r')
variants=['B2_hard','B5_2D','F1','F2']
dest={v:np.lib.format.open_memmap(OUT/f'{v}_s{a.seed}_pred.npy',mode='w+',dtype='uint8',shape=(2341,512,512)) for v in variants}
conf={v:{str(res):{pat:np.zeros((2,2),np.int64) for pat in patients} for res in [224,512]} for v in variants}
diag={v:{} for v in variants};losses={};slice_scores={v:[None]*2341 for v in variants};boundary={v:{} for v in variants}
yy,xx=np.mgrid[-3:4,-3:4];disk=xx*xx+yy*yy<=9
structure=np.ones((3,3),int)
def loss_stats(before,after,gt,lab,n):
 removed=int((before & ~after & gt).sum())
 before_counts=np.bincount(lab[before & gt],minlength=n+1)[1:]
 after_counts=np.bincount(lab[after & gt],minlength=n+1)[1:]
 return dict(removed_true_positive_pixels=removed,
  previously_hit_components=int((before_counts>0).sum()),
  previously_hit_components_entirely_lost=int(((before_counts>0)&(after_counts==0)).sum()))
for patient in patients:
 indices=sorted([i for i,r in enumerate(records) if r['patient']==patient],key=lambda i:records[i]['physical_position'])
 assert len(indices)==meta['patient_counts'][patient]
 assert len({records[i]['physical_position'] for i in indices})==len(indices)
 subsets={tuple(original['results'][str(res)]['folds'][patient][m]['clusters']) for res in [224,512] for m in ['optimal','majority']}
 assert len(subsets)==1;subset=list(subsets.pop());lookup=np.zeros(9,bool);lookup[subset]=True
 for res in [224,512]:assert set(original['results'][str(res)]['folds'][patient]['mapping_patients'])==set(patients)-{patient}
 multicomponent=[];empty=[];component_loss=[];pre_lcc=[]
 for i in indices:
  r=records[i];gt=truth[i].astype(bool);lab,n=ndi.label(gt,structure=structure)
  if n>=2:multicomponent.append(i)
  if not gt.any():empty.append(i)
  hard=lookup[native_prediction(grids[i],r,'B2_448').astype(np.uint8)]
  b5=largest(hard,structure)
  field=weights[i][...,subset].sum(-1) if subset else np.zeros(weights.shape[1:3],np.float32)
  soft=native_prediction(field,r,'B2_448',mode='bilinear',outside=0)
  guided_binary=guided(raw_image(r),soft)>=.5
  lcc=largest(guided_binary,structure)
  filled=ndi.binary_fill_holes(lcc)
  closed=ndi.binary_closing(np.pad(filled,3),structure=disk)[3:-3,3:-3]
  for v,pred in [('B2_hard',hard),('B5_2D',b5),('F1',closed)]:dest[v][i]=pred
  if n>=2:
   component_loss.append(dict(index=i,sample_id=r['sample_id'],gt_components=n,gt_pixels=int(gt.sum()),
    gt_component_sizes=np.bincount(lab.ravel())[1:].tolist(),
    hard_to_B5=loss_stats(hard,b5,gt,lab,n),guided_to_lcc=loss_stats(guided_binary,lcc,gt,lab,n)))
 f1=np.asarray(dest['F1'][indices],dtype=bool)
 # Only the slice axis is filtered. Endpoint slices are replicated within this patient.
 dest['F2'][indices]=ndi.median_filter(f1,size=(3,1,1),mode='nearest')
 losses[patient]=dict(multicomponent_indices=multicomponent,empty_gt_indices=empty,details=component_loss)
 bd={v:[] for v in variants};cases={v:dict(both_empty=0,one_empty=0,both_present=0) for v in variants}
 for v in variants:diag[v][patient]=dict(empty_gt_slices=len(empty),empty_fp_pixels=[],empty_fp_indices=[],multicomponent=[])
 for i in indices:
  gt=truth[i].astype(bool);tp_points=np.argwhere(surface(gt))
  for v in variants:
   pred=dest[v][i].astype(bool);c=confusion(pred,gt,2);s=scores(c,[1]);slice_scores[v][i]=s
   conf[v]['512'][patient]+=c;conf[v]['224'][patient]+=confusion(resize224(pred),small[i],2)
   distances,case=boundary_distances(pred,gt,tp_points);bd[v].append(distances);cases[v][case]+=1
   if i in empty:
    fp=int(pred.sum());diag[v][patient]['empty_fp_pixels'].append(fp)
    if fp:diag[v][patient]['empty_fp_indices'].append(i)
   if i in multicomponent:diag[v][patient]['multicomponent'].append(dict(index=i,**s))
 for v in variants:boundary[v][patient]=dict(**boundary_summary(bd[v]),cases=cases[v])
 print('seed',a.seed,'patient',patient,'seconds',elapsed(),flush=True)
result=dict(script=__file__,seed=a.seed,labels_used='Inherited other19 mappings; heldout scoring and diagnostics only. No label-based parameter choice.',
 stages='soft -> guided -> threshold.5 -> LCC8 -> fill holes4 -> disk3 closing -> F2 only median(3,1,1) nearest endpoints',
 forward_passes=0,kmeans_fits=0,variants={},component_loss=losses,elapsed_seconds=elapsed())
for v in variants:
 dest[v].flush();rr=dict(results={},diagnostics=diag[v],boundary=boundary[v],slice_scores=slice_scores[v])
 for res in ['224','512']:
  per={pat:scores(c,[1]) for pat,c in conf[v][res].items()}
  rr['results'][res]=dict(patients=per,summary={metric:patient_stats([s[metric] for s in per.values()]) for metric in ['iou','dice']})
 if v in ['B2_hard','B5_2D']:
  oldname='B2_448' if v=='B2_hard' else 'B5_2D';old=json.loads((OLD/f'{oldname}_s{a.seed}_scores.json').read_text())
  error=max(abs(rr['results'][res]['patients'][pat][metric]-old['results'][res]['folds'][pat][m][metric]) for res in ['224','512'] for pat in patients for metric in ['iou','dice'] for m in ['optimal','majority'])
  assert error==0,(v,error);rr['reference_max_metric_difference']=error
 result['variants'][v]=rr
dump(f'seed{a.seed}.json',result)
print('DONE',a.seed,elapsed(),flush=True)
