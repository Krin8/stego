"""Annotation-derived grid approximations and scoring masks; no fitting."""
from pipeline import *
from data import CHAOS
meta=metadata();records=meta['records'];patients=list(meta['patient_counts'])
native=np.lib.format.open_memmap(OUT/'scoring_masks_native.npy',mode='w+',dtype='uint8',shape=(2341,512,512))
small=np.lib.format.open_memmap(OUT/'scoring_masks_224.npy',mode='w+',dtype='uint8',shape=(2341,224,224))
confs={f'{g}_{kind}_{res}':{p:np.zeros((2,2),np.int64) for p in patients}
 for g in [14,21,28,56] for kind in ['majority','mean_bilinear'] for res in [224,512]}
largest={};positions={}
for g in [14,21,28,56]:
 y=np.arange(512)*g//512;cell=y[:,None]*g+y[None,:]
 positions[g]=(cell,np.bincount(cell.ravel(),minlength=g*g))
for i,r in enumerate(records):
 lab=CHAOS._load_mask(r['mask'],'CT');truth=np.array(lab);assert np.isin(truth,[0,1]).all()
 native[i]=truth;small[i]=get_chaos_transform(224,True,'center')(lab).squeeze(0).numpy()
 if r['split']=='val':
  count=int(truth.sum());old=largest.get(r['patient'])
  if old is None or count>old['pixels']:largest[r['patient']]=dict(index=i,pixels=count)
 for g in [14,21,28,56]:
  cell,counts=positions[g];fraction=np.bincount(cell.ravel(),weights=truth.ravel(),minlength=g*g)/counts
  majority=(fraction>0.5)[cell]
  field=F.interpolate(torch.tensor(fraction.reshape(g,g),dtype=torch.float32)[None,None],(512,512),mode='bilinear',align_corners=False)[0,0].numpy()
  for kind,pred in [('majority',majority),('mean_bilinear',field>=0.5)]:
   for res in [224,512]:
    pr=resize224(pred) if res==224 else pred;y=small[i] if res==224 else truth
    confs[f'{g}_{kind}_{res}'][r['patient']]+=confusion(pr,y,2)
 if i%300==0:print('ceilings',i,elapsed(),flush=True)
native.flush();small.flush()
old=np.load(FORE/'val_labels.npy',mmap_mode='r');assert np.array_equal(small[1869:],old)
rows=[]
baseline=json.loads((FINAL/'summary.json').read_text())['summary'][0]['iou']['mean']
for tag,cs in confs.items():
 g,kind,res=tag.split('_',1)[0],None,None
 g=int(tag.split('_')[0]);res=int(tag.split('_')[-1]);kind='_'.join(tag.split('_')[1:-1])
 per={p:scores(c,[1]) for p,c in cs.items()}
 stats={metric:patient_stats([s[metric] for s in per.values()]) for metric in ['iou','dice']}
 rows.append(dict(grid=g,method=kind,resolution=res,patients=per,summary=stats,
  gap_to_100_pp=100*(1-stats['iou']['mean']),
  fraction_of_224_baseline_gap=(1-stats['iou']['mean'])/(1-baseline) if res==224 else None))
dump('ceiling.json',dict(script=__file__,labels_used='All masks directly construct grid approximations and score them; ceiling only, not a method',rows=rows,elapsed_seconds=elapsed()))
dump('figure_selection.json',dict(script=__file__,fixed_label_free=meta['fixed'],largest_liver_validation=largest,
 eight_indices=meta['fixed']+[largest[p]['index'] for p in sorted(largest,key=int)],labels_used='Largest-liver slice selection uses validation masks; fixed midpoint slices do not'))
print('DONE',elapsed(),flush=True)
