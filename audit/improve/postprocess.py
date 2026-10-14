"""Fixed exploratory descendants; mappings inherited from the other19-parent folds."""
from pipeline import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
meta=metadata();records=meta['records'];patients=list(meta['patient_counts'])
parent=json.loads((OUT/'parent_selection.json').read_text())['parent']
original=json.loads((OUT/f'{parent}_s{a.seed}_scores.json').read_text())
grids=np.load(OUT/f'{parent}_s{a.seed}_grids.npy',mmap_mode='r')
weights=np.load(OUT/f'B4_s{a.seed}_weights.npy',mmap_mode='r')
labels={res:np.load(OUT/f'scoring_masks_{"native" if res==512 else 224}.npy',mmap_mode='r') for res in [224,512]}
variants=['B4_soft','B4_guided','B5_2D','B5_3D']
cs={v:{res:{m:{p:np.zeros((2,2),np.int64) for p in patients} for m in ['optimal','majority']} for res in [224,512]} for v in variants}
selected=json.loads((OUT/'figure_selection.json').read_text())['eight_indices'];panels={}
def largest(mask,structure):
 lab,n=ndi.label(mask,structure=structure)
 if not n:return mask
 counts=np.bincount(lab.ravel());counts[0]=0
 return lab==counts.argmax()
def guided(image,field):
 size=17;mi=ndi.uniform_filter(image,size=size,mode='reflect');mp=ndi.uniform_filter(field,size=size,mode='reflect')
 var=ndi.uniform_filter(image*image,size=size,mode='reflect')-mi*mi
 cov=ndi.uniform_filter(image*field,size=size,mode='reflect')-mi*mp
 aa=cov/(var+.01);bb=mp-aa*mi
 return np.clip(ndi.uniform_filter(aa,size=size,mode='reflect')*image+ndi.uniform_filter(bb,size=size,mode='reflect'),0,1)
for patient in patients:
 indices=sorted([i for i,r in enumerate(records) if r['patient']==patient],key=lambda i:records[i]['physical_position'])
 assert len({records[i]['physical_position'] for i in indices})==len(indices)
 rawclusters=[native_prediction(grids[i],records[i],parent).astype(np.uint8) for i in indices]
 prediction_cache={}
 for res in [224,512]:
  for mapping in ['optimal','majority']:
   subset=original['results'][str(res)]['folds'][patient][mapping]['clusters'];lookup=np.zeros(9,bool);lookup[subset]=True
   key=tuple(subset)
   if key not in prediction_cache:
    volume=np.stack([lookup[c] for c in rawclusters]);filtered3=largest(volume,ndi.generate_binary_structure(3,1))
    cached=[]
    for j,i in enumerate(indices):
     image=raw_image(records[i]);field=weights[i][...,subset].sum(-1) if subset else np.zeros(weights.shape[1:3],np.float32)
     soft=native_prediction(field,records[i],parent,mode='bilinear',outside=0)
     cached.append(dict(B4_soft=soft>=.5,B4_guided=guided(image,soft)>=.5,
       B5_2D=largest(volume[j],np.ones((3,3),int)),B5_3D=filtered3[j]))
    prediction_cache[key]=cached
   for j,i in enumerate(indices):
    predictions=prediction_cache[key][j]
    for variant,native in predictions.items():
     pr=resize224(native) if res==224 else native
     cs[variant][res][mapping][patient]+=confusion(pr,labels[res][i],2)
     if a.seed==0 and i in selected:
      panels[f'{variant}_{res}_{mapping}_{i}']=native.astype(np.uint8)
 print('post',a.seed,patient,elapsed(),flush=True)
for variant in variants:
 result=dict(script=__file__,variant=variant,seed=a.seed,parent=parent,exploratory=True,
  labels_used='Parent selection uses cohort LOPO scores; each inherited mapping uses other19 labels, heldout labels only score; filters use image/prediction only',results={})
 for res in [224,512]:
  folds={patient:{m:scores(cs[variant][res][m][patient],[1]) for m in ['optimal','majority']} for patient in patients}
  for patient in patients:
   for m in ['optimal','majority']:folds[patient][m]['parent_liver_clusters']=original['results'][str(res)]['folds'][patient][m]['clusters']
   folds[patient]['mapping_patients']=[p for p in patients if p!=patient]
  result['results'][str(res)]=dict(folds=folds,
    summary={m:{metric:patient_stats([f[m][metric] for f in folds.values()]) for metric in ['iou','dice']} for m in ['optimal','majority']})
 result['elapsed_seconds']=elapsed();dump(f'{variant}_s{a.seed}_scores.json',result)
if panels:np.savez_compressed(OUT/'postprocess_figure_predictions.npz',**panels)
print('DONE',a.seed,elapsed(),flush=True)
