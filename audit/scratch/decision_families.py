"""Count processed families; compare20 seeded current CT slices to legacy views."""
import decision_guard as g
import random,json,collections
from pathlib import Path
import numpy as np,pydicom,SimpleITK as sitk
from PIL import Image
base=g.ROOT/'src/pytorch_data_dir/preprocessed';rawroot=g.ROOT/'src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets/CT'
files=[];counts=collections.Counter();patients=collections.defaultdict(set)
for split in ['train','val']:
    for p in sorted((base/split/'images').glob('*.png')):
        parts=p.stem.split('_');mod,pid=parts[:2];family='legacy' if pid.startswith('p') else 'current'
        counts[(split,mod,family)]+=1;patients[(split,mod)].add(pid.lstrip('p').lstrip('0') or '0')
        if mod=='CT' and family=='current':files.append(p)
selected=random.Random(20260930).sample(files,20);cache={};observations=[]
for p in selected:
    _,pid,z=p.stem.split('_');z=int(z);folder=rawroot/pid/'DICOM_anon'
    if pid not in cache:
        rawfiles=sorted(folder.glob('*.dcm'));heads={x:pydicom.dcmread(x,stop_before_pixels=True) for x in rawfiles}
        ids=sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder));series=[sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder),i) for i in ids];ordered=list(map(Path,max(series,key=len)))
        h=heads[ordered[0]];orient=np.asarray(h.ImageOrientationPatient,float);normal=np.cross(orient[:3],orient[3:]);positions={x:float(np.dot(np.asarray(v.ImagePositionPatient,float),normal)) for x,v in heads.items()}
        legacy=sorted((p.parent).glob('CT_p'+pid+'_*.png'))
        cache[pid]=(rawfiles,ordered,positions,legacy)
    rawfiles,ordered,positions,legacy=cache[pid]
    proposed_position=positions[ordered[0]]+z # generator target spacing1mm in normal direction
    closest=min(rawfiles,key=lambda x:abs(positions[x]-proposed_position));rank=rawfiles.index(closest)
    if len(legacy)!=len(rawfiles):observations.append(dict(current=str(p),status='UNVERIFIED count mismatch'));continue
    old=legacy[rank]
    a=Image.open(p).convert('L');b=Image.open(old).convert('L');aa=np.array(a.resize((256,256),Image.Resampling.BILINEAR),dtype=float);bb=np.array(b.resize((256,256),Image.Resampling.BILINEAR),dtype=float)
    ma=np.array(Image.open(p.parent.parent/'labels'/p.name).resize((256,256),Image.Resampling.NEAREST))>0
    mb=np.array(Image.open(old.parent.parent/'labels'/old.name).resize((256,256),Image.Resampling.NEAREST))>0
    union=(ma|mb).sum();iou=float((ma&mb).sum()/union) if union else None
    r=dict(current=str(p.relative_to(g.ROOT)),legacy_candidate=str(old.relative_to(g.ROOT)),candidate_raw=str(closest.relative_to(g.ROOT)),pair_rule='hypothesis: nearest physical slice for current1mm index; legacy rank in same patient',current_shape=a.size,legacy_shape=b.size,current_percentiles=np.percentile(aa,[0,1,50,99,100]).tolist(),legacy_percentiles=np.percentile(bb,[0,1,50,99,100]).tolist(),pixel_mae_0_255=float(np.abs(aa-bb).mean()),pixel_correlation=float(np.corrcoef(aa.ravel(),bb.ravel())[0,1]),exact_pixels=bool(np.array_equal(aa,bb)),mask_iou=iou,position_error_mm=abs(positions[closest]-proposed_position),raw_count=len(rawfiles),current_count=len(list(p.parent.glob('CT_'+pid+'_*.png'))),legacy_count=len(legacy))
    observations.append(r)
result=dict(counts={str(k):v for k,v in counts.items()},patient_ids={str(k):sorted(v,key=int) for k,v in patients.items()},train_val_patient_overlap={mod:sorted(patients[('train',mod)]&patients[('val',mod)]) for mod in ['CT','T1DUAL','T2SPIR']},comparisons=observations)
g.append('## M5 — Processed-family counts and20-pair comparison\n\nCommand: `python3 -B audit/scratch/decision_families.py`; seed20260930. Images compared after bilinear resize to256×256; masks use nearest. Pairing is an explicit hypothesis, not a recovered conversion manifest: current index maps to generator1mm geometry, legacy file rank maps to raw filename rank within the same patient. Headers/GDCM order only were read for source geometry; no additional raw image pixels were sampled.\n\n```json\n'+json.dumps(result,indent=2)+'\n```')
print(json.dumps(dict(counts=result['counts'],overlap=result['train_val_patient_overlap'],exact_pairs=sum(r.get('exact_pixels',False) for r in observations),median_mae=float(np.median([r['pixel_mae_0_255'] for r in observations if 'pixel_mae_0_255' in r])),correlations=[round(r['pixel_correlation'],4) for r in observations if 'pixel_correlation' in r],mask_ious=[None if r['mask_iou'] is None else round(r['mask_iou'],4) for r in observations if 'mask_iou' in r],examples=observations[:2])))
