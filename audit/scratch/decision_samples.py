"""Seeded20 slices per CT/T1 phase/T2 sequence; no data writes, six overlays."""
import decision_guard as g
import random,json,collections
from pathlib import Path
import numpy as np,pydicom
from PIL import Image,ImageDraw
SEED=20260930
BASE=g.ROOT/'src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets'
Q=[.5,1,5,50,95,99,99.5]
groups={k:[] for k in ['CT','T1-in','T1-out','T2']}; counts=[]
for patient in sorted((BASE/'CT').iterdir()):
    ds=sorted((patient/'DICOM_anon').glob('*.dcm')); ms=sorted((patient/'Ground').glob('*.png'))
    if ds:counts.append(('CT',patient.name,len(ds),len(ms)))
    for i,d in enumerate(ds):
        if i<len(ms):groups['CT'].append((d,ms[i],patient.name,i))
for patient in sorted((BASE/'MR').iterdir()):
    for name,seq,phase in [('T1-in','T1DUAL','InPhase'),('T1-out','T1DUAL','OutPhase'),('T2','T2SPIR','')]:
        ds=sorted((patient/seq/'DICOM_anon'/phase).glob('*.dcm'));ms=sorted((patient/seq/'Ground').glob('*.png'))
        if ds:counts.append((name,patient.name,len(ds),len(ms)))
        for i,d in enumerate(ds):
            if i<len(ms):groups[name].append((d,ms[i],patient.name,i))
rng=random.Random(SEED); selection={k:rng.sample(v,min(20,len(v))) for k,v in groups.items()}
rows=['## M5 — Seeded raw-slice intensity and pairing evidence',f'Command: `python3 -B audit/scratch/decision_samples.py`; seed={SEED};20 slices per sequence (T1 phases sampled separately). Percentiles are in order{Q}. CT values use slope/intercept; MRI values are stored pixel intensities. Pairing follows the current loader: sorted DICOM filenames zipped to sorted mask filenames. Thus intensity/organ statistics are conditional on that pairing; no anatomical label identity is inferred from a numeric value.','### Series counts (DICOM, PNG)','`'+str(counts)+'`']
overlays=[]; summary=[]; overlay_dir=g.SCRATCH/'overlays';overlay_dir.mkdir(exist_ok=True)
for name,selected in selection.items():
    whole=[]; organs=collections.defaultdict(list); slices=[]; headers=[]; window_sums={w:[] for w in [(-160,240),(-150,250),(-400,400)]}
    clipping=collections.defaultdict(lambda:np.zeros(3,dtype=np.int64)); ctclip=collections.defaultdict(lambda:np.zeros(3,dtype=np.int64)); labels=collections.Counter(); outside_ct=[]
    chosen_overlay=0
    rows+=['### '+name,'Image / mask | Patient / filename-order index | Shape | Stored labels/counts | Header: instance; position; orientation; spacing; slope/intercept','--- | --- | --- | --- | ---']
    for d,m,pid,index in selected:
        ds=pydicom.dcmread(d); raw=ds.pixel_array.astype(np.float32)
        arr=raw*float(getattr(ds,'RescaleSlope',1))+float(getattr(ds,'RescaleIntercept',0)) if name=='CT' else raw
        mask=np.array(Image.open(m).convert('L'))
        if arr.shape!=mask.shape:
            rows.append(f'`{d}` / `{m}` | {pid}/{index} | MISMATCH {arr.shape}/{mask.shape} | UNVERIFIED | skipped');continue
        vals,cnts=np.unique(mask,return_counts=True);labels.update(dict(zip(map(int,vals),map(int,cnts))))
        whole.append(arr.ravel()); foreground=mask>0
        for value in vals:
            if value:organs[int(value)].append(arr[mask==value])
        lo,hi=np.percentile(arr,[1,99])
        for value in vals:
            if value:
                pix=arr[mask==value];clipping[int(value)]+=np.array([(pix<lo).sum(),(pix>hi).sum(),pix.size])
        if name=='CT':
            for low,high in window_sums:
                pix=arr[foreground];ctclip[(low,high)]+=np.array([(pix<low).sum(),(pix>high).sum(),pix.size])
                v=np.clip((arr-low)/(high-low),0,1)
                if foreground.any() and (~foreground).any():window_sums[(low,high)].append((float(v[foreground].mean()),float(v[~foreground].mean())))
        header={k:str(getattr(ds,k,None)) for k in ['InstanceNumber','ImagePositionPatient','ImageOrientationPatient','PixelSpacing','RescaleSlope','RescaleIntercept']}
        rows.append(f'`{d.relative_to(g.ROOT)}` / `{m.relative_to(g.ROOT)}` | {pid}/{index} | {arr.shape} | `{dict(zip(map(int,vals),map(int,cnts)))}` | `{header}`')
        #3 CT overlays and one from each MRI sequence, only nonempty masks.
        limit=3 if name=='CT' else 1
        if foreground.any() and chosen_overlay<limit:
            chosen_overlay+=1
            lower,upper=(-160,240) if name=='CT' else (lo,hi)
            v=np.clip((arr-lower)/max(upper-lower,1e-8),0,1)
            rgb=np.repeat(v[:,:,None],3,axis=2)
            color=rgb.copy(); palette={255:[1,0,0],63:[1,0,0],126:[0,1,0],189:[0,.5,1],252:[1,1,0]}
            for value,col in palette.items():color[mask==value]=.55*color[mask==value]+.45*np.array(col)
            h,w=arr.shape; canvas=Image.new('RGB',(w*2,h+45),'white')
            canvas.paste(Image.fromarray((rgb*255).astype('uint8')),(0,45));canvas.paste(Image.fromarray((color*255).astype('uint8')),(w,45))
            draw=ImageDraw.Draw(canvas);draw.text((5,4),f'{name} patient {pid}; {d.name}; mask {m.name}',fill='black');draw.text((5,22),'Original | overlay: 63/255 red,126 green,189 blue,252 yellow',fill='black')
            outfile=overlay_dir/f'decision_{name}_{chosen_overlay}.png'
            if not outfile.exists():canvas.save(outfile)
            inside=float(arr[foreground].mean());outside=float(arr[~foreground].mean())
            body=(arr>-500) if name=='CT' else (arr>max(0,float(np.percentile(arr,5))))
            tissue={}
            for transform,mm in [('identity',mask),('flip_x',mask[:,::-1]),('flip_y',mask[::-1]),('transpose',mask.T)]:
                if mm.shape==arr.shape:tissue[transform]=float(body[mm>0].mean())
            overlays.append(dict(sequence=name,image=str(d.relative_to(g.ROOT)),mask=str(m.relative_to(g.ROOT)),inside_mean=inside,outside_mean=outside,tissue_overlap=tissue,overlay=str(outfile)))
    result=dict(sequence=name,n=len(whole),whole_percentiles=np.percentile(np.concatenate(whole),Q).round(4).tolist(),mask_counts=dict(labels),organs={str(v):dict(count=sum(x.size for x in arrays),percentiles=np.percentile(np.concatenate(arrays),Q).round(4).tolist(),clip_1_99_counts=clipping[v].tolist(),clip_1_99_fraction=float(sum(clipping[v][:2])/clipping[v][2])) for v,arrays in organs.items()})
    if name=='CT':result['windows']={str(w):dict(clip_counts=ctclip[w].tolist(),clipped_fraction=float(sum(ctclip[w][:2])/max(1,ctclip[w][2])),mean_liver=float(np.mean([v[0] for v in pairs])),mean_nonliver=float(np.mean([v[1] for v in pairs])),mean_contrast=float(np.mean([v[0]-v[1] for v in pairs]))) for w,pairs in window_sums.items()}
    else:
        result['ct_window_clipping']={str(w):{str(v):float(np.mean((np.concatenate(arrays)<w[0])|(np.concatenate(arrays)>w[1]))) for v,arrays in organs.items()} for w in [(-160,240),(-150,250),(-400,400)]}
    summary.append(result); rows+=['','Aggregate result (pooled pixels; contrast averaged across nonempty-mask slices):','```json',json.dumps(result,indent=2),'```']
rows+=['### Six overlay/tissue checks','Numerical tissue overlap cannot prove left/right organ identity. Different flips may all remain inside the body; assess the image overlays and source slice correspondence as well.','```json',json.dumps(overlays,indent=2),'```']
for r in overlays:rows.append('!['+r['sequence']+' overlay]('+r['overlay']+')')
g.append('\n'.join(rows))
for r in summary: print(json.dumps(r))
print('OVERLAYS',json.dumps(overlays))
