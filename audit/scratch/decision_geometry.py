"""Recheck the SAME sampled MRI slices using physical-position matching."""
import decision_guard as g
import re,json,collections
import pydicom,numpy as np
from PIL import Image,ImageDraw
report=g.REPORT.read_text(); evidence=[]; results=[]; saved=False
def records(name):
    section=report.split('### '+name+'\n',1)[1].split('Aggregate result',1)[0]
    return [(g.ROOT/a,g.ROOT/b) for a,b in re.findall(r'^`([^`]+\.dcm)` / `([^`]+\.png)`',section,re.M)]
cache={}
def read_headers(folder):
    if folder not in cache:
        cache[folder]=[(p,pydicom.dcmread(p,stop_before_pixels=True)) for p in sorted(folder.glob('*.dcm'))]
    return cache[folder]
for name in ['CT','T1-in','T1-out','T2']:
    organ_stats=collections.defaultdict(list); clip=collections.defaultdict(lambda:np.zeros(3,dtype=np.int64)); matched=0; unmatched=[]; changes=[]; identity=[]
    for image,oldmask in records(name):
        ds=pydicom.dcmread(image); maskpath=oldmask
        position=np.asarray(ds.ImagePositionPatient,dtype=float)
        orientation=np.asarray(ds.ImageOrientationPatient,dtype=float)
        series=read_headers(image.parent)
        coords=[float(np.dot(np.asarray(h.ImagePositionPatient,float),np.cross(orientation[:3],orientation[3:]))) for _,h in series]
        direction='ascending' if np.all(np.diff(coords)>0) else ('descending' if np.all(np.diff(coords)<0) else 'NONMONOTONIC/duplicate')
        if name=='T1-out':
            candidates=[(p,h) for p,h in read_headers(image.parent.parent/'InPhase') if np.max(np.abs(np.asarray(h.ImagePositionPatient,float)-position))<.01 and np.max(np.abs(np.asarray(h.ImageOrientationPatient,float)-orientation))<1e-5]
            if len(candidates)!=1:
                unmatched.append(str(image.relative_to(g.ROOT)));continue
            ip,ih=candidates[0];maskpath=image.parents[2]/'Ground'/(ip.stem+'.png')
            if not maskpath.exists():unmatched.append(str(image.relative_to(g.ROOT)));continue
            if oldmask!=maskpath:changes.append((str(image.relative_to(g.ROOT)),oldmask.name,maskpath.name))
        raw=ds.pixel_array.astype(np.float32);mask=np.array(Image.open(maskpath).convert('L'))
        if name=='CT':raw=raw*float(getattr(ds,'RescaleSlope',1))+float(getattr(ds,'RescaleIntercept',0))
        matched+=1
        if name in ['T1-in','T2']:identity.append(image.stem==maskpath.stem)
        for label in np.unique(mask):
            if not label:continue
            pix=raw[mask==label];organ_stats[int(label)].append(pix)
            for upper in [99,99.5,99.9]:
                lo,hi=np.percentile(raw,[1,upper]);clip[(int(label),upper)]+=np.array([(pix<lo).sum(),(pix>hi).sum(),pix.size])
        evidence.append(dict(sequence=name,image=str(image.relative_to(g.ROOT)),mask=str(maskpath.relative_to(g.ROOT)),filename_order_physical_direction=direction,physical_step_range=[float(min(np.abs(np.diff(coords)))),float(max(np.abs(np.diff(coords))))] if len(coords)>1 else [],matrix=list(raw.shape),iop=orientation.tolist(),position=position.tolist()))
        if name=='T1-out' and not saved:
            v=np.clip(raw/max(float(np.percentile(raw,99.9)),1),0,1);rgb=np.repeat(v[:,:,None],3,axis=2);rgb[mask>0]=rgb[mask>0]*.5+np.array([.5,0,0])
            canvas=Image.new('RGB',(raw.shape[1],raw.shape[0]+35),'white');canvas.paste(Image.fromarray((rgb*255).astype('uint8')),(0,35));ImageDraw.Draw(canvas).text((2,2),image.name+' / '+maskpath.name+'\nPhysical-position matched overlay',fill='black')
            out=g.SCRATCH/'overlays/decision_T1-out_physical.png'
            if not out.exists():canvas.save(out)
            saved=True
    r=dict(sequence=name,matched_samples=matched,unmatched_samples=unmatched,changed_pairings=changes,all_sampled_image_mask_stems_equal=all(identity) if identity else None,organ_percentiles={str(k):np.percentile(np.concatenate(v),[.5,1,5,50,95,99,99.5]).round(4).tolist() for k,v in organ_stats.items()},clipping_fraction={str((k,u)):float(sum(v[:2])/v[2]) for (k,u),v in clip.items()})
    results.append(r);print(json.dumps(r))
g.append('## M5 — Physical-position pairing correction and MRI percentile alternatives\n\nCommand: `python3 -B audit/scratch/decision_geometry.py`; reuses the same selected image paths. Earlier sorted-order T1-out organ statistics/overlay are a diagnostic of wrong pairing, **not valid organ intensity estimates** where this check changes/rejects the pair. No source files or masks were corrected. T1-out masks are associated with the in-phase image of the same position/orientation (tolerance.01mm/1e-5), requiring exactly one match.\n\n```json\n'+json.dumps(results,indent=2)+'\n```\n\n### Sample physical geometry evidence\n\n```json\n'+json.dumps(evidence,indent=2)+'\n```\n\n![Out-phase physical match]('+str(g.SCRATCH/'overlays/decision_T1-out_physical.png')+')')
