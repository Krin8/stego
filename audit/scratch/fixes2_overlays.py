"""Bounded overlays from the v2 dry-run pairing; never generates derivatives."""
import fixes2_guard as g
import sys,json,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(g.ROOT/'src'))
from chaos_preprocessing import plan_dataset,read_image
raw=g.ROOT/'src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets'
out=g.SCRATCH/'overlays'
if '--verify-only' in sys.argv:
    assert (out/'numbers.json').is_file(),'BEFORE: no overlays/numbers for approved v2 pairs'
    rows=json.loads((out/'numbers.json').read_text())
    assert len(rows)==6 and all((out/r['overlay']).is_file() for r in rows)
    assert all(r['foreground_pixels']>0 and np.isfinite(r['inside_mean']) and np.isfinite(r['outside_mean']) for r in rows)
    print('PASS six v2 overlays and numerical records');sys.exit(0)
records=plan_dataset(raw);rows=[]
out.mkdir(parents=True,exist_ok=True)
for mod,pid in [('CT','1'),('CT','22'),('CT','30'),('T1DUAL','8'),('T1DUAL','37'),('T2SPIR','37')]:
    candidates=[r for r in records if r['modality']==mod and r['patient_id']==pid]
    # Choose centrally located labeled planes for a readable overlay; this is diagnostic only.
    selected=None
    for r in sorted(candidates,key=lambda r:abs(candidates.index(r)-len(candidates)//2))[:10]:
        mask=np.array(Image.open(raw/r['mask']).convert('L'))
        if (mask>0).any():selected=r;break
    assert selected is not None
    values=read_image(raw/r['dicom'],mod).astype(np.float32)/65535
    fg=mask>0;rgb=np.repeat(values[:,:,None],3,axis=2);color=rgb.copy()
    for k,palette in {255:[1,0,0],63:[1,0,0],126:[0,1,0],189:[0,.5,1],252:[1,1,0]}.items():
        color[mask==k]=.55*color[mask==k]+.45*np.array(palette)
    h,w=values.shape;image=Image.new('RGB',(2*w,h+50),'white')
    image.paste(Image.fromarray((rgb*255).round().astype('uint8')),(0,50))
    image.paste(Image.fromarray((color*255).round().astype('uint8')),(w,50))
    ImageDraw.Draw(image).text((5,5),f"{mod}/{pid}: {r['dicom'].split('/')[-1]}\nNative image | paired mask; no flip/transpose",fill='black')
    name=f'{mod}_{pid}.png';image.save(out/name)
    rows.append(dict(modality=mod,patient=pid,sample_id=r['sample_id'],image=r['dicom'],mask=r['mask'],overlay=name,inside_mean=float(values[fg].mean()),outside_mean=float(values[~fg].mean()),foreground_pixels=int(fg.sum()),foreground_nonzero_fraction=float((values[fg]>0).mean()),labels=np.unique(mask).tolist()))
(out/'numbers.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
