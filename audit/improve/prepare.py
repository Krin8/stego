import os
os.environ['IMPROVE_LABEL_FREE']='1'
from pipeline import *
import inspect,importlib.metadata
from validation_visuals import fixed_validation_indices
inv=json.loads((FORE/'inventory.json').read_text());model,ck=load(inv['seed0']['path'])
verification=json.loads((FINAL/'verification.json').read_text())
records=[];fallbacks=0
for split in ['train','val']:
 ds=dataset(model.cfg,split)
 for r,sample in zip(ds.dataset.records,ds.dataset.samples):
  image,mask,mod,p=sample;assert mod=='CT'
  with Image.open(image) as im:a=np.array(im,dtype=np.float32)/65535.
  box,fallback=body_box(a);fallbacks+=fallback
  records.append(dict(sample_id=r['sample_id'],patient=str(p),split=split,image=image,mask=mask,
    shape=list(a.shape),box=box,physical_position=r['physical_position'],stem=r['stem']))
assert [r['sample_id'] for r in records]==verification['ordered_sample_ids']
assert {tuple(r['shape']) for r in records}=={(512,512)}
ds=dataset(model.cfg,'val');fixed=[1869+i for i in fixed_validation_indices(ds.dataset.records)]
dump('metadata.json',dict(script=__file__,labels_used=False,records=records,fixed=fixed,
 patient_counts=verification['patient_counts'],ordered_ids_sha256=verification['ordered_ids_sha256'],
 body_fallbacks=fallbacks,versions={m:importlib.metadata.version(m) for m in ['torch','timm','scikit-learn','scipy','opencv-python']}))
tests={};img=raw_image(records[0]);crop=crop_image(img,records[0]['box'])
with torch.inference_mode():
 for res in [224,336,448]:
  try:
   x=image_tensor(crop,res)[None];f,_=model.net(x)
   assert tuple(f.shape)==(1,768,res//16,res//16)
   tests[str(res)]=dict(supported=True,shape=list(f.shape))
  except Exception as e:tests[str(res)]=dict(supported=False,error=repr(e))
poscode=inspect.getsource(type(model.net.model)._pos_embed)
(OUT/'position_embedding_source.txt').write_text(poscode)
tests['position_interpolation_code']=poscode
tests['script']=__file__;tests['elapsed_seconds']=elapsed();dump('support.json',tests)
print(json.dumps({k:v for k,v in tests.items() if k!='position_interpolation_code'}),flush=True)
