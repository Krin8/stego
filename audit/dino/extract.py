"""D1 only: strict local weight load, real project DINO implementation, no annotations."""
from common import *
os.environ['LABEL_FREE']='1'
from dino.vision_transformer import vit_base
inv=json.loads((OUT/'inventory.json').read_text());entry=inv['DINO_v1_ViT_B8']['files'][0]
assert (OUT/'PROTOCOL.json').is_file();assert not (OUT/'D1_extract.json').exists()
model=vit_base(patch_size=8).cpu();sd=torch.load(entry['path'],map_location='cpu',weights_only=True,mmap=True)
loaded=model.load_state_dict(sd,strict=True);del sd;model.eval().requires_grad_(False)
rr=records();fixed=json.loads((OUT/'PROTOCOL.json').read_text())['figures']['indices']
def forward(x):
 with torch.inference_mode():
  tokens=model.get_intermediate_layers(x,n=1)[0]
  assert tokens.shape[1:]==(785,768)
  return F.normalize(tokens[:,1:],dim=-1)
x=image_tensor(crop_image(raw_image(rr[fixed[0]]),rr[fixed[0]]['box']),224,MEAN,STD)[None]
f1=forward(x);f2=forward(x);assert torch.equal(f1,f2)
assert not model.training and not any(p.requires_grad for p in model.parameters())
out=np.lib.format.open_memmap(OUT/'D1_features.npy',mode='w+',dtype=np.float32,shape=(2341,784,768))
with torch.inference_mode():
 for start in range(0,len(rr),4):
  batch=rr[start:start+4]
  x=torch.stack([image_tensor(crop_image(raw_image(r),r['box']),224,MEAN,STD) for r in batch])
  f=forward(x);assert torch.isfinite(f).all();out[start:start+len(batch)]=f.numpy()
  if start%128==0:print('D1',start,'seconds',elapsed(),flush=True)
out.flush()
dump('D1_extract.json',dict(producer='extract.py',weight=entry,missing_keys=list(loaded.missing_keys),unexpected_keys=list(loaded.unexpected_keys),model_eval=not model.training,all_requires_grad_false=not any(p.requires_grad for p in model.parameters()),deterministic_exact=torch.equal(f1,f2),deterministic_max_abs_difference=float((f1-f2).abs().max()),cls_excluded=True,feature_shape=list(out.shape),labels_used=False,seconds=elapsed()))
print('DONE D1',elapsed(),flush=True)
