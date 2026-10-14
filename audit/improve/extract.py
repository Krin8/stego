"""Frozen extraction for spatial variants, without annotation reads."""
import os
os.environ['IMPROVE_LABEL_FREE']='1'
from pipeline import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['B1','B2_336','B2_448','B3','B6'],required=True);a=p.parse_args()
variant=a.variant;records=metadata()['records']
assert not (OUT/f'{variant}_extract.json').exists(),'Completed extraction already exists'
if variant=='B6':
 hub=Path.home()/'.cache/torch/hub/facebookresearch_dinov2_main'
 weights=Path.home()/'.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth'
 assert hub.is_dir() and weights.is_file()
 model=torch.hub.load(str(hub),'dinov2_vitl14',source='local',pretrained=False)
 sd=torch.load(weights,map_location='cpu',weights_only=True,mmap=True)
 loaded=model.load_state_dict(sd,strict=True);del sd
 weight_info=dict(path=str(weights),bytes=weights.stat().st_size,strict_load=str(loaded),hub=str(hub),architecture='DINOv2 ViT-L/14')
 model.cpu().eval().requires_grad_(False);res=224;grid=16;dim=1024
 def extract(x):return model.forward_features(x)['x_norm_patchtokens'].transpose(1,2).reshape(-1,1024,16,16)
 mean,std=(.485,.456,.406),(.229,.224,.225)
else:
 inv=json.loads((FORE/'inventory.json').read_text());model,ck=load(inv['seed0']['path']);del ck
 weight_info=dict(path=inv['seed0']['path'],strict_load=True,architecture='BiomedCLIP ViT-B/16',source='Frozen backbone only; head output unused')
 res={'B1':224,'B2_336':336,'B2_448':448,'B3':448}[variant];grid=res//16;dim=768
 def extract(x):return model.net(x)[0]
 mean,std=MEAN,STD
if variant.startswith('B2'):assert json.loads((OUT/'support.json').read_text())[str(res)]['supported']
dest=np.lib.format.open_memmap(OUT/f'{variant}_features.npy',mode='w+',dtype='float32',shape=(2341,grid*grid,dim))
batch_size=8 if res==448 or variant=='B6' else 16
with torch.inference_mode():
 for start in range(0,len(records),batch_size):
  batch=records[start:start+batch_size]
  x=torch.stack([image_tensor(crop_image(raw_image(r),r['box']),res,mean,std) for r in batch])
  if variant=='B3':
   f=torch.zeros((len(batch),768,28,28));counts=torch.zeros((1,1,28,28))
   for y in [0,112,224]:
    for z in [0,112,224]:
     tile=extract(x[:,:,y:y+224,z:z+224]);f[:,:,y//16:y//16+14,z//16:z//16+14]+=tile
     counts[:,:,y//16:y//16+14,z//16:z//16+14]+=1
   f/=counts
  else:f=extract(x)
  f=F.normalize(f,dim=1);assert torch.isfinite(f).all()
  dest[start:start+len(batch)]=f.permute(0,2,3,1).reshape(len(batch),grid*grid,dim).numpy()
  if start%128==0:print(variant,start,elapsed(),flush=True)
dest.flush()
dump(f'{variant}_extract.json',dict(script=__file__,variant=variant,labels_used=False,
 input_resolution=res,grid=grid,feature_shape=list(dest.shape),mean=list(mean),std=list(std),
 weights=weight_info,
 frozen=True,cpu=True,eval=True,patch_norm='final block LayerNorm then L2; tile features averaged before L2',
 elapsed_seconds=elapsed()))
print('DONE',variant,elapsed(),flush=True)
