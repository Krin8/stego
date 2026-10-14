"""Read-only feature and geometry checks, no labels."""
import os
os.environ['IMPROVE_LABEL_FREE']='1'
from pipeline import *
meta=metadata();inv=json.loads((FORE/'inventory.json').read_text())
model,ck=load(inv['seed0']['path']);differences={}
for split in ['train','val']:
 ds=dataset(model.cfg,split);ds.dataset=ImagesOnlyDataset(ds.dataset)
 batch=next(iter(DataLoader(ds,batch_size=16,shuffle=False,num_workers=0)))
 with torch.inference_mode():f,_=model.net(batch['img'])
 if split=='train':actual=f.permute(0,2,3,1).reshape(16,196,768).numpy();old=np.load(ROOT/'audit/scratch/baseline/train_features.npy',mmap_mode='r')[:16]
 else:actual=f.numpy();old=np.load(FORE/'val_features.npy',mmap_mode='r')[:16]
 differences[split]=float(np.abs(actual-old).max());assert differences[split]<1e-5
# Synthetic native square crop must invert exactly when no resizing occurs.
for box in [[1,2,3],[-1,-2,8],[0,0,5]]:
 r=dict(box=box,shape=[5,5]);im=np.arange(25,dtype=np.float32).reshape(5,5)
 cropped=crop_image(im,box);back=inverse_grid(cropped,r,outside=-1)
 y,x,s=box;ya,xa,yb,xb=max(0,y),max(0,x),min(5,y+s),min(5,x+s)
 assert np.array_equal(back[ya:yb,xa:xb],im[ya:yb,xa:xb])
# Missing-crop liver pixels remain false negatives in the original scoring function.
conf=np.zeros((9,2),dtype=np.int64);conf[0]=[80,0];conf[1]=[0,10];conf[8]=[0,10]
test=mapped_report(conf);assert test['optimal']['clusters']==[1] and test['optimal']['iou']==.5
dump('checks.json',dict(script=__file__,labels_used=False,live_feature_max_abs_differences=differences,
 synthetic_crop_roundtrip=True,outside_crop_false_negatives_preserved=True,
 elapsed_seconds=elapsed()))
print('PASS',differences,flush=True)
