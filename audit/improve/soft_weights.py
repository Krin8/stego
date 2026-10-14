import os
os.environ['IMPROVE_LABEL_FREE']='1'
from pipeline import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
variant=json.loads((OUT/'parent_selection.json').read_text())['parent'];tag=f'{variant}_s{a.seed}'
fit=json.loads((OUT/f'{tag}_fit.json').read_text());g=fit['grid']
centers=np.load(OUT/f'{tag}_centers.npy');temperature=fit['temperature']
dest=np.lib.format.open_memmap(OUT/f'B4_s{a.seed}_weights.npy',mode='w+',dtype='float32',shape=(2341,g,g,8))
idx=0
for part in features(variant):
 for start in range(0,len(part),16):
  f=np.array(part[start:start+16]);n=len(f)
  if variant=='B0':f/=np.maximum(np.linalg.norm(f,axis=-1,keepdims=True),1e-12)
  x=f.reshape(-1,centers.shape[1]);d=np.maximum(0,np.square(x).sum(1)[:,None]+np.square(centers).sum(1)[None,:]-2*x@centers.T)
  z=-(d-d.min(axis=1,keepdims=True))/temperature;w=np.exp(z);w/=w.sum(1,keepdims=True)
  dest[idx:idx+n]=w.reshape(n,g,g,8);idx+=n
assert idx==2341;dest.flush()
dump(f'B4_s{a.seed}_weights.json',dict(script=__file__,seed=a.seed,parent=variant,temperature=temperature,
 labels_used='None for distances, temperature or weights; parent selection itself used LOPO labels',elapsed_seconds=elapsed()))
print('DONE',a.seed,temperature,elapsed(),flush=True)
