"""Actual extractor/preprocessing, all val blocks; no annotations opened."""
from common import *
from numpy.lib.format import open_memmap
from collections import Counter
def no_labels(event,args):
 if event=='open' and not isinstance(args[0],int) and ('/labels/' in str(args[0]) or str(args[0]).endswith('labels.npy')):raise PermissionError('No annotations during extraction')
sys.addaudithook(no_labels)
inv=json.loads((FORE/'inventory.json').read_text());model,ck=load(inv['seed0']['path'])
ds=dataset(model.cfg,'val',True);images=ImagesOnlyDataset(ds.dataset)
meta=json.loads((BASE/'samples.json').read_text());ids=[r['sample_id'] for r in ds.dataset.records]
assert ids==meta['val_ids'] and len(ids)==472
nns=np.asarray(ds.nns);old=np.load(PAIR/'val_image_neighbors.npy');assert np.array_equal(nns,old)
np.save(OUT/'neighbors.npy',nns[:,1:8]);dump('records.json',ds.dataset.records)
assert model.cfg.num_neighbors==7 and np.all(nns[:,0]==np.arange(472))
net=model.net;original=net.model.get_intermediate_layers;captured={}
def capture(x,*args,**kwargs):
 assert kwargs==dict(n=1,reshape=True,norm=True)
 layers=original(x,n=[8,10,11],reshape=True,norm=True)
 captured.update(dict(zip([9,11,12],layers)))
 return (layers[-1],)
net.model.get_intermediate_layers=capture
arrays={b:open_memmap(OUT/f'block{b}.npy',mode='w+',dtype='float32',shape=(472,768,14,14)) for b in [9,11]}
final=np.load(FORE/'val_features.npy',mmap_mode='r');maxdiff=0.0
with torch.inference_mode():
 for start in range(0,472,16):
  x=torch.stack([images[i][0] for i in range(start,min(start+16,472))]);f,c=net(x)
  assert torch.equal(f,captured[12]) and f.shape[1:]==(768,14,14)
  maxdiff=max(maxdiff,float(np.max(np.abs(f.numpy()-final[start:start+len(x)]))))
  for b in arrays:arrays[b][start:start+len(x)]=captured[b].numpy()
  if start%128==0:print('extract',start,elapsed(),flush=True)
for x in arrays.values():x.flush()
assert maxdiff==0.0,maxdiff
mean=np.load(BASE/'train_mean.npy');train=np.load(BASE/'train_features.npy',mmap_mode='r')
recomputed=np.zeros(768,dtype=np.float64)
for start in range(0,len(train),32):recomputed+=np.asarray(train[start:start+32],dtype=np.float64).sum((0,1))
recomputed=(recomputed/(len(train)*196)).astype(np.float32);assert np.array_equal(mean,recomputed)
assignment=np.load(BASE/'k8_c0_s0_assignments.npy',mmap_mode='r');tokens=np.array(assignment[:,::16,::16])
assert np.array_equal(np.repeat(np.repeat(tokens,16,1),16,2),assignment)
np.save(OUT/'kmeans_tokens.npy',tokens)
result=dict(script=__file__,labels_used=False,val_slices=len(ids),patients=dict(Counter(meta['val_patients'])),source_checkpoint=inv['seed0']['path'],blocks=[9,11,12],norm=True,cls_excluded=True,features_shape=[472,768,14,14],final_max_abs_difference=maxdiff,train_mean_exactly_verified=True,train_mean_images=len(train),backbone_frozen=all(not p.requires_grad for p in net.full_model.parameters()),backbone_eval=not net.full_model.training,cache_verified_by_actual_loader=True,neighbors_used=7,kmeans_reference='baseline/k8_c0_s0_assignments.npy: seed0 K8 no mean subtraction',kmeans_nearest_upsampling_exactly_verified=True,elapsed_seconds=elapsed())
dump('extract.json',result);note('Extraction completed: `audit/scratch/selectivity/extract.py` → `extract.json`; actual full-cohort final features match FORENSICS exactly, train mean and K8 token recovery verified without reading labels.');print(json.dumps(result,indent=2))
