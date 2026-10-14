from common import *
from collections import Counter
inv=json.loads((FORE/'inventory.json').read_text()); model,ck=load(inv['seed0']['path'])
train=dataset(model.cfg,'train');val=dataset(model.cfg,'val')
assert len(train)==1869 and len(val)==472
assert [r['sample_id'] for r in val.dataset.records]==json.loads((FORE/'val_order.json').read_text())
print('record',train.dataset.records[0],flush=True)
def patient(r):return str(r['patient_id'])
meta=dict(script=__file__,train_ids=[r['sample_id'] for r in train.dataset.records],val_ids=[r['sample_id'] for r in val.dataset.records],train_patients=[patient(r) for r in train.dataset.records],val_patients=[patient(r) for r in val.dataset.records])
assert not set(meta['train_patients'])&set(meta['val_patients'])
meta['train_counts']=dict(Counter(meta['train_patients']));meta['val_counts']=dict(Counter(meta['val_patients']))
dump('samples.json',meta)
train.dataset=ImagesOnlyDataset(train.dataset)
dest=np.lib.format.open_memmap(OUT/'train_features.npy',mode='w+',dtype='float32',shape=(len(train),196,768))
with torch.inference_mode():
 for batch in DataLoader(train,batch_size=16,shuffle=False,num_workers=0):
  f,c=model.net(batch['img']);dest[batch['ind'].numpy()]=f.permute(0,2,3,1).reshape(-1,196,768).numpy()
  if int(batch['ind'][-1])%256==255:print('train',int(batch['ind'][-1])+1,'seconds',elapsed(),flush=True)
 dest.flush();frozen=np.load(FORE/'val_features.npy',mmap_mode='r');labels=np.load(FORE/'val_labels.npy',mmap_mode='r');error=0.
 for batch in DataLoader(val,batch_size=16,shuffle=False,num_workers=0):
  f,c=model.net(batch['img']);idx=batch['ind'].numpy();error=max(error,float(np.max(np.abs(f.numpy()-frozen[idx]))));assert np.array_equal(batch['label'].numpy(),labels[idx])
assert error<1e-5
mean=dest.reshape(-1,768).mean(axis=0,dtype=np.float64).astype(np.float32);np.save(OUT/'train_mean.npy',mean)
result=dict(script=__file__,train_slices=len(train),train_patches=len(train)*196,val_slices=len(val),train_patients=meta['train_counts'],val_patients=meta['val_counts'],val_feature_max_abs_difference=error,all_val_labels_match_forensics=True,train_labels_opened=False,source_checkpoint=inv['seed0']['path'],feature_shape=list(dest.shape),elapsed_seconds=elapsed())
dump('extract.json',result);note('Extraction: '+json.dumps(result));print(json.dumps(result),flush=True)
