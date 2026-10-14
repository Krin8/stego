from common import *
from knn_cache import expected_identity,cache_path
from collections import Counter
inv=json.loads((FORE/'inventory.json').read_text());model,ck=load(inv['seed0']['path'])
result=dict(script=__file__,checkpoints=inv,source_sha256={name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest() for name in ['modules.py','data.py','chaos_preprocessing.py','train_segmentation.py']})
try:
 ds=dataset(model.cfg,'val',True);records=ds.dataset.records;assert len(ds)==472
 assert [r['sample_id'] for r in records]==json.loads((FORE/'val_order.json').read_text())
 identity=expected_identity(ds.dataset,model.cfg);path=cache_path(model.cfg.pytorch_data_dir,identity)
 result.update(validation_cache=str(path),cache_shape=list(ds.nns.shape),num_neighbors=int(model.cfg.num_neighbors),patient_counts=dict(Counter(r['patient_id'] for r in records)),identity_validated_by_project=True)
 dump('records.json',records);np.save(OUT/'val_image_neighbors.npy',ds.nns.numpy())
except Exception as e:
 result['validation_cache_error']=repr(e);dump('setup.json',result);raise
features=np.load(FORE/'val_features.npy',mmap_mode='r');labels=np.load(FORE/'val_labels.npy',mmap_mode='r');max_error=0
with torch.inference_mode():
 for start in range(0,472,16):
  pack=[ds.dataset[i] for i in range(start,min(472,start+16))];x=torch.stack([p[0] for p in pack]);y=torch.stack([p[1] for p in pack]);f,c=model.net(x)
  max_error=max(max_error,float((f-torch.from_numpy(np.array(features[start:start+len(pack)]))).abs().max()));assert np.array_equal(y.numpy(),labels[start:start+len(pack)])
assert max_error==0
result.update(all_val_labels_match=True,max_feature_error=max_error,shifts={k:float(getattr(model.cfg,k)) for k in ['pos_intra_shift','pos_inter_shift','neg_inter_shift']},elapsed_seconds=elapsed())
dump('setup.json',result);note('Setup: '+json.dumps(result));print(json.dumps(result,indent=2))
