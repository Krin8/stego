"""Evaluate a bounded chunk of actual saved heads/probes on every validation pixel."""
from common import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--start',type=int,required=True);p.add_argument('--stop',type=int,required=True);args=p.parse_args()
ev=json.loads((ROOT/'audit/run_records/fixes3_full_C/postrun_evidence.json').read_text());candidates=ev['checkpoints'][args.start:args.stop];ref=torch.load(inventory()['seed0']['path'],map_location='cpu',mmap=True,weights_only=False)
features=np.load(OUT/'val_features.npy',mmap_mode='r');labels=np.load(OUT/'val_labels.npy',mmap_mode='r');results=[];model=None
for candidate in candidates:
 path=candidate['path']
 if not Path(path).is_file():results.append(dict(path=path,missing=True));continue
 model,ck=load(path,model)
 # Exact equality of every saved visual-trunk parameter/buffer establishes eligibility
 # for reuse of the read-only diagnostic activations from the real frozen extractor.
 keys=[k for k in ck['state_dict'] if k.startswith('net.model.')]
 assert keys and all(torch.equal(ck['state_dict'][k],ref['state_dict'][k]) for k in keys)
 ds=dataset(model.cfg);assert [r['sample_id'] for r in ds.dataset.records]==json.loads((OUT/'val_order.json').read_text())
 original=model.net.model.get_intermediate_layers
 x=torch.stack([ds[i]['img'] for i in range(16)])
 with torch.inference_mode():
  f_real,c_real=model.net(x)
  feature_error=float((f_real-torch.from_numpy(np.array(features[:16]))).abs().max())
  assert feature_error<1e-5
  provider={}
  def from_diagnostic_features(img,*a,**kw):
   assert kw==dict(n=1,reshape=True,norm=True)
   assert provider['features'].shape[0]==img.shape[0]
   return (provider['features'],)
  model.net.model.get_intermediate_layers=from_diagnostic_features
  provider['features']=torch.from_numpy(np.array(features[:16]));_,c_reuse=model.net(x)
  code_error=float((c_real-c_reuse).abs().max());assert code_error<1e-5
  conf=np.zeros((8,2),dtype=np.int64)
  for start in range(0,472,16):
   provider['features']=torch.from_numpy(np.array(features[start:start+16]));n=len(provider['features'])
   # Actual extractor forward, head modules and probe run unchanged. Only the
   # already-verified frozen trunk output is supplied from diagnostic scratch.
   _,code=model.net(torch.zeros((n,3,224,224)))
   up=F.interpolate(code,(224,224),mode='bilinear',align_corners=False);_,probs=model.cluster_probe(up,None)
   pred=probs.argmax(1).numpy();y=np.asarray(labels[start:start+n])
   conf+=np.bincount((pred*2+y).ravel(),minlength=16).reshape(8,2)
  model.net.model.get_intermediate_layers=original
 r=mapping_report(conf);r.update(path=path,batch=ck['global_step']//3,optimizer_step=ck['global_step'],stored_miou=candidate['filename_score'],stored_score_validation_batch=candidate['preceding_validation'],backbone_equal=True,feature_reuse_error=feature_error,code_reuse_error=code_error,script=__file__,elapsed_seconds=elapsed());results.append(r)
 dump(f'task5_{args.start}_{args.stop}.json',results);print(r['batch'],r['majority']['iou'],r['optimal']['iou'],'seconds',elapsed(),flush=True);del ck
note(f'Task5 chunk{args.start}:{args.stop} completed; actual heads/probes evaluated all472 slices per checkpoint, frozen-trunk exact equality and live-forward equality verified. Producer`task5.py`; detail`task5_{args.start}_{args.stop}.json`; runtime{elapsed():.2f}s. Reused scratch diagnostic activations only; no pipeline cache generated/modified.')
