"""Bounded checkpoint chunk: actual objective/heads/probe, no annotation reads."""
from common import *
from sklearn.metrics import mutual_info_score
import argparse
p=argparse.ArgumentParser();p.add_argument('--start',type=int,required=True);p.add_argument('--stop',type=int,required=True);args=p.parse_args()
node=next(n for n in ast.parse((FORE/'task2.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='agreement')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(FORE/'task2.py'),'exec'),globals())
def no_annotations(event,args):
 if event=='open' and not isinstance(args[0],int) and ('/labels/' in str(args[0]) or str(args[0]).endswith('val_labels.npy')):raise PermissionError('No labels in proxy computation')
sys.addaudithook(no_annotations)
ev=json.loads((ROOT/'audit/run_records/fixes3_full_C/postrun_evidence.json').read_text());candidates=ev['checkpoints'][args.start:args.stop]
inv=json.loads((FORE/'inventory.json').read_text());ref=torch.load(inv['seed0']['path'],map_location='cpu',mmap=True,weights_only=False)
val=np.load(FORE/'val_features.npy',mmap_mode='r');train=np.load(OUT/'train_features.npy',mmap_mode='r');base=np.load(OUT/'k8_c0_s0_assignments.npy',mmap_mode='r')
fixed=json.loads((FORE/'task4.json').read_text())['results']['best600'];pairs=np.asarray(fixed['anchor_positive_indices']);assert pairs.shape==(16,16,2)
results=[];model=None
for item in candidates:
 model,ck=load(item['path'],model);cfg=model.cfg
 keys=[k for k in ck['state_dict'] if k.startswith('net.model.')]
 assert keys and all(torch.equal(ck['state_dict'][k],ref['state_dict'][k]) for k in keys)
 ds=dataset(cfg,'train');assert [r['sample_id'] for r in ds.dataset.records]==json.loads((OUT/'samples.json').read_text())['train_ids'];ds.dataset=ImagesOnlyDataset(ds.dataset)
 original=model.net.model.get_intermediate_layers;provider={}
 def from_diagnostic_features(img,*a,**kw):
  assert kw==dict(n=1,reshape=True,norm=True) and len(img)==len(provider['features'])
  return (provider['features'],)
 def forward_features(f):
  provider['features']=f
  return model.net(torch.zeros((len(f),3,224,224)))
 with torch.inference_mode():
  x=torch.stack([ds[int(i)]['img'] for i in pairs[0,:,0]]);f_real,c_real=model.net(x)
  saved=torch.from_numpy(np.array(train[pairs[0,:,0]])).reshape(16,14,14,768).permute(0,3,1,2).contiguous()
  feature_error=float((f_real-saved).abs().max());assert feature_error<1e-5
  model.net.model.get_intermediate_layers=from_diagnostic_features
  _,c_reuse=forward_features(saved);code_error=float((c_real-c_reuse).abs().max());assert code_error<1e-5
  terms_rows=[]
  for j,indices in enumerate(pairs):
   f=torch.from_numpy(np.array(train[indices[:,0]])).reshape(16,14,14,768).permute(0,3,1,2).contiguous()
   fp=torch.from_numpy(np.array(train[indices[:,1]])).reshape(16,14,14,768).permute(0,3,1,2).contiguous()
   _,c=forward_features(f);_,cp=forward_features(fp);torch.manual_seed(SEED+j)
   terms=model.contrastive_corr_loss_fn(f,fp,None,None,c,cp,f,c)
   intra=float(terms[0]);inter=float(terms[2]);neg=float(terms[4].mean())
   weighted=cfg.correspondence_weight*(cfg.pos_intra_weight*intra+cfg.pos_inter_weight*inter+cfg.neg_inter_weight*neg)
   terms_rows.append(dict(pos_intra=intra,pos_inter=inter,neg_inter=neg,weighted_correspondence=weighted))
  monitor=CollapseMonitor(8);cm=np.zeros((8,8),dtype=np.int64)
  for start in range(0,len(val),16):
   _,code=forward_features(torch.from_numpy(np.array(val[start:start+16])))
   up=F.interpolate(code,(224,224),mode='bilinear',align_corners=False);_,probs=model.cluster_probe(up,None);ids=probs.argmax(1)
   monitor.update(code,ids);pr=ids.numpy();b=np.asarray(base[start:start+len(pr)],dtype=np.int64)
   cm+=np.bincount((pr*8+b).ravel(),minlength=64).reshape(8,8)
  model.net.model.get_intermediate_layers=original
 r=dict(path=item['path'],batch=ck['global_step']//3,backbone_equal=True,feature_reuse_error=feature_error,code_reuse_error=code_error,monitors=monitor.compute(),losses={k:float(np.mean([v[k] for v in terms_rows])) for k in terms_rows[0]},loss_batches=terms_rows,agreement=agreement(cm),labels='None: objective uses fixed16 training image batches; monitors and ARI use all472 validation images/assignments without labels.',elapsed_seconds=elapsed())
 for tag in ['best600','final']:
  if item['path']==inv[tag]['path']:
   old=json.loads((FORE/'task4.json').read_text())['results'][tag]['summary'];r['forensics_loss_max_abs_difference']=max(abs(r['losses'][k]-old[k]) for k in old);assert r['forensics_loss_max_abs_difference']<1e-6
 results.append(r);dump(f'proxies_{args.start}_{args.stop}.json',dict(script=__file__,reference_kmeans='k8_c0_s0; predeclared before scores; fit all train images',results=results,elapsed_seconds=elapsed()))
 print(r['batch'],r['losses']['weighted_correspondence'],r['agreement']['ARI'],elapsed(),flush=True);del ck
note(f'Label-free proxy chunk {args.start}:{args.stop}: `proxies_{args.start}_{args.stop}.json`, producer `proxies.py`, runtime {elapsed():.2f}s. No annotation files opened.')
