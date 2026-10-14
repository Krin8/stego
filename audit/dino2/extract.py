"""Local transformers final-normalized patch features; no labels or network."""
from common import *
import argparse,inspect,importlib.metadata
from safetensors import safe_open
from transformers import ViTModel,Dinov2Model
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['D2','D3'],required=True);a=p.parse_args();arm=a.arm
os.environ['LABEL_FREE']='1';verify=json.loads((OUT/'weight_verification.json').read_text());assert verify['passed']
protocol=json.loads((OUT/'PROTOCOL.json').read_text());spec=protocol['arms'][arm];folder=OUT/spec['folder'];v=verify['arms'][arm]
assert sha(folder/'model.safetensors')==v['expected_sha256'],'STOP: weight digest changed'
pre=json.loads((folder/'preprocessor_config.json').read_text());mean=pre['image_mean'];std=pre['image_std'];assert mean==list(MEAN) and std==list(STD)
kwargs=dict(local_files_only=True,trust_remote_code=False,use_safetensors=True,output_loading_info=True)
if arm=='D2':kwargs['add_pooling_layer']=False
try:
 cls=ViTModel if arm=='D2' else Dinov2Model
 model,info=cls.from_pretrained(str(folder),**kwargs)
 bad={k:info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'] if info.get(k)}
 if bad:raise ValueError('Nonempty load diagnostics: '+str(bad))
except Exception as e:
 dump(f'{arm}_FAILED.json',dict(stage='local load',error=repr(e),loading_info=locals().get('info'),continue_other_arm=True));print('ARM STOPPED',arm,repr(e),flush=True);raise SystemExit(0)
model=model.cpu().float().eval().requires_grad_(False)
with safe_open(str(folder/'model.safetensors'),framework='pt',device='cpu') as f:tensors=len(f.keys())
# Capture exactly the model's final layer norm and verify returned last_hidden_state.
captured={}
hook=model.layernorm.register_forward_hook(lambda m,inp,out:captured.update(final=out))
def forward(x):
 with torch.inference_mode():
  outputs=model(pixel_values=x,**({'interpolate_pos_encoding':True} if arm=='D2' else {}))
  t=outputs.last_hidden_state
  assert t.shape==(len(x),785,768),str(t.shape)
  assert torch.equal(t,captured['final']),'last_hidden_state differs from final LayerNorm'
  return F.normalize(t[:,1:],dim=-1)
rr=records();fixed=protocol['figures']['indices'];res=spec['input']
def images(batch):return torch.stack([image_tensor(crop_image(raw_image(r),r['box']),res,mean,std) for r in batch])
try:
 x=images([rr[i] for i in fixed]);f1=forward(x);f2=forward(x);assert torch.equal(f1,f2)
 t=time.monotonic();x=images([rr[i] for i in fixed]);prep=time.monotonic()-t;t=time.monotonic();fb=forward(x);timing=time.monotonic()-t
 projected=(prep+timing)*len(rr)/8
 check=dict(arm=arm,producer='extract.py',strict_loading_info=info,tensors=tensors,weight_sha256=sha(folder/'model.safetensors'),model_eval=not model.training,all_parameters_frozen=not any(p.requires_grad for p in model.parameters()),device=str(next(model.parameters()).device),dtype=str(next(model.parameters()).dtype),shape=list(f1.shape),including_CLS_shape=[8,785,768],final_LayerNorm_hook_exact=True,deterministic_exact=True,max_abs_difference=float((f1-f2).abs().max()),batch8_preprocessing_seconds=prep,batch8_forward_seconds=timing,projected_full_seconds=projected,attention_backend=model.config._attn_implementation,layer_norm_eps=model.config.layer_norm_eps,transformers_version=importlib.metadata.version('transformers'),feature_tap_comparison='D1 post-final-LayerNorm, CLS excluded and patch L2: same semantic tap',labels_used=False)
 dump(f'{arm}_checks.json',check)
 (OUT/f'{arm}_model_forward_source.txt').write_text(inspect.getsource(type(model).forward)+'\n'+inspect.getsource(type(model.embeddings).interpolate_pos_encoding))
except Exception as e:
 dump(f'{arm}_FAILED.json',dict(stage='forward sanity',error=repr(e),continue_other_arm=True));print('ARM STOPPED',arm,repr(e),flush=True);raise SystemExit(0)
print('CHECKS',arm,'batch8',timing,'projected',projected,flush=True)
if projected>5400:
 inds=np.sort(np.random.default_rng(SEED).choice(len(rr),200,replace=False));t=time.monotonic()
 for start in range(0,200,8):forward(images([rr[i] for i in inds[start:start+8]]))
 secs=time.monotonic()-t;check.update(subset_indices=inds.tolist(),subset_seconds=secs,subset_projected_full_seconds=secs*2341/200);dump(f'{arm}_checks.json',check)
 if check['subset_projected_full_seconds']>5400:
  dump(f'{arm}_FAILED.json',dict(stage='timing limit',error='200-slice measured projection exceeds90 minutes; full extraction not run',continue_other_arm=True));raise SystemExit(0)
dest=np.lib.format.open_memmap(OUT/f'{arm}_features.npy',mode='w+',dtype=np.float32,shape=(2341,784,768))
t=time.monotonic()
for start in range(0,2341,8):
 f=forward(images(rr[start:start+8]));assert torch.isfinite(f).all();dest[start:start+len(f)]=f.numpy()
 if start%256==0:print(arm,start,'seconds',time.monotonic()-t,flush=True)
dest.flush();hook.remove()
dump(f'{arm}_extract.json',dict(arm=arm,producer='extract.py',labels_used=False,shape=list(dest.shape),forward_extraction_seconds=time.monotonic()-t,total_seconds=elapsed(),checks_file=f'{arm}_checks.json',normalization=dict(mean=mean,std=std,application_count=1),ordered_ids_sha256=hashlib.sha256(json.dumps([r['sample_id'] for r in rr]).encode()).hexdigest()))
print('DONE',arm,elapsed(),flush=True)
