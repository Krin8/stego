"""Read-only TensorBoard/W&B cross-check and checkpoint copy/hash provenance."""
from pathlib import Path
import os,json,hashlib,shutil,math,collections
os.environ.update(HF_HUB_OFFLINE='1',WANDB_MODE='offline')
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'audit/run_records/fixes3_full_C'
ev=json.loads((OUT/'postrun_evidence.json').read_text());rows=ev['validations']
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
tb=[]
for p in sorted(Path(ev['validation_source']).parent.glob('events.out*')):
 for e in EventFileLoader(str(p)).Load():
  for v in e.summary.value:
   if v.tag.startswith(('monitor/','test/cluster/')):
    if v.HasField('simple_value'):val=v.simple_value
    elif v.tensor.float_val:val=v.tensor.float_val[0]
    elif v.tensor.double_val:val=v.tensor.double_val[0]
    else:continue
    tb.append(dict(file=p.name,step=e.step,tag=v.tag,value=val))
(OUT/'tensorboard_validation_scalars.json').write_text(json.dumps(tb,indent=2)+'\n')
checks=[]
for row in rows:
 expected={**{k:v for k,v in row['metrics'].items() if k.startswith('test/cluster/')},**{'monitor/'+k:row[k] for k in ['code_std','cross_image_cosine','cluster_entropy']},**{f'monitor/cluster_usage_{i}':v for i,v in enumerate(row['cluster_usage'])}}
 for tag,v in expected.items():
  matches=[x for x in tb if x['tag']==tag and x['step']==row['batches']-1]
  checks.append(dict(batch=row['batches'],tag=tag,match=any(math.isclose(x['value'],v,abs_tol=1e-5) for x in matches)))
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2
wp=ROOT/'logs/wandb/offline-run-20260930_194801-liqimwvz/run-liqimwvz.wandb'
ds=DataStore();ds.open_for_scan(str(wp));wh=[];summary={};types=collections.Counter()
while True:
 blob=ds.scan_data()
 if blob is None:break
 record=wandb_internal_pb2.Record();record.ParseFromString(blob);kind=record.WhichOneof('record_type');types[kind]+=1
 if kind in ['history','summary']:
  items=record.history.item if kind=='history' else record.summary.update
  vals={}
  for item in items:
   key=item.key or '/'.join(item.nested_key)
   if key.startswith(('monitor/','test/cluster/')) or key in ['_step','trainer/global_step']:
    try: vals[key]=json.loads(item.value_json)
    except ValueError: vals[key]=item.value_json
  if kind=='history' and any(k.startswith(('monitor/','test/cluster/')) for k in vals):wh.append(vals)
  if kind=='summary':summary.update(vals)
(OUT/'wandb_validation_scalars.json').write_text(json.dumps(dict(source=str(wp),record_types=dict(types),history=wh,summary=summary),indent=2)+'\n')
last=ev['checkpoints'][-1];cb=next(v for v in last['callbacks'].values() if v.get('monitor')=='test/cluster/mIoU')
best=Path(cb['best_model_path']);copy=OUT/'best_checkpoint.ckpt'
if not copy.exists():
 with best.open('rb') as src,copy.open('xb') as dst:shutil.copyfileobj(src,dst)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
verification={name:dict(path=str(p),exists=p.is_file(),bytes=p.stat().st_size,sha256=digest(p)) for name,p in [('best_original',best),('best_audit_copy',copy),('final',Path(last['path']))]}
assert verification['best_original']['sha256']==verification['best_audit_copy']['sha256']
result=dict(script=str(Path(__file__).resolve()),tensorboard_checks=len(checks),tensorboard_mismatches=[x for x in checks if not x['match']],tensorboard_tags=dict(collections.Counter(x['tag'] for x in tb)),wandb_history_records=len(wh),wandb_summary=summary,checkpoint_verification=verification)
(OUT/'auxiliary_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
