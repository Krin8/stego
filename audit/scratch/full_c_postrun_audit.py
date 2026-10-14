"""Read-only full-C log/checkpoint audit; outputs only audit records."""
import os
os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
import json,re,collections
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'audit/run_records/fixes3_full_C'
LOGDIR=ROOT/'logs/chaos/chaos_fixes3_full_C_date_Sep30_19-47-58/lightning_logs/version_0'
CKPTDIR=ROOT/'checkpoints/chaos/chaos_fixes3_full_C_date_Sep30_19-47-58'
rows=[json.loads(s) for s in (LOGDIR/'validation_diagnostics.jsonl').read_text().splitlines()]
def simple(x):
 if isinstance(x,torch.Tensor):return x.item() if x.numel()==1 else {'tensor_shape':list(x.shape)}
 if isinstance(x,dict):return {str(k):simple(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [simple(v) for v in x]
 if isinstance(x,(str,int,float,bool)) or x is None:return x
 return str(x)
checkpoints=[]
for path in CKPTDIR.rglob('*.ckpt'):
 d=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
 callbacks=simple(d.get('callbacks',{}))
 row=dict(path=str(path),bytes=path.stat().st_size,global_step=d['global_step'],batches=d['global_step']//3,epoch=d['epoch'],callbacks=callbacks,loops=simple(d.get('loops',{})))
 hp=d.get('hyper_parameters',{})
 row['hparams']=simple(hp)
 row['preceding_validation']=max((v['batches'] for v in rows if v['optimizer_steps']<=d['global_step']),default=None)
 row['filename_score']=float(re.search(r'mIoU=([0-9.]+)\.ckpt$',path.name).group(1))
 checkpoints.append(row);del d
checkpoints.sort(key=lambda x:x['global_step'])
log=(OUT/'train.log').read_text()
warning_lines=[{'line':i,'text':s} for i,s in enumerate(log.splitlines(),1) if 'Warning' in s or 'Error' in s or 'Traceback' in s or 'Using ModelCheckpoint with manual optimization' in s or 'could not find' in s or 'Trainer.fit' in s]
result=dict(script=str(Path(__file__).resolve()),validation_source=str(LOGDIR/'validation_diagnostics.jsonl'),validations=rows,checkpoints=checkpoints,warnings=warning_lines,exit_code=(OUT/'exit_code.txt').read_text().strip(),seeds=[json.loads((ROOT/f'audit/scratch/fixes3/fixes3_C_seed{i}_result.json').read_text()) for i in range(3)])
(OUT/'postrun_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
print('validations',len(rows),'checkpoints',len(checkpoints))
for key,item in [('last',checkpoints[-1]),('max_filename',max(checkpoints,key=lambda c:c['filename_score']))]:
 print(key,json.dumps({k:v for k,v in item.items() if k not in ['loops','hparams']},indent=2))
print('last loops',json.dumps(checkpoints[-1]['loops'],indent=2))
print('distinct_warnings',*collections.Counter(x['text'] for x in warning_lines).items(),sep='\n')
