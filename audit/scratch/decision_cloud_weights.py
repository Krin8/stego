"""Target known cloud checkpoint roots; header-only local weight classification."""
import decision_guard as g
import os,stat,json,struct,time,signal
from pathlib import Path
from datetime import datetime,timezone
start=time.monotonic(); rows=['## m6 — Targeted cloud checkpoint inventory','Command: `python3 -B audit/scratch/decision_cloud_weights.py`. Targeted checkpoint/model folders avoid stalled traversal of unrelated cloud logs.','Path | Bytes | Modified UTC | State keys','--- | ---: | --- | ---'];found=[];coverage=[]
for drive in sorted((Path.home()/'Library/CloudStorage').glob('GoogleDrive-*')):
    for relative in ['My Drive/checkpoints','My Drive/STEGO/checkpoints','My Drive/STEGO/models']:
        base=drive/relative
        if not base.is_dir():continue
        count=0
        for directory,dirs,files in os.walk(base,followlinks=False):
            if time.monotonic()-start>70:raise RuntimeError('targeted inventory deadline')
            for name in files:
                if Path(name).suffix.lower() not in ('.ckpt','.pt','.pth','.safetensors'):continue
                p=Path(directory)/name;s=p.stat();offline=bool(s.st_flags & stat.SF_DATALESS)
                rows.append(f'`{p}` | {s.st_size} | {datetime.fromtimestamp(s.st_mtime,timezone.utc).isoformat()} | '+('ONLINE-ONLY; UNVERIFIED' if offline else 'locally available'))
                found.append(p);count+=1
        coverage.append((str(base),count))
g.append('\n'.join(rows)+'\n\nCompleted targeted roots/counts: `'+str(coverage)+'`.')
print(dict(targets=coverage,total=len(found),elapsed=round(time.monotonic()-start,2)))
# Read only locally available state metadata, never tensor payloads.
import sys
sys.modules['wandb']=None
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
torch.set_num_threads(2)
local=[g.ROOT/'models/model_epoch_0720_iter_085000.pth',Path.home()/'Documents/vscode/Python/Cosmetics/SkinCareModel.safetensors',g.ROOT/'src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin']
local += [p for p in found if not p.stat().st_flags & stat.SF_DATALESS]
for p in local:
    if not p.exists():continue
    result=dict(path=str(p),bytes=p.stat().st_size,mtime_utc=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat())
    try:
        if p.suffix=='.safetensors':
            with p.open('rb') as f:length=struct.unpack('<Q',f.read(8))[0];data=json.loads(f.read(length))
            keys=[k for k in data if k!='__metadata__'];shapes={k:data[k].get('shape') for k in keys[:12]}
        else:
            with FakeTensorMode():data=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
            sd=data.get('state_dict',data.get('model',data)) if isinstance(data,dict) else {}
            keys=list(sd);shapes={k:list(v.shape) for k,v in list(sd.items())[:12] if hasattr(v,'shape')}
            result['top_keys']=list(data)[:20]
        head=[k for k in keys if any(t in k for t in ['cluster_probe','linear_probe','cluster1','cluster2'])]
        result.update(key_count=len(keys),sample_keys=keys[:20],head_keys=head[:20],shapes=shapes,classification='trained STEGO-style head keys present' if head else 'No STEGO head keys; inspect architecture keys below')
    except Exception as e:result.update(classification='UNVERIFIED: safe metadata read failed',error=type(e).__name__+': '+str(e)[:1000])
    g.append('### Local weight metadata\n\n```json\n'+json.dumps(result,indent=2,default=str)+'\n```');print(json.dumps(result,default=str))
