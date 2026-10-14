"""Small provenance and corrected-mask counts, reusing the sampled paths."""
import decision_guard as g
import json, re, ast, collections, subprocess
from datetime import datetime, timezone
import yaml
import numpy as np
from PIL import Image
report=g.REPORT.read_text()
section=report.split('### Sample physical geometry evidence',1)[1]
geometry=json.loads(section.split('```json\n',1)[1].split('\n```',1)[0])
counts={}; directions={}
for sequence in ['CT','T1-in','T1-out','T2']:
    totals=collections.Counter(); order=collections.Counter()
    for r in geometry:
        if r['sequence']!=sequence:continue
        labels,n=np.unique(np.array(Image.open(g.ROOT/r['mask']).convert('L')),return_counts=True)
        totals.update({int(a):int(b) for a,b in zip(labels,n)})
        order[r['filename_order_physical_direction']]+=1
    counts[sequence]=dict(totals);directions[sequence]=dict(order)
runs=[]
for name in ['run-20260426_000459-5e6nhit6','run-20260909_044210-y3l0vxcn']:
    folder=g.ROOT/'logs/wandb'/name/'files'
    meta=json.loads((folder/'wandb-metadata.json').read_text())
    raw=yaml.safe_load((folder/'config.yaml').read_text())['cfg']['value']
    cfg=ast.literal_eval(raw) if isinstance(raw,str) else raw
    commit=meta.get('git',{}).get('commit','')
    result=subprocess.run(['git','cat-file','-t',commit],cwd=g.ROOT,capture_output=True,text=True,timeout=5)
    runs.append(dict(run=name,metadata=meta,config={k:cfg.get(k,'UNVERIFIED: absent') for k in ['arch','model_type','dino_patch_size','dim','extra_clusters','chaos_n_classes','use_text_prompts','max_steps','batch_size','res','chaos_modality','use_preprocessed_data','loader_crop_type','crop_type','crop_ratio','shallow_layer_n','pytorch_data_dir']},summary=json.loads((folder/'wandb-summary.json').read_text()),git_object_check=dict(returncode=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip())))
# Metadata only: do not open online-only files.
cloud=report.split('## m6 — Targeted cloud checkpoint inventory',1)[1].split('### Local weight metadata',1)[0]
rows=[line for line in cloud.splitlines() if line.startswith('`/')]
times=[line.split(' | ')[2] for line in rows]
matches=[line for line in rows if 'Apr26_00-04-56' in line and 'epoch=21-' in line]
result=dict(corrected_mask_counts=counts,filename_order_directions=directions,cloud_files=len(rows),cloud_mtime_range=[min(times),max(times)],epoch21_candidate=matches,runs=runs)
g.append('## Additional provenance and corrected sample counts\n\nCommand: `python3 -B audit/scratch/decision_followup.py`. Counts reuse the same masks; T1-out uses only the18 physical matches. Cloud dates are metadata, not proof of training date. Git object checks are read-only.\n\n```json\n'+json.dumps(result,indent=2)+'\n```')
print(json.dumps({k:v for k,v in result.items() if k!='runs'}))
for r in runs:print(json.dumps({k:v for k,v in r.items() if k!='metadata'}))
