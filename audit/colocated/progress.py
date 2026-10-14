"""Read current label-free training monitors only."""
import json
from pathlib import Path
P=Path(__file__).resolve().parents[2]/'experiments/stego_colocated'
for arm in ['C448','P1']:
    for stage in ['stage1','stage2']:
        for seed in [0,1,2]:
            log=P/arm/stage/'logs'/f'seed{seed}_train.jsonl'
            if not log.exists():continue
            rows=[]
            for line in log.read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:pass
            if not rows:continue
            monitors=[r for r in rows if 'code_std' in r]
            latest=monitors[-1] if monitors else {}
            complete=log.with_name(f'seed{seed}_complete.json')
            done=json.loads(complete.read_text()) if complete.exists() else {}
            print(arm,stage,seed,'batch',rows[-1]['batch'],'monitor',latest.get('batch'),
                'std',latest.get('code_std'),'cos',latest.get('cross_image_cosine'),
                'healthy',latest.get('healthy'),'screen',done.get('screen_pass'),flush=True)
