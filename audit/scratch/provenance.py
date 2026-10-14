import guard
from pathlib import Path
import json,ast,hashlib
from collections import Counter
root=guard.ROOT
out=[]
for base in ['Dataset','src/pytorch_data_dir','checkpoints']:
    files=[p for p in (root/base).rglob('*') if p.is_file()]
    out.append(base+' extensions '+str(Counter(''.join(p.suffixes) for p in files)))
for d in sorted((root/'src/pytorch_data_dir/preprocessed').glob('*')):
    imgs=list((d/'images').glob('*.png')); masks=list((d/'labels').glob('*.png'))
    out.append(f'{d.relative_to(root)} images={len(imgs)} labels={len(masks)} patients={sorted(set(p.name.split("_")[1] for p in imgs))} missing_masks={len({p.name for p in imgs}-{p.name for p in masks})}')
for p in sorted((root/'logs/wandb').glob('run-202609*/files/wandb-summary.json')):
    x=json.loads(p.read_text()); cfg=p.with_name('config.yaml').read_text()
    out.append(str(p.relative_to(root))+':1 '+str({k:v for k,v in x.items() if 'mIoU' in k or 'Accuracy' in k or k in ['trainer/global_step','epoch','cd/pos_intra','cd/pos_inter','cd/neg_inter']}))
for a,b in [('dino/vision_transformer.py','src/dino/vision_transformer.py'),('dino/utils.py','src/dino/utils.py'),('picie/backbone.py','src/picie/backbone.py'),('picie/fpn.py','src/picie/fpn.py')]:
    out.append(f'duplicate {a} {b}: byte-identical={(root/a).read_bytes()==(root/b).read_bytes()}')
for p in [root/'src/precompute_knns.log',root/'src/train_segmentation.log']:
    ls=p.read_bytes().decode(errors='replace').replace('\x00','').splitlines()
    hits=[f'{p.relative_to(root)}:{i} {l[:250]}' for i,l in enumerate(ls,1) if any(k in l for k in ['Loaded KNN','Saved NNs','already exists','Error','Traceback','torch.Size','All keys matched','Missing','Unexpected'])]
    out.extend(hits[-22:])
guard.record('Phase 2 provenance observations', '\n'.join(out))
