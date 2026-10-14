import guard
import sys,signal
signal.alarm(110)
sys.path.insert(0,str(guard.ROOT/'src'))
import torch,numpy as np
from collections import Counter
from PIL import Image
from omegaconf import OmegaConf
from data import CHAOS,ContrastiveSegDataset
from torchvision import transforms as T
ns=guard.definitions('src/utils.py',['normalize','ToTargetTensor','identity_transform','get_transform'],dict(torch=torch,np=np,T=T,Image=Image))
get_transform=ns['get_transform']
cfg=OmegaConf.load(guard.ROOT/'src/configs/train_config.yaml')
out=[]; sets={}
for split in ['train','val']:
    ds=ContrastiveSegDataset(cfg.pytorch_data_dir,'chaos',cfg.crop_type,split,get_transform(224,False,'center'),get_transform(224,True,'center'),cfg,num_neighbors=7,pos_images=True,mask=True)
    sets[split]=ds
    out.append(f'{split} dataset={len(ds)} patients_by_modality={dict((mod,sorted(set(s[3] for s in ds.dataset.samples if s[2]==mod))) for mod in CHAOS.MODALITIES)}')
    out.append(f'{split} prefixes={Counter(s[3].startswith("p") for s in ds.dataset.samples)}')
    ns=ds.nns
    if ns is not None:
        out.append(f'{split} nns={tuple(ns.shape)} min={ns.min().item()} max={ns.max().item()} out_of_range={int((ns>=len(ds)).sum())} first_col_not_self={int((ns[:,0]!=torch.arange(len(ns))).sum())} npz_keys={np.load(guard.ROOT/f"src/pytorch_data_dir/nns/nns_vit_base_chaos_{split}_five_224.npz").files}')
        if len(ds)>len(ns):
            try:ds[len(ns)]
            except Exception as e:out.append(f'{split} accessing first uncached item {len(ns)}: {type(e).__name__}: {e}')
    for idx in [0,len(ds)//2]:
        x,y,m=ds.dataset[idx]
        out.append(f'{split}[{idx}] {ds.dataset.samples[idx][0]} x={tuple(x.shape)}/{x.dtype} label={tuple(y.shape)}/{y.dtype} unique={torch.unique(y).tolist()} foreground={float((y==1).float().mean()):.5f} valid={float(m.mean()):.5f}')
    pids={s[3].lstrip('p').lstrip('0') for s in ds.dataset.samples};out.append(f'{split} canonical_pids={sorted(pids)}')
out.append('canonical CT overlap='+str({s[3].lstrip('p').lstrip('0') for s in sets['train'].dataset.samples}&{s[3].lstrip('p').lstrip('0') for s in sets['val'].dataset.samples}))
guard.record('Phase 3 data/cache runtime evidence', '\n'.join(out))
