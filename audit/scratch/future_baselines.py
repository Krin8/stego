"""NOT RUN in audit. Opt-in supervised plumbing controls, not STEGO scores.

Four training slices, four held-out-patient slices. No checkpoint/file output.
These test whether frozen patch features support a tiny supervised decoder.
"""
import argparse
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode', choices=['overfit', 'shuffled-label', 'random-feature'], required=True)
p.add_argument('--steps', type=int, default=1000)
p.add_argument('--allow-training', action='store_true')
a = p.parse_args()
if not a.allow_training:
    p.error('This expensive diagnostic was not authorized/run during the audit; opt in explicitly.')
import guard, sys
sys.modules['wandb'] = None
import torch, numpy as np
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from omegaconf import OmegaConf
sys.path.insert(0, str(guard.ROOT / 'src'))
from data import CHAOS
torch.set_num_threads(4)
torch.manual_seed(0)
cfg = OmegaConf.load(guard.ROOT / 'src/configs/train_config.yaml')
tns = guard.definitions('src/utils.py', ['normalize', 'ToTargetTensor', 'identity_transform', 'get_transform'], dict(torch=torch, np=np, T=T, Image=Image))
ns = guard.definitions('src/modules.py', ['BiomedCLIPFeaturizer'], dict(torch=torch, nn=nn, F=F, __file__=str(guard.ROOT / 'src/modules.py')))
net = ns['BiomedCLIPFeaturizer'](70, cfg).cpu().eval()
def collect(split):
    ds = CHAOS(cfg.pytorch_data_dir, 'CT', split, tns['get_transform'](224, False, 'center'), tns['get_transform'](224, True, 'center'), 2, True)
    chosen = {}
    for i, sample in enumerate(ds.samples):
        if '/CT_p' not in sample[0]:
            chosen.setdefault(sample[3], []).append(i)
    xs, ys = [], []
    for patient in sorted(chosen)[:4]:
        x, y, _ = ds[chosen[patient][len(chosen[patient]) // 2]]
        xs.append(x); ys.append(y)
    with torch.no_grad():
        feats = torch.cat([net(x[None])[0] for x in xs])
    return feats, torch.stack(ys)
f, y = collect('train'); vf, vy = collect('val')
if a.mode == 'random-feature':
    f, vf = torch.randn_like(f), torch.randn_like(vf)
target = y.clone()
if a.mode == 'shuffled-label':
    for i in range(len(target)):
        flat = target[i].flatten()
        valid = torch.where(flat >= 0)[0]
        flat[valid] = flat[valid[torch.randperm(len(valid))]].clone()
head = nn.Sequential(nn.Conv2d(768, 70, 1), nn.ReLU(), nn.Conv2d(70, 2, 1))
opt = torch.optim.Adam(head.parameters(), lr=5e-4)
for _ in range(a.steps):
    opt.zero_grad()
    logits = F.interpolate(head(f), y.shape[-2:], mode='bilinear', align_corners=False)
    loss = F.cross_entropy(logits, target, ignore_index=-1)
    loss.backward(); opt.step()
def score(feats, labels):
    with torch.no_grad():
        pred = F.interpolate(head(feats), labels.shape[-2:], mode='bilinear', align_corners=False).argmax(1)
    valid = labels >= 0
    ious = [float(((pred == c) & (labels == c) & valid).sum() / (((pred == c) | (labels == c)) & valid).sum().clamp_min(1)) for c in range(2)]
    return dict(background_iou=ious[0], liver_iou=ious[1], miou=sum(ious)/2)
print(dict(mode=a.mode, train=score(f, y), held_out_patients=score(vf, vy), final_target_ce=float(loss.detach())))
