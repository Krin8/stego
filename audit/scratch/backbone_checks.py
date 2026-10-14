"""M2 before/after loader and CPU feature check; prints only, never writes artifacts."""
import guard, sys, signal, argparse
sys.modules['wandb'] = None
signal.alarm(110)
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms as T
import numpy as np
import open_clip.factory as factory
p = argparse.ArgumentParser(); p.add_argument('--expect-fixed', action='store_true'); args = p.parse_args()
torch.set_num_threads(4); torch.manual_seed(11)
loads = []; original = factory.load_checkpoint
def checked(*a, **kw):
    result = original(*a, **kw); loads.append((kw.get('strict', True), str(result))); return result
factory.load_checkpoint = checked
ns = guard.definitions('src/modules.py', ['BiomedCLIPFeaturizer'], dict(torch=torch, nn=nn, F=F, __file__=str(guard.ROOT/'src/modules.py')))
cfg = OmegaConf.load(guard.ROOT/'src/configs/train_config.yaml')
net = ns['BiomedCLIPFeaturizer'](cfg.dim, cfg).cpu()
initial_eval = not net.full_model.training
frozen = all(not p.requires_grad for p in net.full_model.parameters())
tns = guard.definitions('src/utils.py', ['normalize', 'ToTargetTensor', 'identity_transform', 'get_transform'], dict(torch=torch, np=np, T=T, Image=Image))
img = Image.open(guard.ROOT/'src/pytorch_data_dir/preprocessed/train/images/CT_28_0110.png').convert('RGB')
x = tns['get_transform'](224, False, 'center')(img)[None]
net.eval()
with torch.no_grad():
    f1, c1 = net(x); f2, c2 = net(x)
    expected = net.model.get_intermediate_layers(x, n=1, reshape=True, norm=True)[0]
    f320, c320 = net(tns['get_transform'](320, False, 'center')(img)[None])
err = float((f1 - expected).abs().max()); repeated = float((f1-f2).abs().max())
net.train()
with torch.no_grad(): net(x)
forward_eval = not net.full_model.training
ok = initial_eval and frozen and forward_eval and err == 0 and repeated == 0 and f1.shape == (1,768,14,14) and c1.shape == (1,70,14,14)
print(dict(fixed=ok, strict_load=loads, initial_full_model_eval=initial_eval, full_pretrained_model_frozen=frozen, full_model_eval_after_parent_train_forward=forward_eval, final_norm_max_error=err, eval_repeat_max_error=repeated, feature_shape=tuple(f1.shape), code_shape=tuple(c1.shape), shapes320=(tuple(f320.shape), tuple(c320.shape)), head_trainable=all(p.requires_grad for p in net.cluster1.parameters())))
if args.expect_fixed: assert ok
