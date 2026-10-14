from fixes3_common import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--fixed',action='store_true');args=p.parse_args()
if not args.fixed:
    text=(ROOT/'src/train_segmentation.py').read_text()
    assert 'CollapseMonitor' not in text and 'collapse_early_stop' not in text
    print('REPRODUCED: validation lacks code-collapse monitoring and early-stop switch');raise SystemExit
from collapse_monitor import CollapseMonitor,CollapseStopper
m=CollapseMonitor(8);m.update(torch.ones(4,70,14,14),torch.zeros(4,224,224,dtype=torch.long));r=m.compute()
assert abs(r['cross_image_cosine']-1)<1e-6 and r['code_std']<1e-6 and r['cluster_usage']==[1.]+[0.]*7
s=CollapseStopper(.99,.01,3)
assert [s.check(r) for _ in range(3)]==[False,False,True]
torch.manual_seed(3);healthy=torch.randn(4,70,14,14);pred=torch.arange(4*224*224).reshape(4,224,224)%8
m=CollapseMonitor(8);m.update(healthy,pred);h=m.compute();assert h['cross_image_cosine']<.99 and h['code_std']>.01
assert not s.check(h) and s.bad_checks==0
# Accumulation must match direct all-image/all-pixel computation across arbitrary batches.
b=CollapseMonitor(8);b.update(healthy[:1],pred[:1]);b.update(healthy[1:],pred[1:]);br=b.compute()
assert abs(br['code_std']-h['code_std'])<1e-10 and abs(br['cross_image_cosine']-h['cross_image_cosine'])<1e-10
f=np.load(ROOT/'audit/scratch/diag/train_block12.npy',mmap_mode='r')[:20]
f=torch.from_numpy(np.array(f)).transpose(1,2).reshape(20,768,14,14)
m=CollapseMonitor(8);m.update(f,torch.zeros(20,14,14,dtype=torch.long));backbone=m.compute()
print('PASS: constant codes stop on third check; healthy reset; batch-invariant statistics')
print('Healthy random70:',h['code_std'],h['cross_image_cosine'])
print('Frozen20train reference768:',backbone['code_std'],backbone['cross_image_cosine'])
