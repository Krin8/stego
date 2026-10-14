import fixes2_guard as g
import torch, numpy as np
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(2);torch.manual_seed(11)
ns=g.definitions('src/modules.py',['norm','tensor_correlation','ContrastiveCorrelationLoss'],dict(torch=torch,nn=nn,F=F))
from types import SimpleNamespace
fn=ns['ContrastiveCorrelationLoss'](SimpleNamespace(pointwise=True,zero_clamp=True,stabalize=False))
f=torch.randn(2,3,2,2);o=torch.randn(2,3,2,2);c=torch.randn(2,4,2,2);d=torch.randn(2,4,2,2)
got=[float(fn.helper(a,b,x,y,s)[0].mean()) for a,b,x,y,s in [(f,f,c,c,.18),(f,o,c,d,.12),(f,-o,c,d,.46)]]
assert np.allclose(got,[-.20717449486255646,-.032486625015735626,.15386627614498138],atol=1e-7)
comment='Pinned-reference variant' in (g.ROOT/'src/modules.py').read_text()
print('Unchanged reference loss values:',got,'; explicit comment:',comment)
if '--expect-fixed' in __import__('sys').argv:assert comment
