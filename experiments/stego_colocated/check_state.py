"""Tiny synthetic next-update equality test; no dataset or label reads."""
from runtime import *
from model import StegoHead,objective
from reference_loss import ContrastiveCorrelationLoss
from state import capture,restore
from loss import ColocatedCorrelationLoss
import random

seed_all(123);c=read(PACKAGE/'P1/stage1/configs/locked.json');cfg=SimpleNamespace(**c)
model=StegoHead();opt=torch.optim.Adam(model.parameters(),lr=cfg.lr);rng=np.random.default_rng(123)
criterion=ColocatedCorrelationLoss(cfg,torch.ones(2,dtype=torch.bool));data=F.normalize(torch.randn(4,768,4,4),dim=1);order=rng.permutation(4)
def update(m,o):
    m.train();rows=rng.choice(4,2,replace=False);x=data[rows];xp=data[rows[::-1].copy()]
    f,z=m(x);fp,zp=m(xp);loss,_=objective(criterion,cfg,f,fp,z,zp)
    o.zero_grad(set_to_none=True);loss.backward();o.step()
    return float(loss.detach()),random.random(),float(np.random.rand())
update(model,opt);update(model,opt)
path=PACKAGE/'logs/state_test.pt';torch.save(capture(model,opt,rng,order,2,2,0),path)
expected=update(model,opt);expected_state={k:v.clone() for k,v in model.state_dict().items()}
other=StegoHead();other_opt=torch.optim.Adam(other.parameters(),lr=cfg.lr)
loaded=torch.load(path,map_location='cpu',weights_only=True)
restored=restore(loaded,other,other_opt,rng);actual=update(other,other_opt)
assert expected==actual and all(torch.equal(v,other.state_dict()[k]) for k,v in expected_state.items())
assert np.array_equal(restored[0],order) and restored[1:]==(2,2,0)
for i,s in opt.state_dict()['state'].items():
    for k,v in s.items():assert torch.equal(v,other_opt.state_dict()['state'][i][k])
dump('logs/state_verification.json',dict(bit_identical_next_update=True,expected=expected,actual=actual,model_equal=True,optimizer_equal=True,rng_equal=True,
  boundary='same CPU runtime/thread count; serialized model, Adam, torch/Python/NumPy/generator RNG, order/cursor, bad-check state',production_resume='forbidden; rerun fresh unique directory'))
ledger('logs/state_test_access.json');print('PASS: exact next-update equality')
