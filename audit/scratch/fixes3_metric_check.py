from fixes3_common import *
import argparse
from utils import UnsupervisedMetrics
p=argparse.ArgumentParser();p.add_argument('--fixed',action='store_true');args=p.parse_args()
m=UnsupervisedMetrics('',2,6,True)
conf=torch.tensor([[100,0]]*4+[[0,25]]*4)
for cluster,row in enumerate(conf):
 for label,n in enumerate(row.tolist()):m.update(torch.full((n,),cluster),torch.full((n,),label))
r=m.compute();mapping=m.map_clusters(torch.arange(8)).tolist()
print('K8 perfect split:',r,'mapping',mapping)
if args.fixed:
 assert mapping==[0,0,0,0,1,1,1,1] and r['mIoU']==100. and r['liver_IoU']==100. and r['liver_Dice']==100.
 assert m.histogram.shape==(2,2)
 n=UnsupervisedMetrics('',2,0,True);n.update(torch.tensor([1,1,0,0]),torch.tensor([0,0,1,1]));assert n.compute()['mIoU']==100. and n.map_clusters(torch.tensor([0,1])).tolist()==[1,0]
 # Empty/tied clusters map deterministically to class0; non-perfect counts validate real denominators.
 z=UnsupervisedMetrics('',2,6,True);z.stats.copy_(torch.tensor([[80,10],[5,30]]+[[0,0]]*6));z.update(torch.tensor([],dtype=torch.long),torch.tensor([],dtype=torch.long))
 rz=z.compute();assert abs(rz['liver_IoU']-100*30/45)<1e-4 and abs(rz['liver_Dice']-100*60/75)<1e-4
 print('PASS Hungarian K2; many-to-one K8; real-class denominators; absent/tied clusters')
else:assert mapping!=[0,0,0,0,1,1,1,1];print('REPRODUCED: extra clusters are not mapped by majority')
