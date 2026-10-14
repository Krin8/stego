"""Reuse PAIRS sampling, strata and scoring functions; project loss remains actual."""
from common import *
import modules
def reuse_pair_function(name,namespace):
 node=next(n for n in ast.parse((PAIR/'purity.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(PAIR/'purity.py'),'exec'),namespace)
def initialize(split):
 inv=json.loads((FORE/'inventory.json').read_text());model,ck=load(inv['seed0']['path']);cfg=model.cfg
 assert cfg.pointwise and not cfg.stabalize and not cfg.use_salience and cfg.feature_samples==11
 records=json.loads((OUT/f'{split}_records.json').read_text());anchors=np.array(json.loads((OUT/f'{split}_anchors.json').read_text()))
 path=FORE/'val_features.npy' if split=='val' else BASE/'train_features.npy';features=torch.from_numpy(np.array(np.load(path,mmap_mode='r')))
 if split=='train':features=features.permute(0,2,1).reshape(-1,768,14,14)
 k=30 if split=='val' else 29
 current=np.load(OUT/f'{split}_current30.npy')[:,:k] if split=='val' else np.load(OUT/'train_cached29.npy')
 cross=np.load(OUT/f'{split}_cross30.npy')[:,:k]
 pid=np.array([r['patient_id'] for r in records]);series=np.array([r['series_uid'] for r in records]);rank=np.empty(len(records),int)
 for key in set(zip(pid,series)):
  indices=[i for i in range(len(records)) if (pid[i],series[i])==key];indices.sort(key=lambda i:records[i]['physical_position']);rank[indices]=np.arange(len(indices))
 return model,cfg,records,anchors,features,current,cross,k,pid,series,rank
def batches(anchors,k,seed_offset=0):
 rng=np.random.default_rng(SEED)
 for neighbor_rank in range(k):
  order=rng.permutation(anchors)
  for j,start in enumerate(range(0,len(order),16)):
   yield neighbor_rank,order[start:start+16],SEED+seed_offset+(neighbor_rank+1)*100+j
def image_id_codes(ids):return torch.from_numpy(ids.astype(np.float32)+1)[:,None,None,None].expand(len(ids),1,14,14)
