"""Read-only rank-tie audit using the cache writer's actual similarity statements."""
from common import *
src=ast.parse((ROOT/'src/precompute_knns.py').read_text());app=next(n for n in src.body if isinstance(n,ast.FunctionDef) and n.name=='my_app')
loop=next(n for n in ast.walk(app) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='i')
code=compile(ast.Module(body=loop.body[:4],type_ignores=[]),str(ROOT/'src/precompute_knns.py'),'exec');result={}
for split in ['val','train']:
 f=torch.from_numpy(np.load(OUT/f'{split}_image_features.npy'));fresh=np.load(OUT/f'{split}_current30.npy')[:,:29];stored=np.load(OUT/f'{split}_cached29.npy');rows=np.flatnonzero(np.any(fresh!=stored,axis=1));records=[];step=max(1,len(f)//16)
 for row in rows:
  i=int(row)//step*step;scope=dict(normed_feats=f,step=step,i=i,device=torch.device('cpu'),torch=torch);exec(code,scope);sims=scope['pairwise_sims'][int(row)-i]
  for rank in np.flatnonzero(fresh[row]!=stored[row]):
   a=int(fresh[row,rank]);b=int(stored[row,rank]);records.append(dict(anchor=int(row),rank=int(rank)+1,recomputed=a,stored=b,recomputed_similarity=float(sims[a]),stored_similarity=float(sims[b]),absolute_difference=abs(float(sims[a])-float(sims[b]))))
 result[split]=dict(rows=len(rows),changed_rank_entries=len(records),details=records,max_similarity_difference=max([x['absolute_difference'] for x in records] or [0.]),same_all29_neighbour_sets=all(set(a)==set(b) for a,b in zip(fresh,stored)))
dump('rank_check.json',dict(script=__file__,labels_used=False,result=result,elapsed_seconds=elapsed()));print(json.dumps(result,indent=2))
