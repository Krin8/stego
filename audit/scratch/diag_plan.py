"""Fix label-blind patient/position-stratified samples; verify the existing cache."""
from diag_common import *
from knn_cache import expected_identity,cache_path,read_cache
from collections import Counter
rng=np.random.default_rng(SEED)
plan={'seed':SEED,'selection':'equal patient quotas; one seeded random choice per equal slice-position stratum; no labels read','scripts':{'selection':__file__}}
sets={}
for split,n in [('train',200),('val',100)]:
    ds=dataset(split);sets[split]=ds
    patients=sorted(set(r['patient_id'] for r in ds.records),key=int)
    ix=[]
    for pidx,p in enumerate(patients):
        pool=[i for i,r in enumerate(ds.records) if r['patient_id']==p]
        quota=n//len(patients)+(pidx<n%len(patients))
        for stratum in np.array_split(pool,quota):ix.append(int(rng.choice(stratum)))
    plan[split]=[dict(index=i,**ds.records[i]) for i in ix]
    plan[split+'_counts']=dict(Counter(ds.records[i]['patient_id'] for i in ix))
ds=sets['train'];identity=expected_identity(ds,cfg());path=cache_path(ds.root,identity)
nns=read_cache(path,identity)
selected=[r['index'] for r in plan['train']];lookup={idx:i for i,idx in enumerate(selected)}
edges=[]
for i,idx in enumerate(selected):
    for rank,j in enumerate(nns[idx,1:cfg().num_neighbors+1],start=1):
        if int(j) in lookup:edges.append([i,lookup[int(j)],rank])
plan['knn_pairs']=edges
plan['knn_cache']=dict(path=str(path.relative_to(ROOT)),identity_sha256=__import__('chaos_preprocessing').digest(identity),available_anchor_count=len(set(e[0] for e in edges)),edge_count=len(edges),policy='Use all directed existing-cache edges in training ranks1..7 whose endpoints are BOTH in the fixed 200-slice sample; do not recompute neighbors or add slices')
plan['visual_indices']=[next(i for i,r in enumerate(plan['val']) if r['patient_id']==p)+12 for p in sorted(plan['val_counts'],key=int)]
plan['collapse_indices']=[i for p in sorted(plan['val_counts'],key=int) for i in [j for j,r in enumerate(plan['val']) if r['patient_id']==p][::5]]
plan['task3_limit']='Only four CT validation patients exist; 20 fixed slices across all four. Ten validation patients cannot be tested without changing the split.'
plan['elapsed_seconds']=elapsed();dump('plan.json',plan)
summary={k:plan[k] for k in ['seed','train_counts','val_counts','knn_cache','visual_indices','collapse_indices','task3_limit','elapsed_seconds']}
print(json.dumps(summary,indent=2))
(ROOT/'audit/DIAG.md').write_text('# CHAOS frozen-feature and collapse diagnostics\n\nIN PROGRESS. Read-only source/data/cache/checkpoint investigation. Diagnostic scripts: `audit/scratch/diag_*.py`; intermediate arrays/results: `audit/scratch/diag/`. All scripts enforce CPU, offline access, write confinement and a570-second timeout.\n\n## Fixed protocol\n\n'+json.dumps(summary,indent=2)+'\n\nLabels do not select samples, sources, shifts, k-means or PCA. The user explicitly authorized isolated supervised linear fitting. No project training loop or model update is run. Layers are numbered1–12: final=12, candidates11,10,9,8. Existing final-layer kNN cache is reused for all candidates; no source-specific cache is generated. Majority/Hungarian maps computed from validation labels are scoring-only oracle maps, not deployable label assignments.\n')
