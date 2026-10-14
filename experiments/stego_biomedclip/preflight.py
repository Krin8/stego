"""Seal the declared protocol and train-only neighbors before any optimizer step."""
from runtime import *
import platform

assert read(HERE/'logs/checks.json')['passed']
assert not (HERE/'configs/locked.json').exists(), 'Refuse to change sealed protocol'
c=read(HERE/'configs/proposed.json')
assert c['batches']==5000 and c['aug_alignment_weight']==0 and c['seeds']==[0,1,2]
c['status']='LOCKED'
c['preprocessing']=contract()
c['reference_commit']=read(HERE/'reference/manifest.json')['commit']
c['input_feature_contract']='Existing B2_448 final-LayerNorm tokens; CLS removed; per-patch L2 before head; verified against exact local backbone'
c['deviations_document']='FIDELITY.md'
baseline=baseline_seal()
rr=records();ids=[i for i,r in enumerate(rr) if r['split']=='train']
features=np.load(OLD/'B2_448_features.npy',mmap_mode='r')
assert features.shape==(2341,784,768)
pooled=[]
for i in ids:
    row=np.array(features[i]);assert np.isfinite(row).all()
    assert np.max(np.abs(np.linalg.norm(row,axis=-1)-1))<1e-4
    pooled.append(row.mean(0))
pooled=np.stack(pooled);pooled/=np.linalg.norm(pooled,axis=-1,keepdims=True)
sim=pooled@pooled.T;np.fill_diagonal(sim,-np.inf)
# Deterministic identity tie-break; remove self explicitly rather than assuming rank0.
neighbors=np.argsort(-sim,axis=1,kind='stable')[:,:30]
assert np.all(neighbors!=np.arange(len(ids))[:,None])
np.save(HERE/'logs/train_neighbors.npy',neighbors)
dump('logs/train_neighbors.json',dict(ordered_sample_ids=[rr[i]['sample_id'] for i in ids],
     k=30,sampling_topk=7,method='L2(global mean of normalized spatial tokens), cosine, self explicitly excluded',
     patients=sorted({rr[i]['patient'] for i in ids},key=int),labels_used=False))
dump('configs/locked.json',c)
(HERE/'configs/locked.sha256').write_text(sha(HERE/'configs/locked.json')+'\n')
weight=ROOT/'src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin'
digest=sha(weight);assert digest=='52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be'
dump('logs/preflight.json',dict(passed=True,backbone_trainable=0,
    backbone_sha256=digest,baseline_manifest_sha256=baseline,
    feature_cache_sha256=sha(OLD/'B2_448_features.npy'),
    feature_cache_shape=list(features.shape),train_slices=len(ids),validation_slices=472,
    source_sha256={str(p.relative_to(HERE)):sha(p) for p in HERE.glob('*.py')},
    config_sha256=sha(HERE/'configs/locked.json'),python=platform.python_version(),
    torch=torch.__version__,train_only_neighbor_sha256=sha(HERE/'logs/train_neighbors.npy')))
ledger('logs/preflight_access.json');print('PREFLIGHT SEALED; zero optimizer steps',flush=True)
