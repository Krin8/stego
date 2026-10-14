"""Cross-check saved counts, thresholds, metric definitions and cohort coverage."""
from common import *
from sklearn.metrics import roc_auc_score,average_precision_score
meta=json.loads((OUT/'affinities.json').read_text());ex=json.loads((OUT/'extract.json').read_text())
assert ex['final_max_abs_difference']==0 and ex['train_mean_exactly_verified'] and ex['kmeans_nearest_upsampling_exactly_verified']
assert ex['backbone_eval'] and ex['backbone_frozen'];assert not meta['labels_used'] and not ex['labels_used']
labels=torch.from_numpy(np.array(np.load(FORE/'val_labels.npy',mmap_mode='r'),dtype=np.float32))[:,None]
patch=(F.avg_pool2d(labels,16)>.5)[:,0].numpy().reshape(472,196)
checks={}
for pool in meta['pools']:
 d=json.loads((OUT/f'score_{pool}.json').read_text());a=np.load(OUT/f'{pool}_a.npy');b=np.load(OUT/f'{pool}_b.npy')
 ta=np.load(OUT/f'{pool}_ta.npy');tb=np.load(OUT/f'{pool}_tb.npy')
 assert np.array_equal(np.bincount(a),np.full(472,7))
 assert np.array_equal(a,b) if pool=='intra' else np.all(a!=b)
 anchor=patch[a[:,None],ta].reshape(-1);target=patch[b[:,None],tb].reshape(-1)
 for variant in meta['variants']:
  r=d['results'][variant];x=np.load(OUT/f'scores_{variant}_{pool}.npy',mmap_mode='r').reshape(-1)
  assert x.size==r['pairs']==6766592 and r['liver_anchor_pairs']==int(anchor.sum())
  assert abs(roc_auc_score(target[anchor],x[anchor])-r['auc'])<1e-12
  assert abs(average_precision_score(target[anchor],x[anchor])-r['average_precision'])<1e-12
  for row in r['sweep']:
   selected=x>row['threshold'];n=int(selected.sum());c=row['counts_BB_BL_LB_LL'];assert sum(c)==n
   assert abs(row['attraction']-selected.mean())<1e-12
   if row['budget'] is not None:assert abs(row['attraction']-row['budget'])<1e-5
   tp=int((selected&anchor&target).sum());fp=int((selected&anchor&~target).sum());fn=int((~selected&anchor&target).sum())
   assert tp==c[3] and fp==c[2]
   assert abs(row['liver_target_precision']-tp/(tp+fp))<1e-12
   assert abs(row['liver_pair_recall']-tp/(tp+fn))<1e-12
  for point in r['precision_envelopes'].values():assert point['actual_recall']>=point['target_recall']
 checks[pool]=dict(all_472_anchors_appear_seven_times=True,pairs=6766592,liver_anchor_pairs=int(anchor.sum()),liver_liver_pairs=int((anchor&target).sum()),background_target_pairs=int((anchor&~target).sum()),independent_boolean_scoring_matches=True)
 assert d['elapsed_seconds']<570
assert meta['elapsed_seconds']<570 and ex['elapsed_seconds']<570
dump('verify.json',dict(script=__file__,labels='validation scoring verification only',checks=checks,elapsed_seconds=elapsed()));print(json.dumps(checks,indent=2))
