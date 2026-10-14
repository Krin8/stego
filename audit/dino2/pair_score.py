from common import *
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve
reuse(SELECT/'score.py',['basic','envelope','summarize'],globals())
rr=records();vi=[i for i,r in enumerate(rr) if r['split']=='val'];masks=np.load(OLD/'scoring_masks_native.npy',mmap_mode='r');patch=[]
for i in vi:
 crop=crop_image(masks[i],rr[i]['box']);t=F.interpolate(torch.from_numpy(crop)[None,None],(448,448),mode='nearest')
 patch.append((F.avg_pool2d(t,16)[0,0]>.5).numpy().reshape(784))
patch=np.asarray(patch,np.uint8);a=np.load(OUT/'pair_a.npy');ta=np.load(OUT/'pair_ta.npy');tb=np.load(OUT/'pair_tb.npy')
cat=(2*patch[a[:,None],ta]+patch[a[:,None],tb]).reshape(-1)
meta=json.loads((OUT/'pair_affinities.json').read_text());result={}
for name in meta['sources']:
 x=np.asarray(np.load(OUT/f'pair_scores_{name}.npy',mmap_mode='r')).reshape(-1)
 result[name]=summarize(cat,x,meta['thresholds'][name],detail=False)
 print(name,result[name]['auc'],result[name]['average_precision'],flush=True)
dump('pair_selectivity.json',dict(producer='pair_score.py',labels='Only472 validation masks for diagnostic AUC/AP/precision/recall; thresholds/affinities were saved label-free first',label_grid='common body-crop nearest448 then16x16 majority ->28x28; ties background; PAIRS actual nearest sampler',results=result,seconds=elapsed()))
