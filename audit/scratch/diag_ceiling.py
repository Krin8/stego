"""Fixed-seed k-means/PCA and explicitly authorized diagnostic logistic probe; no project training."""
from diag_common import *
import argparse,warnings
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from PIL import Image,ImageDraw
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--block',type=int,required=True);p.add_argument('--center',type=int,choices=[0,1],required=True);args=p.parse_args()
plan=json.loads((OUT/'plan.json').read_text())
train=np.array(np.load(OUT/f'train_block{args.block}.npy',mmap_mode='r')).reshape(-1,768)
val=np.array(np.load(OUT/f'val_block{args.block}.npy',mmap_mode='r')).reshape(-1,768)
mean=train.mean(axis=0,dtype=np.float64).astype(np.float32)
if args.center:train-=mean;val-=mean
train/=np.maximum(np.linalg.norm(train,axis=1,keepdims=True),1e-12)
val/=np.maximum(np.linalg.norm(val,axis=1,keepdims=True),1e-12)
yval=np.load(OUT/'val_labels.npy',mmap_mode='r');assert set(np.unique(yval))<={0,1}
def scores(pred):
    truth=np.asarray(yval)==1;pred=np.asarray(pred)==1
    tp=int((truth&pred).sum());fp=int((~truth&pred).sum());fn=int((truth&~pred).sum())
    return dict(liver_iou=tp/max(1,tp+fp+fn),liver_dice=2*tp/max(1,2*tp+fp+fn),tp=tp,fp=fp,fn=fn)
def upsample_assignments(pred):
    return F.interpolate(torch.from_numpy(pred.reshape(100,1,14,14).astype(np.float32)),(224,224),mode='nearest')[:,0].numpy().astype(np.int64)
results={};warnings_list=[]
for k in [2,4,8]:
    km=KMeans(n_clusters=k,random_state=SEED,n_init=3,max_iter=100,tol=1e-4,algorithm='lloyd')
    with warnings.catch_warnings(record=True) as ws:km.fit(train)
    warnings_list.extend(str(w.message) for w in ws)
    pred=upsample_assignments(km.predict(val))
    conf=np.bincount(pred.ravel()*2+np.asarray(yval).ravel(),minlength=k*2).reshape(k,2)
    # Validation-label oracle ONLY for reporting, not fitting or choosing unsupervised parameters.
    mapping=(conf[:,1]>conf[:,0]).astype(int)
    res=dict(majority_oracle=scores(mapping[pred]),majority_map=mapping.tolist(),confusion_cluster_by_label=conf.tolist(),iterations=int(km.n_iter_),inertia=float(km.inertia_))
    if k==2:
        r,c=linear_sum_assignment(conf,maximize=True);hung=np.zeros(2,dtype=int);hung[r]=c
        res['hungarian_oracle']=scores(hung[pred]);res['hungarian_map']=hung.tolist()
    results[str(k)]=res
    print('kmeans',args.block,args.center,k,res['majority_oracle'], 'elapsed',elapsed(),flush=True)
# Supervised diagnostic only: fixed hyperparameters, no validation tuning.
ytrain=np.load(OUT/'train_labels.npy',mmap_mode='r')
yfit=F.interpolate(torch.from_numpy(np.array(ytrain).astype(np.float32))[:,None],(14,14),mode='nearest')[:,0].numpy().astype(int).ravel()
probe=LogisticRegression(C=1.0,class_weight='balanced',solver='lbfgs',max_iter=200,tol=1e-5,random_state=SEED)
with warnings.catch_warnings(record=True) as ws:probe.fit(train,yfit)
warnings_list.extend(str(w.message) for w in ws)
logits=probe.decision_function(val).reshape(100,1,14,14)
up=F.interpolate(torch.from_numpy(logits),(224,224),mode='bilinear',align_corners=False)[:,0].numpy()
linear=scores(up>0)
linear.update(iterations=probe.n_iter_.tolist(),fit_patch_count=len(yfit),fit_liver_patches=int(yfit.sum()),class_weight='balanced',C=1.,threshold=0.,model_saved=False)
print('linear',args.block,args.center,linear,'elapsed',elapsed(),flush=True)
# PCA basis and display range fit exclusively to the fixed training features.
pca=PCA(n_components=3,svd_solver='randomized',random_state=SEED)
train_rgb=pca.fit_transform(train);v_rgb=pca.transform(val).reshape(100,14,14,3)
lo=np.percentile(train_rgb,1,axis=0);hi=np.percentile(train_rgb,99,axis=0)
from chaos_preprocessing import MEAN,STD
imgs=np.load(OUT/'val_images.npy',mmap_mode='r');paths=[]
for idx in plan['visual_indices']:
    rgb=np.clip((v_rgb[idx]-lo)/np.maximum(hi-lo,1e-12),0,1)
    vis=Image.fromarray(np.round(rgb*255).astype(np.uint8)).resize((224,224),resample=Image.Resampling.NEAREST)
    base=np.clip(np.asarray(imgs[idx]).transpose(1,2,0)*np.array(STD)+np.array(MEAN),0,1)
    overlay=base.copy();mask=np.asarray(yval[idx])==1;overlay[mask]=.55*base[mask]+.45*np.array([1,0,0])
    canvas=Image.new('RGB',(672,255),'white');draw=ImageDraw.Draw(canvas)
    canvas.paste(Image.fromarray(np.round(base*255).astype(np.uint8)),(0,31))
    canvas.paste(Image.fromarray(np.round(overlay*255).astype(np.uint8)),(224,31));canvas.paste(vis,(448,31))
    draw.text((4,6),plan['val'][idx]['stem'],fill='black');draw.text((230,6),'Ground truth overlay',fill='black');draw.text((453,6),f'PCA block{args.block} mean{args.center}',fill='black')
    name=f'pca_b{args.block}_c{args.center}_{plan["val"][idx]["stem"]}.png';canvas.save(OUT/name);paths.append(str((OUT/name).relative_to(ROOT)))
result=dict(script=__file__,block=args.block,mean_subtracted=bool(args.center),seed=SEED,train_patch_count=len(train),val_patch_count=len(val),normalization='LayerNorm tokens; optional training mean subtraction; then per-patch L2',kmeans=results,linear=linear,pca=dict(explained_variance_ratio=pca.explained_variance_ratio_.tolist(),paths=paths,fit='train-only PCA and train-only1/99percentile color bounds'),warnings=warnings_list,elapsed_seconds=elapsed())
dump(f'ceiling_b{args.block}_c{args.center}.json',result)
with (ROOT/'audit/DIAG.md').open('a') as out:out.write(f'\nCeiling block{args.block}, centered{args.center}: `audit/scratch/diag/ceiling_b{args.block}_c{args.center}.json` generated by `{__file__}`, elapsed{elapsed()}s.\n')
print(json.dumps(dict(block=args.block,center=args.center,linear=linear,warnings=warnings_list,elapsed_seconds=elapsed()),indent=2))
