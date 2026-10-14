"""Label-free fit/name/predict stages. Evaluation lives in a separate process."""
import os,sys,argparse
p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','name','predict']);p.add_argument('--seed',type=int,choices=[0,1,2],required=True);args=p.parse_args();os.environ['STEGO_STAGE']={'fit':'fitting','name':'naming','predict':'inference'}[args.stage]
from runtime import *
from model import StegoHead
assert read(PACKAGE/'screen_decision.json')[read(HERE/'configs/locked.json')['candidate']]['passed'], 'Stage 1 gate not passed'
from sklearn.cluster import KMeans
import pydicom
c=read(HERE/'configs/locked.json');assert c['status']=='LOCKED' and sha(HERE/'configs/locked.json')==(HERE/'configs/locked.sha256').read_text().strip();seed_all(args.seed);rr=records();ids=[i for i,r in enumerate(rr) if r['split']=='train'];val=[i for i,r in enumerate(rr) if r['split']=='val'];key=f'seed{args.seed}';out=HERE/f'predictions/seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
checkpoint=HERE/f'checkpoints/{key}/final.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=True);assert state['config_sha256']==sha(HERE/'configs/locked.json');model=StegoHead().eval().requires_grad_(False);loaded=model.load_state_dict(state['state_dict'],strict=True);assert not loaded.missing_keys and not loaded.unexpected_keys
features=np.load(OLD/'B2_448_features.npy',mmap_mode='r');G=28

def embedding(i):
 f=torch.from_numpy(np.array(features[i])).reshape(28,28,768).permute(2,0,1)[None]
 with torch.no_grad():z=model.embed(f)[0].permute(1,2,0).contiguous().numpy()
 assert z.shape==(G,G,70);return z

def save_meta(name,value):dump(str((out/name).relative_to(HERE)),value)
def score_fields(z,im,r,centers,selected,temp):
 # Exact inherited F1 operations; the only generalization is actual grid size.
 d=distances(z.reshape(-1,70),centers);assignment=d.argmin(1).reshape(G,G).astype(np.uint8);native=inverse_grid(assignment,r,outside=8).astype(np.uint8);result={}
 weights=np.exp(-(d-d.min(1,keepdims=True))/temp);weights/=weights.sum(1,keepdims=True)
 for rule,cluster in selected.items():
  hard=(native==cluster) if cluster is not None else np.zeros(tuple(r['shape']),bool)
  field=weights[:,cluster].reshape(G,G) if cluster is not None else np.zeros((G,G),np.float32);soft=inverse_grid(field,r,mode='bilinear',outside=0)
  pred=largest(guided(im,soft)>=.5,np.ones((3,3),int));pred=ndi.binary_fill_holes(pred);yy,xx=np.mgrid[-3:4,-3:4];pred=ndi.binary_closing(np.pad(pred,3),structure=xx*xx+yy*yy<=9)[3:-3,3:-3]
  result[rule]=(hard,pred)
 return assignment,native,result

if args.stage=='fit':
 assert not(out/'centers.npy').exists();sample=np.lib.format.open_memmap(out/'fit_sample.npy',mode='w+',dtype=np.float32,shape=(len(ids)*784,70));start=time.monotonic()
 for j,i in enumerate(ids):
  z=embedding(i);sample[j*784:(j+1)*784]=z.reshape(784,70)
  if j%200==0:print('FIT_FEATURES',key,j,len(ids),time.monotonic()-start,flush=True)
 sample.flush();km=KMeans(n_clusters=8,random_state=args.seed,n_init=c['kmeans_n_init'],max_iter=c['kmeans_max_iter'],tol=c['kmeans_tol'],algorithm=c['kmeans_algorithm']).fit(sample);np.save(out/'centers.npy',km.cluster_centers_);d=distances(sample,km.cluster_centers_);two=np.partition(d,1,axis=1)[:,:2];temperature=max(float(np.median(two[:,1]-two[:,0])/np.log(4)),1e-6)
 save_meta('fit.json',dict(fit_patients=sorted({rr[i]['patient'] for i in ids},key=int),fit_samples=len(ids),spatial_samples=len(sample),K=8,seed=args.seed,centers_sha256=sha(out/'centers.npy'),checkpoint_sha256=sha(checkpoint),temperature=temperature,inertia=float(km.inertia_),iterations=int(km.n_iter_),seconds=time.monotonic()-start,sampling='all patches',sample_sha256=sha(out/'fit_sample.npy'),labels_read=False))
elif args.stage=='name':
 assert not(out/'naming.json').exists();centers=np.load(out/'centers.npy');fit=read(out/'fit.json');assert sha(checkpoint)==fit['checkpoint_sha256'];sums=np.zeros((8,4),np.float64);nearest=[[] for _ in range(8)];start=time.monotonic();fallback_count=0
 yy,xx=np.mgrid[-5:6,-5:6];disk=xx*xx+yy*yy<=25
 for n,i in enumerate(ids):
  r=rr[i];ds=pydicom.dcmread(r['dicom']);hu=ds.pixel_array.astype(np.float32)*float(getattr(ds,'RescaleSlope',1))+float(getattr(ds,'RescaleIntercept',0));body=ndi.binary_fill_holes(largest(ndi.binary_opening(hu>-500,structure=disk),np.ones((3,3),int)));ys,xs=np.where(body);assert len(ys)
  y0,x0,side=r['box'];gy,gx=np.mgrid[:G,:G];cy=y0+(gy.ravel()+.5)*side/G-.5;cx=x0+(gx.ravel()+.5)*side/G-.5;iy=np.floor(cy+.5).astype(int);ix=np.floor(cx+.5).astype(int);eligible=(iy>=0)&(ix>=0)&(iy<512)&(ix<512);valid=np.flatnonzero(eligible);eligible[valid]=body[iy[valid],ix[valid]];use=np.flatnonzero(eligible)
  ya=np.clip(np.ceil(y0+gy.ravel()*side/G-.5).astype(int),0,512);yb=np.clip(np.ceil(y0+(gy.ravel()+1)*side/G-.5).astype(int),0,512);xa=np.clip(np.ceil(x0+gx.ravel()*side/G-.5).astype(int),0,512);xb=np.clip(np.ceil(x0+(gx.ravel()+1)*side/G-.5).astype(int),0,512)
  def rectsum(a):
   integral=np.pad(a,((1,0),(1,0))).cumsum(0,dtype=np.float64).cumsum(1,dtype=np.float64);return integral[yb,xb]-integral[ya,xb]-integral[yb,xa]+integral[ya,xa]
  count=rectsum(body);total=rectsum(hu*body);means=np.divide(total,count,out=np.zeros_like(total),where=count>0);empty=use[count[use]==0];means[empty]=hu[iy[empty],ix[empty]];fallback_count+=len(empty)
  z=embedding(i).reshape(-1,70);d=distances(z,centers);assignment=d.argmin(1);nx=(cx-xs.min())/max(1,xs.max()-xs.min());ny=(cy-ys.min())/max(1,ys.max()-ys.min())
  for cluster in range(8):
   u=use[assignment[use]==cluster];sums[cluster,0]+=len(u);sums[cluster,1:]+=[nx[u].sum(),ny[u].sum(),means[u].sum()]
   v=d[use,cluster];cut=np.partition(v,min(199,len(v)-1))[min(199,len(v)-1)];pool=use[v<=cut];order=pool[np.lexsort((pool,d[pool,cluster]))][:200]
   candidates=[(float(d[t,cluster]),i,int(t),float(cx[t]),float(cy[t])) for t in order];nearest[cluster]=sorted(nearest[cluster]+candidates)[:200]
  if n%200==0:print('NAME_NEAREST',key,n,len(ids),time.monotonic()-start,flush=True)
 stats=[]
 for k in range(8):
  assert len(nearest[k])==200
  count=sums[k,0];x,y,h=sums[k,1:]/count if count else [float(1),float(1),float(0)];stats.append(dict(cluster=k,body_patches=int(count),area=float(count/max(1,sums[:,0].sum())),centroid_x=float(x),centroid_y=float(y),mean_HU=float(h),candidate=bool(count and x<.5 and y<.6 and 40<=h<=180)))
 possible=[x for x in stats if x['candidate']];n2=max(possible,key=lambda x:x['area'])['cluster'] if possible else None
 # Exact same fixed prompts and joint-space embeddings as original sealed naming.
 texts=np.load(BASE/'text_embeddings.npy');net=frozen_backbone();full=net.full_model;keys=sorted(set((i,t,x,y) for nn in nearest for _,i,t,x,y in nn));emb=np.empty((len(keys),512),np.float32)
 for s in range(0,len(keys),16):
  inp=[]
  for i,t,x,y in keys[s:s+16]:inp.append(image_tensor(crop_image(load_image(rr[i]),[int(np.floor(y+.5))-56,int(np.floor(x+.5))-56,112]),224))
  with torch.no_grad():emb[s:s+len(inp)]=full.encode_image(torch.stack(inp),normalize=True).numpy()
 lookup={k:i for i,k in enumerate(keys)};avg=np.stack([emb[[lookup[(i,t,x,y)] for _,i,t,x,y in nn]].mean(0) for nn in nearest]);avg/=np.maximum(np.linalg.norm(avg,axis=1,keepdims=True),1e-12);sim=avg@texts.T;std=sim.std(0,keepdims=True);zz=np.divide(sim-sim.mean(0,keepdims=True),std,out=np.zeros_like(sim),where=std>0);n1=int(zz[:,0].argmax());selected=dict(N1=n1,N2=n2,N3=n1 if n1==n2 else None)
 save_meta('naming.json',dict(selected=selected,nearest=nearest,N2_stats=stats,N1_cosine=sim.tolist(),N1_z_scores=zz.tolist(),liver_margin=float(np.sort(zz[:,0])[-1]-np.sort(zz[:,0])[-2]),checkpoint_sha256=sha(checkpoint),centers_sha256=sha(out/'centers.npy'),fit_patients=fit['fit_patients'],labels_read=False,mechanical_grid_generalization=dict(grid=G,subpixel_footprints_without_native_pixel=fallback_count,empty_footprint_rule='Use HU of same rounded native centre used for body membership; fixed before labels; thresholds unchanged.')))
 np.save(out/'naming_crop_embeddings.npy',emb)
else:
 assert not(out/'prediction_manifest.json').exists();name=read(out/'naming.json');fit=read(out/'fit.json');assert name['checkpoint_sha256']==sha(checkpoint)==fit['checkpoint_sha256'];centers=np.load(out/'centers.npy');assert sha(out/'centers.npy')==name['centers_sha256'];dense=np.lib.format.open_memmap(out/'dense_features.npy',mode='w+',dtype=np.float32,shape=(len(val),G,G,70));grid=np.lib.format.open_memmap(out/'raw_clusters.npy',mode='w+',dtype=np.uint8,shape=(len(val),G,G));native=np.lib.format.open_memmap(out/'raw_clusters_native.npy',mode='w+',dtype=np.uint8,shape=(len(val),512,512));masks={rule:{kind:np.lib.format.open_memmap(out/f'{rule}_{kind}.npy',mode='w+',dtype=np.uint8,shape=(len(val),512,512)) for kind in ['hard','F1']} for rule in ['N1','N2','N3']};ct=np.lib.format.open_memmap(out/'ct_native.npy',mode='w+',dtype=np.float32,shape=(len(val),512,512));start=time.monotonic();replays=[]
 for j,i in enumerate(val):
  z=embedding(i);im=load_image(rr[i]);a,b,res=score_fields(z,im,rr[i],centers,name['selected'],fit['temperature']);dense[j]=z;grid[j]=a;native[j]=b;ct[j]=im
  for rule in masks:
   for kind,pr in zip(['hard','F1'],res[rule]):masks[rule][kind][j]=pr
  if j in [0,95,177,275,471]:
   z2=embedding(i);assert np.array_equal(z,z2);a2,b2,r2=score_fields(z2,im,rr[i],centers,name['selected'],fit['temperature']);assert np.array_equal(a,a2)
   for rule in masks:assert all(np.array_equal(x,y) for x,y in zip(res[rule],r2[rule]))
   replays.append(dict(index=i,exact=True))
  if j%100==0:print('PREDICT',key,j,len(val),time.monotonic()-start,flush=True)
 for x in [ct,dense,grid,native,*[x for r in masks.values() for x in r.values()]]:x.flush()
 outputs={p.name:sha(p) for p in out.glob('*.npy') if p.name!='fit_sample.npy'};save_meta('prediction_manifest.json',dict(seed=args.seed,grid=G,code_dim=70,indices=val,sample_ids=[rr[i]['sample_id'] for i in val],selected=name['selected'],checkpoint_sha256=sha(checkpoint),outputs=outputs,replay=replays,preprocessing=contract(),records=[rr[i] for i in val],K=8,checkpoint=str(checkpoint),labels_read=False,completed_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
ledger(f'logs/{key}_{args.stage}_access.json');baseline_seal();print('COMPLETE',args.stage,key,flush=True)
