"""Names fixed before labels: explicit access guard rejects mask/score/mapping reads."""
import os
os.environ['NAMING_STAGE']='A'
from common import *
import pydicom,inspect
from datetime import datetime,timezone
assert not (OUT/'stageA_manifest.json').exists(),'Sealed Stage A exists; do not overwrite'
objects=['liver','spleen','kidney','stomach','pancreas','gallbladder','aorta','vertebra','muscle','subcutaneous fat','bowel','lung','heart','rib','air']
params=dict(seed=20260930,K=8,kmeans_seeds=[0,1,2],fits=['inductive','transductive'],no_refit=True,
 preprocessing=contract(),feature_source='existing B2_448_features.npy, normalized uncentered final tokens',
 N1=dict(nearest_per_cluster=200,restriction='all body-centre patches in this fit own images; no cluster-assignment restriction',distance='squared Euclidean to saved centre',ties='distance, then global image index, then row-major patch index',crop_native_pixels=112,crop_center='grid cell centre transformed through recorded body crop: (cell+.5)*side/28-.5+offset; nearest integer centre=floor(value+.5)',crop_boundary='zero padding in windowed[0,1] space',resize=224,interpolation='bilinear antialias=True',mean=list(MEAN),std=list(STD),normalization_application_count=1,image_embedding='full BiomedCLIP encode_image with CLIP projection, L2 per crop, mean of200, L2 mean for cosine',text_embedding='full BiomedCLIP encode_text normalize=True; tokenizer context256',prompts=['an axial abdominal CT image showing the '+x for x in objects],zscore='per prompt across8 cluster cosine similarities; population SD(ddof0); zero SD gives0',selection='argmax liver zscore; ties lowest cluster ID',top_prompt='argmax zscored prompt per cluster; ties declared prompt order',margins='top prompt zscore minus second; raw cosine top-minus-second also recorded; liver selection zscore winner minus runner-up'),
 N2=dict(body_HU_threshold=-500,threshold_comparison='>',opening='binary opening disk radius5; native pixels; scipy default1 iteration/border0',connectivity=8,hole_fill='4-connected background',body_patch='patch centre rounded to nearest native pixel inside body mask',bounding_box_coordinates='pixel-centre normalized x=(x-xmin)/(xmax-xmin), y=(y-ymin)/(ymax-ymin); singleton denominator1',patch_HU='mean unwindowed HU over native pixels whose centres lie in that patch cell footprint, restricted to body mask',area='each eligible patch counts one; fraction among all fit body patches',centroid='area-weighted patch centroid; each patch weight1',mean_HU='arithmetic mean of per-body-patch HU means, each patch weight1',candidates='x<.5 AND y<.6 AND40<=mean_HU<=180',selection='candidate with largest area; tie lowest cluster ID; none -> abstain'),
 N3='accept iff N1 and N2 cluster IDs equal, else abstain',
 scoring_plan=dict(inductive='4 validation patients; training16 majority/optimal fixed mapping reference',transductive='all20; other19 LOPO majority/optimal reference',reference_choice_agreement='exact singleton set equality; per-patient when LOPO mappings differ',abstentions='empty masks; report fit-level and evaluated-slice rates',outputs=['hard','F1'],resolutions=[224,512],patient_metrics=['pooled_pixels','mean_slice'],diagnostic='per-cluster liver precision/recall on evaluated cohort; label-using, not fed back',figures='same8 fixed slices; one composite per rule, columns6 fits, hard prediction contours; historical label-selected fixed slices'),
 snapshot='StageA outputs and code SHA256 sealed before StageB; manifest own digest separate sidecar')
dump('PARAMETERS.json',params)
source=json.loads((OLD/'metadata.json').read_text());rr=[{k:v for k,v in r.items() if k!='mask'} for r in source['records']]
manifest=json.loads((ROOT/'src/pytorch_data_dir/preprocessed_v2/manifest.json').read_text());byid={r['sample_id']:r for r in manifest['samples']}
rawroot=ROOT/'src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets'
for r in rr:r['dicom']=str(rawroot/byid[r['sample_id']]['dicom'])
dump('image_records.json',rr)
# Metadata contains image/mask filenames, but mask paths/values are never opened or used.
features=np.load(OLD/'B2_448_features.npy',mmap_mode='r');assert features.shape==(2341,784,768)
geom=np.zeros((2341,784,5),np.float32) # body indicator, normalized x,y, HU, count body pixels
centres_xy=np.zeros((2341,784,2),np.float32)
yy,xx=np.mgrid[-5:6,-5:6];disk=xx*xx+yy*yy<=25
for i,r in enumerate(rr):
 ds=pydicom.dcmread(r['dicom']);hu=ds.pixel_array.astype(np.float32)*float(getattr(ds,'RescaleSlope',1))+float(getattr(ds,'RescaleIntercept',0));assert hu.shape==(512,512)
 body=ndi.binary_fill_holes(largest(ndi.binary_opening(hu>-500,structure=disk),np.ones((3,3),int)))
 ys,xs=np.where(body)
 if not len(ys):raise ValueError('No N2 body mask: '+r['stem'])
 y0,x0,side=r['box'];gy,gx=np.mgrid[:28,:28];cy=y0+(gy.ravel()+.5)*side/28-.5;cx=x0+(gx.ravel()+.5)*side/28-.5
 centres_xy[i]=np.stack([cx,cy],1);ix=np.floor(cx+.5).astype(int);iy=np.floor(cy+.5).astype(int)
 inside=(ix>=0)&(iy>=0)&(ix<512)&(iy<512);eligible=inside.copy();eligible[inside]=body[iy[inside],ix[inside]]
 for t in np.flatnonzero(eligible):
  row,col=divmod(int(t),28);ya=max(0,int(np.ceil(y0+row*side/28-.5)));yb=min(512,int(np.ceil(y0+(row+1)*side/28-.5)));xa=max(0,int(np.ceil(x0+col*side/28-.5)));xb=min(512,int(np.ceil(x0+(col+1)*side/28-.5)))
  pixels=hu[ya:yb,xa:xb][body[ya:yb,xa:xb]];assert len(pixels)
  geom[i,t]=[1,(cx[t]-xs.min())/max(1,xs.max()-xs.min()),(cy[t]-ys.min())/max(1,ys.max()-ys.min()),pixels.mean(),len(pixels)]
 if i%400==0:print('GEOMETRY',i,elapsed(),flush=True)
np.save(OUT/'body_patch_geometry.npy',geom);np.save(OUT/'patch_centres_xy.npy',centres_xy)
fit_records={};unique_patches=set()
for fit in ['inductive','transductive']:
 inds=[i for i,r in enumerate(rr) if fit=='transductive' or r['split']=='train']
 assert len(inds)==(2341 if fit=='transductive' else 1869)
 for seed in [0,1,2]:
  key=f'{fit}_s{seed}';cp=FINAL/f'{"all20" if fit=="transductive" else "inductive"}_seed{seed}_centers.npy';assert cp.is_file()
  centers=np.load(cp,allow_pickle=False);np.save(OUT/f'{key}_centers.npy',centers)
  fp=OLD/f'B2_448_s{seed}_fit.json' if fit=='transductive' else FINAL/f'inductive_seed{seed}_fit.json'
  fitmeta=json.loads(fp.read_text());temperature=fitmeta['temperature']
  sums=np.zeros((8,4),np.float64);nearest=[[] for _ in range(8)];grids=np.empty((2341,28,28),np.uint8)
  for i in range(2341):
   d=distances(features[i],centers);assignment=d.argmin(1);grids[i]=assignment.reshape(28,28)
   if i not in inds:continue
   for c in range(8):
    use=np.flatnonzero((assignment==c)&(geom[i,:,0]>0))
    sums[c,0]+=len(use);sums[c,1:]+=geom[i,use,1:4].astype(np.float64).sum(0)
    body_tokens=np.flatnonzero(geom[i,:,0]>0)
    nearest[c]=sorted(nearest[c]+[(float(d[t,c]),i,int(t)) for t in body_tokens])[:200]
  np.save(OUT/f'{key}_grids.npy',grids)
  stats=[];selected=[]
  for c in range(8):
   assert len(nearest[c])>=200,(key,c,len(nearest[c]))
   nn=sorted(nearest[c])[:200];selected.append([dict(distance=d,image_index=i,patch_index=t) for d,i,t in nn]);unique_patches.update((i,t) for d,i,t in nn)
   count=sums[c,0];x,y,h=sums[c,1:]/count
   stats.append(dict(cluster=c,body_patches=int(count),area=float(count/sums[:,0].sum()),centroid_x=float(x),centroid_y=float(y),mean_HU=float(h),candidate=bool(x<.5 and y<.6 and 40<=h<=180)))
  candidates=[r for r in stats if r['candidate']];n2=max(candidates,key=lambda r:r['area'])['cluster'] if candidates else None
  fit_records[key]=dict(fit=fit,seed=seed,fit_images=len(inds),fit_patient_ids=sorted({rr[i]['patient'] for i in inds},key=int),centers_source=str(cp),centers_source_sha256=sha(cp),fit_metadata_source=str(fp),temperature=temperature,N2=dict(selected_cluster=n2,clusters=stats),nearest=selected)
  print('NEAREST',key,'N2',n2,elapsed(),flush=True)
# N1 full joint-space image and text towers: same local BiomedCLIP weights, no head.
from modules import BiomedCLIPFeaturizer
from open_clip.tokenizer import HFTokenizer
net=BiomedCLIPFeaturizer(70,SimpleNamespace(projection_type=None,dropout=False,use_text_prompts=False)).cpu().eval().requires_grad_(False);model=net.full_model
config=json.loads((ROOT/'src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_config.json').read_text())
tokenizer=HFTokenizer(config['model_cfg']['text_cfg']['hf_tokenizer_name'],context_length=256)
with torch.inference_mode():text_embeddings=model.encode_text(tokenizer(params['N1']['prompts']),normalize=True).cpu().numpy()
np.save(OUT/'text_embeddings.npy',text_embeddings)
keys=sorted(unique_patches);embedding=np.empty((len(keys),512),np.float32);last_images={}
for start in range(0,len(keys),16):
 batch=keys[start:start+16];inputs=[]
 for i,t in batch:
  im=raw_image(rr[i]);cx,cy=centres_xy[i,t];box=[int(np.floor(cy+.5))-56,int(np.floor(cx+.5))-56,112]
  inputs.append(image_tensor(crop_image(im,box),224))
 with torch.inference_mode():emb=model.encode_image(torch.stack(inputs),normalize=True).cpu().numpy()
 embedding[start:start+len(batch)]=emb
 if start%256==0:print('CROP EMBEDDINGS',start,len(keys),elapsed(),flush=True)
np.save(OUT/'crop_embeddings.npy',embedding);dump('crop_keys.json',keys);lookup={tuple(k):j for j,k in enumerate(keys)}
for key,r in fit_records.items():
 averages=np.stack([embedding[[lookup[(v['image_index'],v['patch_index'])] for v in nn]].mean(0) for nn in r['nearest']]);averages/=np.maximum(np.linalg.norm(averages,axis=1,keepdims=True),1e-12)
 sim=averages@text_embeddings.T;mean=sim.mean(0,keepdims=True);std=sim.std(0,keepdims=True);z=np.divide(sim-mean,std,out=np.zeros_like(sim),where=std>0)
 selected=int(z[:,0].argmax());r['N1']=dict(selected_cluster=selected,cosine=sim.tolist(),z_scores=z.tolist(),cluster_prompt_ranking=[])
 for c in range(8):
  order=np.argsort(-z[c],kind='stable');raw=np.argsort(-sim[c],kind='stable')
  r['N1']['cluster_prompt_ranking'].append(dict(cluster=c,top_prompt=objects[order[0]],margin=float(z[c,order[0]]-z[c,order[1]]),prompt_order=[objects[j] for j in order],raw_top_prompt=objects[raw[0]],raw_cosine_margin=float(sim[c,raw[0]]-sim[c,raw[1]])))
 liver=np.sort(z[:,0]);r['N1']['liver_selection_margin']=float(liver[-1]-liver[-2]);r['N3']=dict(selected_cluster=selected if selected==r['N2']['selected_cluster'] else None)
 dump(f'{key}_naming.json',r);print('NAMED',key,[r[n]['selected_cluster'] for n in ['N1','N2','N3']],flush=True)
weights=ROOT/'src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin'
dump('stageA_ledger.json',dict(stage='A',mask_or_label_files_read=0,prior_label_mapping_or_score_files_read=0,blocked_attempts=BLOCKED,opened_read_paths=sorted(READS),metadata_note='Read image inventory and derivative manifest for DICOM associations; they include unused mask filename/hash fields, but no masks/label values were read',backbone_weight_sha256=sha(weights),no_refits=True,unique_crop_embeddings=len(keys),seconds=elapsed()))
# Hash every StageA output and all frozen scripts; manifest/self-digest/log are explicitly excluded.
outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix in ['.py','.json','.npy'] and '.cache' not in p.parts and 'mpl' not in p.parts and 'torchinductor' not in str(p)}
manifest=dict(schema_version=1,sealed_utc=datetime.now(timezone.utc).isoformat(),parameters=params,selections={k:{n:r[n]['selected_cluster'] for n in ['N1','N2','N3']} for k,r in fit_records.items()},output_sha256=outputs,hash_exclusions='manifest itself and digest sidecar (self-reference); runtime logs and incidental library caches are not scientific outputs',labels_read=False)
dump('stageA_manifest.json',manifest);digest=sha(OUT/'stageA_manifest.json');(OUT/'stageA_manifest.sha256').write_text(digest+'  stageA_manifest.json\n')
print('SEALED',digest,elapsed(),flush=True)
