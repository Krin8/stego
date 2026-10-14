from common import *
from datetime import datetime,timezone
from torch._subclasses.fake_tensor import FakeTensorMode
cache=Path.home()/'.cache/torch/hub/checkpoints'
requested={'DINO_v1_ViT_B8':['dino_vitbase8_pretrain.pth'],'DINO_v1_ViT_B16':['dino_vitbase16_pretrain.pth'],'DINO_v1_ViT_S8':['dino_deitsmall8_pretrain.pth','dino_deitsmall8_300ep_pretrain.pth'],'DINOv2_ViT_B14':['dinov2_vitb14_pretrain.pth'],'DINOv2_ViT_L14':['dinov2_vitl14_pretrain.pth']}
inventory={}
for model,names in requested.items():
 entries=[]
 for name in names:
  p=cache/name
  if p.is_file():
   with FakeTensorMode():sd=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
   shape=list(sd['patch_embed.proj.weight'].shape)
   entries.append(dict(path=str(p),bytes=p.stat().st_size,mtime=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),sha256=sha(p),patch_embed_shape=shape,keys=len(sd)))
 inventory[model]=dict(status='FOUND' if entries else 'MISSING',files=entries)
# Rule out the unrelated generic project checkpoint without loading its tensors.
p=ROOT/'models/model_epoch_0720_iter_085000.pth'
try:
 with FakeTensorMode():unknown=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
 inventory['other_project_checkpoint']=dict(path=str(p),top_level_keys=list(unknown)[:25],classification='Not a declared DINO backbone established')
except Exception as e:
 inventory['other_project_checkpoint']=dict(path=str(p),status='UNVERIFIED unreadable checkpoint',error=str(e))
dump('inventory.json',inventory)
protocol=dict(locked_at_utc=datetime.now(timezone.utc).isoformat(),arms={'D1':dict(backbone='DINO_v1_ViT_B8',input=224,grid=28,status='RUN'),'D2':dict(backbone='DINO_v1_ViT_B16',input=448,grid=28,status='SKIP_MISSING'),'D3':dict(backbone='DINOv2_ViT_B14',input=392,grid=28,status='SKIP_MISSING')},
 preprocessing=dict(window_hu=[-150,250],storage='existing native512 uint16 derivatives /65535',spatial='IMPROVE B2 label-free body square crop, threshold.05, LCC8,5% each-side margin,zero padding; no physical resampling',resize='bilinear antialias',grayscale_channels=3,mean=MEAN,std=STD,normalization_application_count=1,orientation='native',features='final block LayerNorm tokens, CLS removed, patch L2, uncentered'),
 segmentation=dict(patients=20,slices=2341,K=[8,2],seeds=[0,1,2],fit='all slices, all patches, no labels',kmeans=dict(init='k-means++',n_init=3,max_iter=100,tol=1e-4,algorithm='lloyd',copy_x=False),mapping='other19 labels only; K8 strict-majority and exhaustive optimal subset; K2 Hungarian maximizing confusion count',outside_crop='forced background not eligible for mapping',scoring=[224,512],mask_resize='nearest',aggregation='pool pixels per patient; average seeds per patient, then equal-patient mean/sample SD/min/max',postprocessing='none'),
 selectivity=dict(cohort='all472 validation slices; 4 patients',sources=['D1','BiomedCLIP_B2_448'],pool='same-image',sampling='SELECTIVITY/PAIRS batch order, seeds, actual loss.forward coordinates;7 passes,11x11 coordinates;fixed2048 of14641 pair positions',centering='actual project helper pointwise restored-global-mean, zero_clamp=True,stabalize=False',budgets=[.05,.1,.2],thresholds='global affinity quantiles; labels not read until scores/cuts saved',labels='common28x28 majority cells of body-cropped nearest448 mask, ties background; actual nearest sampler coordinates',reference='BiomedCLIP cached B2_44828-grid; original SELECTIVITY14-grid full-frame reference listed separately as historical'),
 figures=dict(indices=json.loads((OLD/'figure_selection.json').read_text())['eight_indices'],pca='fit label-free on these8 feature grids jointly; deterministic full SVD; each RGB component scales using joint1st/99th percentiles; no scoring/selection'),
 decision='justified only if an available declared arm reaches K8 224 LOPO liver IoU >=62.87% and same-image liver-anchor AUC >=.90; report mapping rule triggering gate explicitly; no head training')
dump('PROTOCOL.json',protocol)
print(json.dumps(inventory,indent=2))
