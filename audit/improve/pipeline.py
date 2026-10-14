"""New label-free spatial variants; original model and scoring functions imported."""
from common import *
from scipy import ndimage as ndi
from PIL import Image
from chaos_preprocessing import MEAN, STD
from torchvision.transforms import functional as TF, InterpolationMode
FINAL=ROOT/'audit/scratch/final_baseline'
def metadata():return json.loads((OUT/'metadata.json').read_text())
def body_box(im):
 lab,n=ndi.label(im>0.05,structure=np.ones((3,3),dtype=int))
 if n:
  counts=np.bincount(lab.ravel());counts[0]=0
  body=ndi.binary_fill_holes(lab==counts.argmax());yy,xx=np.where(body)
  side=int(np.ceil(max(yy.max()-yy.min()+1,xx.max()-xx.min()+1)*1.1))
  y0=int(np.floor((yy.min()+yy.max()+1-side)/2));x0=int(np.floor((xx.min()+xx.max()+1-side)/2))
  return [y0,x0,side],False
 return [0,0,max(im.shape)],True
def crop_image(im,box):
 y,x,s=box;h,w=im.shape;out=np.zeros((s,s),np.float32)
 ya,xa,yb,xb=max(0,y),max(0,x),min(h,y+s),min(w,x+s)
 out[ya-y:yb-y,xa-x:xb-x]=im[ya:yb,xa:xb]
 return out
def image_tensor(im,res,mean=MEAN,std=STD):
 t=torch.from_numpy(np.array(im,dtype=np.float32)).unsqueeze(0).repeat(3,1,1)
 t=TF.resize(t,[res,res],interpolation=InterpolationMode.BILINEAR,antialias=True)
 return TF.normalize(t,mean,std)
def raw_image(record):
 with Image.open(record['image']) as im:return np.array(im,dtype=np.float32)/65535.
def inverse_grid(grid,record,mode='nearest',outside=8):
 y,x,s=record['box'];h,w=record['shape']
 t=torch.from_numpy(np.asarray(grid,dtype=np.float32).copy())[None,None]
 kwargs=dict(align_corners=False) if mode=='bilinear' else {}
 crop=F.interpolate(t,size=(s,s),mode=mode,**kwargs)[0,0].numpy()
 result=np.full((h,w),outside,dtype=np.float32)
 ya,xa,yb,xb=max(0,y),max(0,x),min(h,y+s),min(w,x+s)
 result[ya:yb,xa:xb]=crop[ya-y:yb-y,xa-x:xb-x]
 return result
def native_prediction(grid,record,variant,mode='nearest',outside=8):
 if variant=='B0':
  t=torch.tensor(np.array(grid),dtype=torch.float32)[None,None]
  kw=dict(align_corners=False) if mode=='bilinear' else {}
  return F.interpolate(t,record['shape'],mode=mode,**kw)[0,0].numpy()
 return inverse_grid(grid,record,mode,outside)
def resize224(arr):
 return F.interpolate(torch.tensor(np.array(arr),dtype=torch.float32)[None,None],(224,224),mode='nearest')[0,0].numpy()
def patient_stats(values):
 v=np.asarray(values,dtype=float)
 return dict(mean=float(v.mean()),std=float(v.std(ddof=1)),min=float(v.min()),max=float(v.max()),median=float(np.median(v)))
def mapped_report(conf):
 # Original exhaustive eight-cluster subset code; ninth row is forced-background outside-crop pixels.
 r=mapping_report(conf)
 r['majority']=scores(conf,np.flatnonzero(conf[:8,1]>conf[:8,0]))
 return r
def results_from_conf(confs):
 total=sum(confs.values());folds={}
 for p,c in confs.items():
  fitted=mapped_report(total-c)
  folds[p]={m:scores(c,fitted[m]['clusters']) for m in ['optimal','majority']}
  folds[p]['mapping_patients']=[q for q in confs if q!=p]
 return dict(patient_confusions={p:c.tolist() for p,c in confs.items()},folds=folds,
  summary={m:{metric:patient_stats([f[m][metric] for f in folds.values()]) for metric in ['iou','dice']} for m in ['optimal','majority']})
def features(variant):
 if variant=='B0':
  train=np.load(ROOT/'audit/scratch/baseline/train_features.npy',mmap_mode='r')
  val=np.load(FORE/'val_features.npy',mmap_mode='r')
  yield train
  yield val.transpose(0,2,3,1).reshape(472,196,768)
 else:yield np.load(OUT/f'{variant}_features.npy',mmap_mode='r')
