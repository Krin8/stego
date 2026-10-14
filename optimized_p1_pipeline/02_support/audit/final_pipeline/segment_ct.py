"""Frozen CT F1 inference. CPU only; no training. See README.md for the PNG contract."""
from pathlib import Path
import os, sys, json, hashlib, tempfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.dont_write_bytecode=True
# Preserve the existing read-only HF cache when redirecting writable plotting caches.
os.environ.setdefault('HF_HOME',str(Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'huggingface'))
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',MPLCONFIGDIR=str(HERE/'mpl'),XDG_CACHE_HOME=str(HERE/'.cache'))
tempfile.tempdir=str(HERE)
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from PIL import Image
from torchvision.transforms import functional as TF, InterpolationMode
sys.path.insert(0,str(ROOT/'src'))
from chaos_preprocessing import MEAN, STD, read_image, file_sha256
from types import SimpleNamespace
torch.set_num_threads(4)

# Verbatim from audit/improve/pipeline.py
def body_box(im):
 lab,n=ndi.label(im>0.05,structure=np.ones((3,3),dtype=int))
 if n:
  counts=np.bincount(lab.ravel());counts[0]=0
  body=ndi.binary_fill_holes(lab==counts.argmax());yy,xx=np.where(body)
  side=int(np.ceil(max(yy.max()-yy.min()+1,xx.max()-xx.min()+1)*1.1))
  y0=int(np.floor((yy.min()+yy.max()+1-side)/2));x0=int(np.floor((xx.min()+xx.max()+1-side)/2))
  return [y0,x0,side],False
 return [0,0,max(im.shape)],True

# Verbatim from audit/improve/pipeline.py
def crop_image(im,box):
 y,x,s=box;h,w=im.shape;out=np.zeros((s,s),np.float32)
 ya,xa,yb,xb=max(0,y),max(0,x),min(h,y+s),min(w,x+s)
 out[ya-y:yb-y,xa-x:xb-x]=im[ya:yb,xa:xb]
 return out

# Verbatim from audit/improve/pipeline.py
def image_tensor(im,res,mean=MEAN,std=STD):
 t=torch.from_numpy(np.array(im,dtype=np.float32)).unsqueeze(0).repeat(3,1,1)
 t=TF.resize(t,[res,res],interpolation=InterpolationMode.BILINEAR,antialias=True)
 return TF.normalize(t,mean,std)

# Verbatim from audit/improve/pipeline.py
def inverse_grid(grid,record,mode='nearest',outside=8):
 y,x,s=record['box'];h,w=record['shape']
 t=torch.from_numpy(np.asarray(grid,dtype=np.float32).copy())[None,None]
 kwargs=dict(align_corners=False) if mode=='bilinear' else {}
 crop=F.interpolate(t,size=(s,s),mode=mode,**kwargs)[0,0].numpy()
 result=np.full((h,w),outside,dtype=np.float32)
 ya,xa,yb,xb=max(0,y),max(0,x),min(h,y+s),min(w,x+s)
 result[ya:yb,xa:xb]=crop[ya-y:yb-y,xa-x:xb-x]
 return result

# Verbatim from audit/improve/pipeline.py
def resize224(arr):
 return F.interpolate(torch.tensor(np.array(arr),dtype=torch.float32)[None,None],(224,224),mode='nearest')[0,0].numpy()

# Verbatim from audit/improve/postprocess.py
def largest(mask,structure):
 lab,n=ndi.label(mask,structure=structure)
 if not n:return mask
 counts=np.bincount(lab.ravel());counts[0]=0
 return lab==counts.argmax()

# Verbatim from audit/improve/postprocess.py
def guided(image,field):
 size=17;mi=ndi.uniform_filter(image,size=size,mode='reflect');mp=ndi.uniform_filter(field,size=size,mode='reflect')
 var=ndi.uniform_filter(image*image,size=size,mode='reflect')-mi*mi
 cov=ndi.uniform_filter(image*field,size=size,mode='reflect')-mi*mp
 aa=cov/(var+.01);bb=mp-aa*mi
 return np.clip(ndi.uniform_filter(aa,size=size,mode='reflect')*image+ndi.uniform_filter(bb,size=size,mode='reflect'),0,1)

def contract():
    return dict(modality='CT',window_hu=[-150,250],intensity='clip HU then linear [0,1]; round(65535*x)/65535',
      png='uint16 grayscale already windowed [-150,250]; reject 8-bit/RGB',orientation='native; no flip/transpose',
      spatial_resampling='none',native_shape=[512,512],body_threshold=.05,body_connectivity=8,
      body_box='largest component; fill holes; square bounding box with 5% each-side margin; zero padding',
      input_resolution=448,resize='torchvision bilinear antialias=True',mean=list(MEAN),std=list(STD),normalization_application_count=1,
      grayscale_channels=3,backbone='BiomedCLIP ViT-B/16',layer='final block',normalization='final LayerNorm then patch L2; uncentered',
      cls_excluded=True,grid=[28,28],eval=True,device='cpu',soft_assignment='softmax(-squared_euclidean_distance/temperature)',
      temperature='median nearest/second-nearest squared-distance gap over fitting patches / log(4); floor 1e-6',
      inverse_crop='bilinear align_corners=False; outside=0',guided_radius=8,guided_epsilon=.01,guided_boundary='reflect',
      threshold=.5,largest_component_connectivity=8,hole_fill_background_connectivity=4,closing_disk_radius=3,
      closing_padding='3 zero pixels then crop',f2_default=False,f2='median(3,1,1), nearest endpoints, one series at a time')

def load_plane(path):
    path=Path(path)
    if path.suffix.lower()=='.png':
        with Image.open(path) as im:
            a=np.asarray(im)
            if a.ndim!=2 or a.dtype not in (np.dtype('uint16'),np.dtype('int32')) or im.mode not in ('I;16','I;16L','I;16B','I'):
                raise ValueError(f'{path}: PNG must be uint16 grayscale under the documented HU-window contract; 8-bit/RGB is ambiguous')
            if a.min()<0 or a.max()>65535:raise ValueError('PNG range must be 0..65535')
            a=a.astype(np.float32)/65535.
    else:
        import pydicom
        ds=pydicom.dcmread(path,stop_before_pixels=True)
        if ds.Modality!='CT':raise ValueError(f'{path}: CT only')
        a=read_image(path,'CT').astype(np.float32)/65535.
    if a.shape!=(512,512):raise ValueError(f'{path}: fixed native contract requires 512x512, got {a.shape}')
    return a

def load_extractor(meta):
    weights=ROOT/'src/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin'
    if file_sha256(weights)!=meta['backbone_weight_sha256']:raise ValueError('Backbone weight digest mismatch')
    from modules import BiomedCLIPFeaturizer
    torch.manual_seed(0)
    net=BiomedCLIPFeaturizer(70,SimpleNamespace(projection_type=None,dropout=False,use_text_prompts=False)).cpu().eval().requires_grad_(False)
    return net

def extract(net,im):
    box,_=body_box(im)
    x=image_tensor(crop_image(im,box),448)[None]
    with torch.inference_mode():f=F.normalize(net(x)[0],dim=1)
    if tuple(f.shape)!=(1,768,28,28):raise ValueError(str(f.shape))
    return f[0].permute(1,2,0).numpy().reshape(784,768),dict(box=box,shape=list(im.shape))

def distances(features,centers):
    x=features.reshape(-1,768)
    return np.maximum(0,np.square(x).sum(1)[:,None]+np.square(centers).sum(1)[None,:]-2*x@centers.T)

def predict_features(features,im,record,centers,meta):
    d=distances(features,centers);ids=np.asarray(meta['liver_clusters'],dtype=int)
    look=np.zeros(9,bool);look[ids]=True
    hard=look[inverse_grid(d.argmin(1).reshape(28,28),record,outside=8).astype(np.uint8)]
    z=-(d-d.min(1,keepdims=True))/meta['temperature'];w=np.exp(z);w/=w.sum(1,keepdims=True)
    field=w[:,ids].sum(1).reshape(28,28) if len(ids) else np.zeros((28,28),np.float32)
    soft=inverse_grid(field,record,mode='bilinear',outside=0)
    pred=largest(guided(im,soft)>=.5,np.ones((3,3),int))
    pred=ndi.binary_fill_holes(pred)
    yy,xx=np.mgrid[-3:4,-3:4]
    pred=ndi.binary_closing(np.pad(pred,3),structure=xx*xx+yy*yy<=9)[3:-3,3:-3]
    return hard,pred

def load_artifact(seed):
    meta=json.loads((HERE/f'all20_seed{seed}.json').read_text())
    path=HERE/f'all20_seed{seed}_centers.npy'
    if file_sha256(path)!=meta['centers_sha256']:raise ValueError('Cluster centres digest mismatch')
    if meta['preprocessing']!=contract():raise ValueError('Preprocessing contract mismatch')
    return np.load(path,allow_pickle=False),meta

def save_output(out,name,im,pred):
    mask_path=out/f'{name}_mask.png';overlay_path=out/f'{name}_overlay.png'
    rgb=np.repeat(im[...,None],3,axis=2);rgb[pred]=.65*rgb[pred]+.35*np.array([1.,0.,0.])
    edge=pred & ~ndi.binary_erosion(pred);rgb[edge]=[1,0,0]
    for path,array in [(mask_path,pred.astype(np.uint8)*255),(overlay_path,np.rint(np.clip(rgb,0,1)*255).astype(np.uint8))]:
        with path.open('xb') as f:Image.fromarray(array).save(f,format='PNG')

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--seed',type=int,choices=[0,1,2],default=0)
    p.add_argument('--check',action='store_true');p.add_argument('--f2',action='store_true')
    p.add_argument('--order-manifest',type=Path,help='For F2 PNG only: JSON list of {file,series_id,physical_position}, one series')
    a=p.parse_args();centers,meta=load_artifact(a.seed)
    if a.check:
        if a.f2:raise ValueError('--check verifies fixed F1, not optional F2')
        check=json.loads((HERE/'check_reference.json').read_text());net=load_extractor(meta);results=[]
        for r in check['samples']:
            path=ROOT/r['image'];im=load_plane(path);f,record=extract(net,im)
            _,pred=predict_features(f,im,record,centers,meta)
            with Image.open(ROOT/r['mask']) as image:y=np.array(image)>0
            union=(pred|y).sum();iou=float((pred&y).sum()/max(1,union));expected=r['iou'][str(a.seed)]
            error=abs(iou-expected);assert error<=check['iou_absolute_tolerance'],(r['stem'],iou,expected,error)
            results.append(dict(stem=r['stem'],iou=iou,expected_iou=expected,absolute_error=error))
        result=dict(passed=True,seed=a.seed,tolerance=check['iou_absolute_tolerance'],samples=results)
        print(json.dumps(result,indent=2));return
    if a.input is None or a.output is None:p.error('--input and --output are required unless --check')
    files=sorted([p for p in a.input.iterdir() if p.is_file() and (p.suffix.lower() in ('.png','.dcm','.dicom') or not p.suffix)])
    if not files:raise ValueError('No PNG/DICOM slices found')
    if len({p.stem for p in files})!=len(files):raise ValueError('Duplicate output stems')
    if a.f2:
        if all(p.suffix.lower()=='.png' for p in files):
            if a.order_manifest is None:raise ValueError('F2 PNG requires --order-manifest; filename order is not physical position')
            order=json.loads(a.order_manifest.read_text())
            if {r['file'] for r in order}!={p.name for p in files} or len(order)!=len(files):raise ValueError('Order manifest does not match input files')
            if len({r['series_id'] for r in order})!=1 or len({r['physical_position'] for r in order})!=len(files):raise ValueError('F2 requires one series, unique physical positions')
            files=[a.input/r['file'] for r in sorted(order,key=lambda r:r['physical_position'])]
        elif all(p.suffix.lower()!='.png' for p in files):
            import pydicom
            ds=[pydicom.dcmread(p,stop_before_pixels=True) for p in files]
            if len({str(d.SeriesInstanceUID) for d in ds})!=1:raise ValueError('F2 requires one DICOM series')
            orient=np.asarray(ds[0].ImageOrientationPatient,float)
            if any(not np.allclose(d.ImageOrientationPatient,orient,atol=1e-5,rtol=0) for d in ds):raise ValueError('Inconsistent orientation')
            normal=np.cross(orient[:3],orient[3:]);pos=[float(np.asarray(d.ImagePositionPatient,float)@normal) for d in ds]
            if len(set(pos))!=len(pos):raise ValueError('Duplicate slice positions')
            files=[files[i] for i in np.argsort(pos)]
        else:raise ValueError('F2 does not accept mixed PNG/DICOM inputs')
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Output folder must be new or empty')
    a.output.mkdir(parents=True,exist_ok=True);net=load_extractor(meta)
    pending=[]
    for path in files:
        im=load_plane(path);f,record=extract(net,im);_,pred=predict_features(f,im,record,centers,meta)
        if a.f2:pending.append((path,im,pred))
        else:save_output(a.output,path.stem,im,pred)
    if a.f2:
        volume=ndi.median_filter(np.stack([r[2] for r in pending]),size=(3,1,1),mode='nearest')
        for (path,im,_),pred in zip(pending,volume):save_output(a.output,path.stem,im,pred)
    (a.output/'run.json').write_text(json.dumps(dict(seed=a.seed,f2=a.f2,metadata=meta,inputs=[str(p) for p in files]),indent=2)+'\n')
if __name__=='__main__':main()
