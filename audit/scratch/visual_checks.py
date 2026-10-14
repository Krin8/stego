"""<=3 sample forwards; embeds PCA/cosine/label QA into AUDIT.md; no image file."""
import guard,sys,signal,io,base64
signal.alarm(110)
sys.modules['wandb']=None
import torch,numpy as np
from torch import nn
import torch.nn.functional as F
from PIL import Image,ImageDraw
from torchvision import transforms as T
from omegaconf import OmegaConf
sys.path.insert(0,str(guard.ROOT/'src'))
from data import CHAOS
torch.set_num_threads(4)
tns=guard.definitions('src/utils.py',['normalize','ToTargetTensor','identity_transform','get_transform'],dict(torch=torch,np=np,T=T,Image=Image))
ns=guard.definitions('src/modules.py',['BiomedCLIPFeaturizer'],dict(torch=torch,nn=nn,F=F,__file__=str(guard.ROOT/'src/modules.py')))
cfg=OmegaConf.load(guard.ROOT/'src/configs/train_config.yaml');net=ns['BiomedCLIPFeaturizer'](512,cfg).eval()
tr=tns['get_transform'](224,False,'center');tt=tns['get_transform'](224,True,'center')
raw=CHAOS(cfg.pytorch_data_dir,'CT','train',tr,tt,2,False)
new=CHAOS(cfg.pytorch_data_dir,'CT','train',tr,tt,2,True)
samples=[(new, next(i for i,s in enumerate(new.samples) if s[0].endswith('CT_28_0110.png')))]
old=[i for i,s in enumerate(new.samples) if '/CT_p28_' in s[0]]
samples.append((new,old[len(old)//2]))
rawids=[i for i,s in enumerate(raw.samples) if s[3]=='28']
samples.append((raw,rawids[len(rawids)//2]))
canvas=Image.new('RGB',(224*4,254*len(samples)),(245,245,245));draw=ImageDraw.Draw(canvas)
out=[]
for row,(ds,idx) in enumerate(samples):
    x,y,mask=ds[idx]
    with torch.no_grad():f,c=net(x[None])
    pixels=f[0].flatten(1).T;center=pixels-pixels.mean(0)
    u,s,v=torch.linalg.svd(center,full_matrices=False)
    pca=u[:,:3]*s[:3];pca=(pca-pca.min(0).values)/(pca.max(0).values-pca.min(0).values).clamp_min(1e-8)
    pca=pca.reshape(14,14,3).numpy()
    n=F.normalize(f,dim=1);cos=(n*n[:,:,7:8,7:8]).sum(1)[0].numpy()
    heat=np.stack([(cos+1)/2,np.zeros_like(cos),(1-cos)/2],-1)
    rgb=(x.numpy()*np.array([.26862954,.26130258,.27577711])[:,None,None]+np.array([.48145466,.4578275,.40821073])[:,None,None]).transpose(1,2,0)
    overlay=rgb.copy(); fg=y.numpy()==1;overlay[fg]=.5*overlay[fg]+np.array([.5,0,0])
    for col,(a,title) in enumerate([(rgb,'Input'),(overlay,'Liver overlay'),(pca,'Block11 patch PCA'),(heat,'Cosine to patch (7,7)')]):
        im=Image.fromarray((np.clip(a,0,1)*255).astype('uint8')).resize((224,224),Image.Resampling.NEAREST)
        canvas.paste(im,(col*224,row*254+30));draw.text((col*224+3,row*254+15),title,fill='black')
    label=str(ds.samples[idx][0]);draw.text((3,row*254),('raw DICOM' if ds is raw else label.split('/')[-1]),fill='black')
    out.append(f'{label}: input={tuple(x.shape)}, label={tuple(y.shape)}, dense={tuple(f.shape)}, random_initialized_code={tuple(c.shape)}, fg_fraction={fg.mean():.5f}, PCA_top3_variance={float(s[:3].square().sum()/s.square().sum()):.5f}, cosine_min/median/max={np.min(cos):.5f}/{np.median(cos):.5f}/{np.max(cos):.5f}')
    if ds is raw:
        import pydicom
        d=pydicom.dcmread(ds.samples[idx][0],stop_before_pixels=True)
        out.append('raw header '+str({k:str(getattr(d,k,None)) for k in ['Rows','Columns','PixelSpacing','SliceThickness','ImageOrientationPatient','RescaleSlope','RescaleIntercept']}))
b=io.BytesIO();canvas.save(b,format='PNG');encoded=base64.b64encode(b.getvalue()).decode()
guard.record('Phase 6 sampled image/mask/PCA/cosine and raw DICOM shape evidence','\n'.join(out))
with guard.REPORT.open('a') as handle:handle.write('\nPCA colors are independently scaled per slice. The code head is newly initialized; these panels inspect pretrained features, not learned segmentation. Cosine scale is fixed [-1,1], red=+1/blue=-1. Labels are overlays only.\n\n![Three-slice image, label, patch PCA and cosine QA](data:image/png;base64,'+encoded+')\n')
print('AUDIT_IMAGE_BASE64='+encoded)
