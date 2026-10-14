"""Optional explicit cache creation; never run automatically by evaluation."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from src2.utils import parser,load_config,setup,new_output,write_json,resolve_path,sha256,labels_closed
from src2.data import records
from src2.preprocessing import load_image,body_box,crop_image,image_tensor
from src2.backbone import load_backbone,tokens


def main():
    p=parser('pipeline.yaml',__doc__);p.add_argument('--output',required=True)
    a=p.parse_args();cfg=load_config(a.config);setup(cfg)
    with labels_closed():
        rr=records(cfg);net=load_backbone(cfg);out=new_output(a.output)
        cache=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float32,shape=(len(rr),784,768))
        for start in range(0,len(rr),8):
            images=[]
            for r in rr[start:start+8]:
                image=load_image(resolve_path(cfg,r['image']),cfg)
                if body_box(image)!=r['box']:raise ValueError('Recorded feature crop differs from preprocessing')
                images.append(image_tensor(crop_image(image,r['box']),cfg))
            f=tokens(net,torch.stack(images)).permute(0,2,3,1).reshape(len(images),784,768).numpy()
            cache[start:start+len(images)]=f
        cache.flush()
        write_json(out/'metadata.json',dict(records=rr,preprocessing=cfg['preprocessing'],labels_read=False,features_sha256=sha256(out/'features.npy')))
        print(out/'features.npy')


if __name__=='__main__':main()
