"""One real CT forward/replay; no fitting, training, annotations or data writes."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from src2.utils import HERE,load_config,setup,labels_closed
from src2.data import FeatureStore
from src2.preprocessing import load_image,body_box,crop_image,image_tensor
from src2.backbone import load_backbone,tokens
from src2.modules import load_head


def main():
    cfg=load_config(HERE/'configs/pipeline.yaml');setup(cfg);started=time.monotonic()
    with labels_closed():
        store=FeatureStore(cfg);r=store.records[0]
        png=load_image(r['image'],cfg);dicom=load_image(r['dicom'],cfg)
        np.testing.assert_array_equal(png,dicom)
        box=body_box(dicom);assert box==r['box']
        x=image_tensor(crop_image(dicom,box),cfg)[None]
        net=load_backbone(cfg);a=tokens(net,x);b=tokens(net,x)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        cached=store.batch([0]);difference=float((a-cached).abs().max())
        torch.testing.assert_close(a,cached,rtol=1e-4,atol=2e-5)
        model,checkpoint_sha=load_head(cfg)
        with torch.inference_mode():code=model.embed(a)
        assert code.shape==(1,70,28,28)
        result=dict(sample_id=r['sample_id'],input_shape=list(x.shape),feature_shape=list(a.shape),
            code_shape=list(code.shape),raw_dicom_equals_windowed_png=True,cache_max_abs_difference=difference,
            backbone_training=net.training,trainable_backbone_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
            deterministic=True,strict_backbone_load=True,checkpoint_sha256=checkpoint_sha,
            labels_opened=False,training_batches=0,kmeans_fits=0,seconds=time.monotonic()-started)
    path=HERE/'verification/backbone_smoke.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
