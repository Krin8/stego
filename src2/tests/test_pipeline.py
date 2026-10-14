"""Numerical parity checks; no training run, k-means fit or annotation reads."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
import pydicom
from torchvision.transforms import functional as TF, InterpolationMode
from src2.utils import ROOT, HERE, load_config, setup, labels_closed, write_json, sha256, contract
from src2.modules import StegoHead, load_head, objective
from src2.data import FeatureStore
from src2.preprocessing import body_box,crop_image,image_tensor,inverse_grid,load_image,read_hu
from src2.postprocessing import refine,largest
from src2.clustering import distances,embed,load_fit
from src2.naming import accumulate,select
from src2.loss import ColocatedCorrelationLoss,adjacent_partners
from src2.metrics import counts,scores,aggregate
from unittest.mock import patch


def definitions(path,names,namespace):
    tree=ast.parse(Path(path).read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=load_config(HERE/'configs/pipeline.yaml');setup(cls.cfg)
        cls.store=FeatureStore(cls.cfg);cls.record=cls.store.records[0]

    def test_configs_and_order(self):
        for p in (HERE/'configs').glob('*config.yaml'):
            self.assertEqual(load_config(p)['seed'],0)
        self.assertEqual(len(self.store.indices('train')),1869)
        self.assertEqual(len(self.store.indices('val')),472)

    def test_optimized_head_parity(self):
        model,_=load_head(self.cfg)
        ns=definitions(ROOT/'audit/ensemble/verify4/run_p1_optimized.py',{'StegoHead'},dict(nn=nn))
        old=ns['StegoHead']().eval().requires_grad_(False)
        result=old.load_state_dict(model.state_dict(),strict=False)
        self.assertEqual(result.unexpected_keys,['cluster1.0.weight','cluster1.0.bias'])
        x=self.store.batch([0,1869])
        with torch.no_grad():
            torch.testing.assert_close(model.embed(x),old.embed(x),rtol=0,atol=0)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_preprocessing_and_inverse_parity(self):
        ns=definitions(ROOT/'audit/improve/pipeline.py',{'body_box','crop_image','image_tensor','inverse_grid'},dict(np=np,ndi=ndi,torch=torch,F=F,TF=TF,InterpolationMode=InterpolationMode,MEAN=self.cfg['preprocessing']['mean'],STD=self.cfg['preprocessing']['std']))
        im=load_image(self.record['image'],self.cfg)
        self.assertEqual(body_box(im),ns['body_box'](im)[0])
        np.testing.assert_array_equal(crop_image(im,self.record['box']),ns['crop_image'](im,self.record['box']))
        crop=crop_image(im,self.record['box'])
        torch.testing.assert_close(image_tensor(crop,self.cfg),ns['image_tensor'](crop,448),rtol=0,atol=0)
        grid=np.arange(784,dtype=np.float32).reshape(28,28)
        for mode in ['nearest','bilinear']:
            np.testing.assert_array_equal(inverse_grid(grid,self.record,mode,0),ns['inverse_grid'](grid,self.record,mode,0))

    def test_guided_refinement_parity(self):
        ns=definitions(ROOT/'audit/ensemble/verify4/run_p1_optimized.py',{'apply_postprocessing'},dict(np=np,ndi=ndi,inverse_grid=inverse_grid,largest=largest))
        image=load_image(self.record['image'],self.cfg)
        field=np.random.default_rng(44).random((28,28),dtype=np.float32)
        np.testing.assert_array_equal(refine(field,image,self.record,self.cfg),ns['apply_postprocessing'](field,image,self.record))

    def test_n2_statistics_parity(self):
        model,_=load_head(self.cfg);z=embed(model,self.store.batch([0])).reshape(784,70);centers=z[::98].copy()
        assignment=distances(z,centers).argmin(1);ours=np.zeros((8,4),np.float64)
        hu=read_hu(self.record['dicom']);accumulate(ours,hu,self.record,assignment,self.cfg)
        tree=ast.parse((ROOT/'audit/ensemble/verify4/run_p1_optimized.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and ast.unparse(n.target)=='(n, i)')
        yy,xx=np.mgrid[-5:6,-5:6]
        ns=dict(np=np,pydicom=pydicom,ndi=ndi,largest=largest,distances=distances,rr=[self.record],train_ids=[0],G=28,disk=xx*xx+yy*yy<=25,z_train=z,centers=centers,sums=np.zeros((8,4),np.float64))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'original N2 accumulation','exec'),ns)
        np.testing.assert_array_equal(ours,ns['sums'])
        result=select(ours,self.cfg)
        candidates=[i for i,row in enumerate(ours) if row[0] and row[1]/row[0]<.5 and row[2]/row[0]<.6 and 40<=row[3]/row[0]<=180]
        expected=max(candidates,key=lambda i:ours[i,0]) if candidates else None
        self.assertEqual(result['selected_cluster'],expected)

    def test_loss_and_gradient_parity(self):
        cfg=SimpleNamespace(**load_config(HERE/'configs/train_config.yaml')['training'])
        ns=dict(torch=torch,nn=nn,F=F)
        source=(ROOT/'experiments/stego_colocated/reference_loss.py').read_text()
        exec(compile(source,'original reference loss','exec'),ns)
        definitions(ROOT/'experiments/stego_colocated/loss.py',{'ColocatedCorrelationLoss'},ns)
        torch.manual_seed(73);x=F.normalize(torch.randn(3,768,8,8),dim=1);xp=F.normalize(torch.randn_like(x),dim=1)
        z=torch.randn(3,70,8,8,requires_grad=True);zp=torch.randn_like(z,requires_grad=True)
        valid=torch.tensor([True,False,True]);new=ColocatedCorrelationLoss(cfg,valid);old=ns['ColocatedCorrelationLoss'](cfg,valid)
        torch.manual_seed(22);a,_=objective(new,cfg,x,xp,z,zp)
        ga=torch.autograd.grad(a,(z,zp),retain_graph=True)
        torch.manual_seed(22);b,_=objective(old,cfg,x,xp,z,zp)
        gb=torch.autograd.grad(b,(z,zp))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for v,w in zip(ga,gb):torch.testing.assert_close(v,w,rtol=0,atol=0)

    def test_adjacent_boundary_rule(self):
        records=[dict(patient_id='1',series_uid='s',physical_position=i) for i in range(4)]
        partners,valid,offsets=adjacent_partners(records,list(range(4)),np.random.default_rng(0))
        for i,(p,v,o) in enumerate(zip(partners,valid,offsets)):
            self.assertEqual(v,0<=i+o<4);self.assertEqual(p,i+o if v else i)

    def test_annotation_guard(self):
        with labels_closed():
            with self.assertRaises(PermissionError):
                (ROOT/'audit/improve/scoring_masks_native.npy').open('rb')

    def test_metrics(self):
        p=np.array([1,1,0,0]);y=np.array([1,0,1,0]);s=scores(counts(p,y))
        self.assertAlmostEqual(s['liver_iou'],1/3);self.assertEqual(s['liver_dice'],.5)
        a=aggregate([dict(patient='a',**s),dict(patient='b',**scores([2,0,0,2]))])
        self.assertAlmostEqual(a['patient_summary']['pooled']['liver_iou']['mean'],2/3)

    def test_fit_identity_rejection(self):
        with tempfile.TemporaryDirectory(dir=HERE/'verification') as folder:
            folder=Path(folder);np.save(folder/'centers.npy',np.zeros((8,70),np.float32))
            write_json(folder/'fit.json',dict(contract=contract(self.cfg),checkpoint_sha256='abc',centers_sha256=sha256(folder/'centers.npy'),temperature=.1))
            load_fit(folder,self.cfg,'abc')
            with self.assertRaises(ValueError):load_fit(folder,self.cfg,'wrong')
            np.save(folder/'centers.npy',np.ones((8,70),np.float32))
            with self.assertRaises(ValueError):load_fit(folder,self.cfg,'abc')

    def test_distance_and_soft_assignment_parity(self):
        from src2.clustering import fields
        tree=ast.parse((ROOT/'audit/final_pipeline/segment_ct.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='distances')
        ns=dict(np=np)
        exec(ast.unparse(fn).replace('reshape(-1, 768)','reshape(-1, centers.shape[1])'),ns)
        rng=np.random.default_rng(42);z=rng.normal(size=(784,70)).astype(np.float32);centers=rng.normal(size=(8,70)).astype(np.float32)
        expected=ns['distances'](z,centers)
        np.testing.assert_array_equal(distances(z,centers),expected)
        raw,field=fields(z,centers,.2,3)
        weights=np.exp(-(expected-expected.min(1,keepdims=True))/.2);weights/=weights.sum(1,keepdims=True)
        np.testing.assert_array_equal(field,weights[:,3].reshape(28,28))
        np.testing.assert_array_equal(raw,expected.argmin(1).reshape(28,28))

    def test_eval_seals_before_synthetic_labels(self):
        # Exercise the whole save/score path on TWO synthetic annotations only.
        # Feature records are mocked; no k-means fit and no real annotation access.
        from src2 import eval_segmentation as evaluation
        from src2.clustering import fields
        model,head_sha=load_head(self.cfg)
        x=self.store.batch([0,1]);z=embed(model,x);centers=z.reshape(-1,70)[::196][:8].copy()
        rr=[dict(self.store.records[i],patient=str(i),split='val') for i in range(2)]
        class TinyStore:
            records=rr
            order_sha256='synthetic-order'
            def indices(self,split):return [0,1]
            def batch(self,indices):return x[indices]
        with tempfile.TemporaryDirectory(dir=HERE/'verification') as temporary:
            base=Path(temporary);cache=base/'features.npy';np.save(cache,np.zeros((1,),np.float32))
            tiny=TinyStore();tiny.path=cache
            label_path=base/'scoring_masks_synthetic.npy'
            y=np.zeros((2,512,512),np.uint8);y[:,100:180,100:180]=1;np.save(label_path,y)
            cfg=dict(self.cfg,paths=dict(self.cfg['paths'],scoring_masks=str(label_path)))
            fitted=dict(ordered_sample_ids_sha256=tiny.order_sha256,feature_cache_sha256=sha256(cache),naming=dict(selected_cluster=3),temperature=.1)
            fit=base/'fit';fit.mkdir();write_json(fit/'fit.json',fitted)
            out=base/'evaluation'
            actual_load=evaluation.np.load
            def check_load(path,*args,**kwargs):
                if Path(path)==label_path:
                    self.assertTrue((out/'prediction_manifest.json').is_file())
                return actual_load(path,*args,**kwargs)
            with patch.object(evaluation,'FeatureStore',return_value=tiny),patch.object(evaluation,'load_head',return_value=(model,head_sha)),patch.object(evaluation,'load_fit',return_value=(centers,fitted)),patch.object(evaluation.np,'load',side_effect=check_load):
                evaluation.evaluate(cfg,fit,out)
            results=json.loads((out/'metrics.json').read_text())
            self.assertEqual(len(results['results']['refined']['512']['patients']),2)
            manifest=json.loads((out/'prediction_manifest.json').read_text())
            for name,h in manifest['outputs'].items():self.assertEqual(sha256(out/name),h)


if __name__=='__main__':unittest.main()
