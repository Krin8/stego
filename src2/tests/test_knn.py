"""Train-only neighbour selection and original full-matrix loss checks; no runs."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from src2.utils import HERE, load_config, setup, labels_closed
from src2.correspondence import feature_neighbors, prepare_correspondence
from src2.reference_loss import ContrastiveCorrelationLoss
from src2.loss import ColocatedCorrelationLoss
from src2.modules import StegoHead, objective
from src2.data import FeatureStore


class FakeStore:
    # Identical vectors deliberately test deterministic ties and self exclusion.
    array = np.array([[[1., 0.]], [[0., 1.]], [[1., 0.]], [[-1., 0.]], [[0., 1.]]], dtype=np.float32)
    records = [dict(sample_id=str(i), split='val' if i == 1 else 'train') for i in range(5)]

    def indices(self, split):
        return [i for i, r in enumerate(self.records) if r['split'] == split]


class KNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(HERE/'configs/train_knn_config.yaml')
        setup(cls.cfg)

    def test_train_only_ranking_and_stable_ties(self):
        store = FakeStore()
        with labels_closed():
            neighbors, meta = feature_neighbors(store, 3)
        np.testing.assert_array_equal(neighbors, [[1, 3, 2], [0, 3, 2], [3, 0, 1], [0, 1, 2]])
        self.assertEqual(meta['ordered_sample_ids'], ['0', '2', '3', '4'])
        self.assertFalse(np.any(neighbors == np.arange(4)[:, None]))
        with self.assertRaises(ValueError): feature_neighbors(store, 4)
        store.array = store.array.copy(); store.array[0] = np.nan
        with self.assertRaises(ValueError): feature_neighbors(store, 3)

    def test_knn_sampling_and_original_criterion(self):
        training = SimpleNamespace(**dict(self.cfg['training'], knn_k=3, num_neighbors=2))
        # A kNN run must not require or open the adjacency/physical-position file.
        with patch('src2.correspondence.train_geometry', side_effect=AssertionError('P1 geometry opened')):
            criterion, choose, _, neighbors = prepare_correspondence(self.cfg, FakeStore(), training)
        self.assertIs(type(criterion), ContrastiveCorrelationLoss)
        anchors = list(range(4))*100
        selected, valid = choose(anchors, np.random.default_rng(0))
        again, _ = choose(anchors, np.random.default_rng(0))
        self.assertEqual(selected, again); self.assertIsNone(valid)
        for i,j in zip(anchors,selected):self.assertIn(j,neighbors[i,:2])
        self.assertTrue(all(len({j for a,j in zip(anchors,selected) if a==i})==2 for i in range(4)))

    def test_loss_matches_reference_and_uses_off_diagonal_pairs(self):
        c = SimpleNamespace(**self.cfg['training'])
        with patch('src2.correspondence.feature_neighbors', return_value=(np.zeros((32,30),dtype=int), {})):
            criterion, _, _, _ = prepare_correspondence(self.cfg, FakeStore(), c)
        torch.manual_seed(18)
        f = F.normalize(torch.randn(3,768,5,5),dim=1); fp=F.normalize(torch.randn_like(f),dim=1)
        z=torch.randn(3,70,5,5,requires_grad=True);zp=torch.randn_like(z,requires_grad=True)
        torch.manual_seed(31);a,parts=objective(criterion,c,f,fp,z,zp)
        grad_a=torch.autograd.grad(a,(z,zp),retain_graph=True)
        torch.manual_seed(31);b,_=objective(ContrastiveCorrelationLoss(c),c,f,fp,z,zp)
        grad_b=torch.autograd.grad(b,(z,zp),retain_graph=True)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for u,v in zip(grad_a,grad_b):torch.testing.assert_close(u,v,rtol=0,atol=0)
        torch.manual_seed(31);_,p1=objective(ColocatedCorrelationLoss(c,torch.ones(3,dtype=torch.bool)),c,f,fp,z,zp)
        self.assertNotEqual(float(parts['pos_inter'].detach()),float(p1['pos_inter'].detach()))
        # Replay exactly the reference's coordinate RNG and full positive matrix.
        torch.manual_seed(31)
        coords1=torch.rand((3,c.feature_samples,c.feature_samples,2))*2-1
        coords2=torch.rand_like(coords1)*2-1
        from src2.reference_loss import sample
        pair,_=criterion.helper(sample(f,coords1),sample(fp,coords2),sample(z,coords1),sample(zp,coords2),c.pos_inter_shift)
        self.assertEqual(pair.shape,(3,11,11,11,11))
        torch.testing.assert_close(parts['pos_inter'],pair.mean(),rtol=0,atol=0)

    def test_p1_default_preserved(self):
        cfg=load_config(HERE/'configs/train_config.yaml');c=SimpleNamespace(**cfg['training'])
        geometry=[dict(patient_id='1',series_uid='s',physical_position=i) for i in range(4)]
        with patch('src2.correspondence.train_geometry',return_value=geometry):
            criterion,choose,_,neighbors=prepare_correspondence(cfg,FakeStore(),c)
        self.assertIs(type(criterion),ColocatedCorrelationLoss);self.assertIsNone(neighbors)
        from src2.loss import adjacent_partners
        actual,valid=choose([0,1,2,3],np.random.default_rng(0))
        expected,expected_valid,_=adjacent_partners(geometry,[0,1,2,3],np.random.default_rng(0))
        self.assertEqual(actual,expected);self.assertEqual(valid.tolist(),expected_valid)
        changed={k for k in self.cfg['training'] if self.cfg['training'][k]!=cfg['training'].get(k)}
        self.assertEqual(changed,{'partner_mode','knn_k','num_neighbors'})

    def test_real_features_head_backward_without_optimizer(self):
        with labels_closed():
            store=FeatureStore(self.cfg)
            ids=store.indices('train')[:4]
            model=StegoHead();c=SimpleNamespace(**self.cfg['training'])
            f,z=model(store.batch(ids));fp,zp=model(store.batch(ids[1:]+ids[:1]))
            loss,parts=objective(ContrastiveCorrelationLoss(c),c,f,fp,z,zp)
            loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(z.shape,(4,70,28,28))


if __name__=='__main__': unittest.main()
