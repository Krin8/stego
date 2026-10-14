"""Ordered cached features and patient metadata; no annotation loading here."""
import numpy as np
import torch
from .utils import read_json, resolve_path, digest


def records(cfg):
    rr = read_json(resolve_path(cfg, cfg['paths']['records']))
    meta = read_json(resolve_path(cfg, cfg['paths']['feature_metadata']))['records']
    if len(rr) != len(meta):
        raise ValueError('Feature/record count mismatch')
    for a, b in zip(rr, meta):
        if any(a[k] != b[k] for k in ('sample_id', 'patient', 'split', 'box', 'shape')):
            raise ValueError('Cached feature ordering or geometry mismatch')
    split = read_json(resolve_path(cfg, cfg['paths']['split_manifest']))['patients']['CT']
    if set(split['train']) & set(split['val']):
        raise ValueError('Patient leakage in split manifest')
    if len({r['sample_id'] for r in rr}) != len(rr):
        raise ValueError('Duplicate sample IDs')
    for r in rr:
        if r['patient'] not in split[r['split']] or r['shape'] != [512, 512]:
            raise ValueError('Unexpected split or native image shape')
    return rr


class FeatureStore:
    def __init__(self, cfg):
        self.records = records(cfg)
        self.path = resolve_path(cfg, cfg['paths']['features'])
        self.array = np.load(self.path, mmap_mode='r', allow_pickle=False)
        if self.array.shape != (len(self.records), 784, 768) or self.array.dtype != np.float32:
            raise ValueError('Expected float32 N x 784 x 768 cache')
        self.order_sha256 = digest([r['sample_id'] for r in self.records])

    def indices(self, split):
        return [i for i, r in enumerate(self.records) if r['split'] == split]

    def batch(self, indices):
        a = np.stack([self.array[int(i)] for i in indices])
        if not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a, axis=-1), 1, atol=2e-4):
            raise ValueError('Feature cache is not finite patch-L2-normalized data')
        return torch.from_numpy(a).reshape(len(indices), 28, 28, 768).permute(0, 3, 1, 2).contiguous()


def train_geometry(cfg, store):
    lookup = {r['sample_id']: r for r in read_json(resolve_path(cfg, cfg['paths']['train_geometry']))}
    return [lookup[store.records[i]['sample_id']] for i in store.indices('train')]
