"""Centre distances, explicit optimized embedding and immutable fit loading."""
import numpy as np
import torch
from .utils import read_json, sha256, contract


def distances(features, centers):
    z = features.reshape(-1, centers.shape[1])
    return np.maximum((z*z).sum(1)[:, None]+(centers*centers).sum(1)[None]-2*z@centers.T, 0)


def embed(model, features):
    with torch.no_grad():
        return model.embed(features).permute(0, 2, 3, 1).contiguous().numpy()


def fields(z, centers, temperature, selected):
    d = distances(z, centers)
    raw = d.argmin(1).reshape(28, 28).astype(np.uint8)
    weights = np.exp(-(d-d.min(1, keepdims=True))/temperature)
    weights /= weights.sum(1, keepdims=True)
    field = weights[:, selected].reshape(28, 28) if selected is not None else np.zeros((28, 28), np.float32)
    return raw, field


def load_fit(folder, cfg, checkpoint_sha):
    from pathlib import Path
    folder = Path(folder); meta = read_json(folder/'fit.json')
    if meta['contract'] != contract(cfg) or meta['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Fit contract/checkpoint mismatch')
    if sha256(folder/'centers.npy') != meta['centers_sha256']:
        raise ValueError('Centre digest mismatch')
    c = np.load(folder/'centers.npy', allow_pickle=False)
    if c.shape != (8, 70) or not np.isfinite(c).all() or meta['temperature'] <= 0:
        raise ValueError('Invalid fitted centres/temperature')
    if not np.isfinite(meta['temperature']):
        raise ValueError('Non-finite fitted temperature')
    named = meta.get('naming')
    if named is not None:
        selected = named['selected_cluster']
        if named['rule'] != 'N2' or (selected is not None and (type(selected) is not int or not 0 <= selected < 8)):
            raise ValueError('Invalid N2 selection')
    return c, meta
