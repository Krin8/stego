"""The selected size-25 guided filter and threshold-.30 refinement."""
import numpy as np
from scipy import ndimage as ndi
from .preprocessing import inverse_grid


def largest(mask, structure=None):
    if structure is None:
        structure = np.ones((3, 3), int)
    labels, n = ndi.label(mask, structure=structure)
    if not n:
        return mask
    counts = np.bincount(labels.ravel()); counts[0] = 0
    return labels == counts.argmax()


def guided(image, field, size=25, epsilon=.01):
    mi = ndi.uniform_filter(image, size=size, mode='reflect')
    mp = ndi.uniform_filter(field, size=size, mode='reflect')
    var = ndi.uniform_filter(image*image, size=size, mode='reflect') - mi*mi
    cov = ndi.uniform_filter(image*field, size=size, mode='reflect') - mi*mp
    aa = cov / (var + epsilon)
    bb = mp - aa*mi
    return np.clip(ndi.uniform_filter(aa, size=size, mode='reflect')*image + ndi.uniform_filter(bb, size=size, mode='reflect'), 0, 1)


def refine(field, image, record, cfg):
    p = cfg['postprocessing']
    soft = inverse_grid(field, record)
    filtered = guided(image, soft, p['guided_size'], p['guided_epsilon'])
    pred = largest(filtered >= p['threshold'])
    pred = ndi.binary_fill_holes(pred)
    rad = p['closing_radius']
    yy, xx = np.mgrid[-rad:rad+1, -rad:rad+1]
    pred = ndi.binary_closing(np.pad(pred, rad), structure=xx*xx+yy*yy <= rad**2)[rad:-rad, rad:-rad]
    return pred
