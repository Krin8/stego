"""N2 only: train-set cluster area, body-relative centroid and raw HU."""
import numpy as np
from scipy import ndimage as ndi
from .postprocessing import largest


def accumulate(sums, hu, record, assignment, cfg):
    p = cfg['naming']; G = 28
    radius = p['opening_radius']
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    body = ndi.binary_fill_holes(largest(ndi.binary_opening(hu > p['body_hu'], structure=xx*xx+yy*yy <= radius**2)))
    ys, xs = np.where(body)
    if not len(ys):
        raise ValueError('N2 body mask is empty; no undeclared fallback')
    y0, x0, side = record['box']
    gy, gx = np.mgrid[:G, :G]
    cy = y0+(gy.ravel()+.5)*side/G-.5; cx = x0+(gx.ravel()+.5)*side/G-.5
    iy = np.floor(cy+.5).astype(int); ix = np.floor(cx+.5).astype(int)
    eligible = (iy>=0)&(ix>=0)&(iy<512)&(ix<512)
    valid = np.flatnonzero(eligible); eligible[valid] = body[iy[valid], ix[valid]]; use = np.flatnonzero(eligible)
    ya = np.clip(np.ceil(y0+gy.ravel()*side/G-.5).astype(int), 0, 512)
    yb = np.clip(np.ceil(y0+(gy.ravel()+1)*side/G-.5).astype(int), 0, 512)
    xa = np.clip(np.ceil(x0+gx.ravel()*side/G-.5).astype(int), 0, 512)
    xb = np.clip(np.ceil(x0+(gx.ravel()+1)*side/G-.5).astype(int), 0, 512)
    def rectsum(a):
        integral = np.pad(a, ((1, 0), (1, 0))).cumsum(0, dtype=np.float64).cumsum(1, dtype=np.float64)
        return integral[yb, xb]-integral[ya, xb]-integral[yb, xa]+integral[ya, xa]
    count = rectsum(body); total = rectsum(hu*body)
    means = np.divide(total, count, out=np.zeros_like(total), where=count>0)
    empty = use[count[use]==0]; means[empty] = hu[iy[empty], ix[empty]]
    nx = (cx-xs.min())/max(1, xs.max()-xs.min()); ny = (cy-ys.min())/max(1, ys.max()-ys.min())
    for cluster in range(8):
        u = use[np.asarray(assignment).ravel()[use] == cluster]
        sums[cluster, 0] += len(u)
        sums[cluster, 1:] += [nx[u].sum(), ny[u].sum(), means[u].sum()]


def select(sums, cfg):
    p = cfg['naming']; stats = []
    for k in range(8):
        count = sums[k, 0]
        x, y, h = sums[k, 1:]/count if count else [1., 1., 0.]
        stats.append(dict(cluster=k, area=float(count/max(1, sums[:, 0].sum())),
            centroid_x=float(x), centroid_y=float(y), mean_hu=float(h),
            candidate=bool(count and x<p['x_max'] and y<p['y_max'] and p['hu_min']<=h<=p['hu_max'])))
    candidates = [s for s in stats if s['candidate']]
    selected = max(candidates, key=lambda s:s['area'])['cluster'] if candidates else None
    return dict(rule='N2', selected_cluster=selected, abstained=selected is None, clusters=stats)
