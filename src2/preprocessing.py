"""Native CT HU -> body crop -> 448 RGB tensor, with normalization once."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF, InterpolationMode


def read_hu(path):
    import pydicom
    ds = pydicom.dcmread(path)
    if ds.Modality != 'CT':
        raise ValueError('This pipeline accepts CT only')
    hu = ds.pixel_array.astype(np.float32) * float(getattr(ds, 'RescaleSlope', 1))
    hu += float(getattr(ds, 'RescaleIntercept', 0))
    if hu.shape != (512, 512) or not np.isfinite(hu).all():
        raise ValueError('Expected a finite native 512x512 CT slice')
    return hu


def load_image(path, cfg):
    path = Path(path)
    if path.suffix.lower() == '.png':
        with Image.open(path) as im:
            a = np.asarray(im)
            if a.ndim != 2 or a.dtype not in (np.dtype('uint16'), np.dtype('int32')) or im.mode not in ('I;16', 'I;16L', 'I;16B', 'I'):
                raise ValueError('PNG must be windowed uint16 grayscale, not 8-bit/RGB')
            if a.min() < 0 or a.max() > 65535:
                raise ValueError('PNG values must be in 0..65535')
            a = a.astype(np.float32) / 65535.
    else:
        lo, hi = cfg['preprocessing']['window_hu']
        a = np.clip((read_hu(path) - lo) / (hi - lo), 0, 1)
        a = np.rint(a * 65535).astype(np.uint16).astype(np.float32) / 65535.
    if a.shape != (512, 512):
        raise ValueError('The declared native frame is 512x512')
    return a


def body_box(im):
    # Exact feature-crop rule; N2 uses a different HU-based body mask.
    lab, n = ndi.label(im > .05, structure=np.ones((3, 3), dtype=int))
    if n:
        counts = np.bincount(lab.ravel()); counts[0] = 0
        body = ndi.binary_fill_holes(lab == counts.argmax())
        yy, xx = np.where(body)
        side = int(np.ceil(max(yy.max()-yy.min()+1, xx.max()-xx.min()+1)*1.1))
        y0 = int(np.floor((yy.min()+yy.max()+1-side)/2))
        x0 = int(np.floor((xx.min()+xx.max()+1-side)/2))
        return [y0, x0, side]
    return [0, 0, max(im.shape)]


def crop_image(im, box):
    y, x, s = box; h, w = im.shape
    out = np.zeros((s, s), np.float32)
    ya, xa, yb, xb = max(0, y), max(0, x), min(h, y+s), min(w, x+s)
    out[ya-y:yb-y, xa-x:xb-x] = im[ya:yb, xa:xb]
    return out


def image_tensor(im, cfg):
    p = cfg['preprocessing']
    t = torch.from_numpy(np.array(im, dtype=np.float32)).unsqueeze(0).repeat(3, 1, 1)
    t = TF.resize(t, [p['resolution']]*2, interpolation=InterpolationMode.BILINEAR, antialias=True)
    return TF.normalize(t, p['mean'], p['std'])


def inverse_grid(grid, record, mode='bilinear', outside=0):
    y, x, s = record['box']; h, w = record['shape']
    t = torch.from_numpy(np.asarray(grid, dtype=np.float32).copy())[None, None]
    kw = dict(align_corners=False) if mode == 'bilinear' else {}
    crop = F.interpolate(t, size=(s, s), mode=mode, **kw)[0, 0].numpy()
    result = np.full((h, w), outside, dtype=np.float32)
    ya, xa, yb, xb = max(0, y), max(0, x), min(h, y+s), min(w, x+s)
    result[ya:yb, xa:xb] = crop[ya-y:yb-y, xa-x:xb-x]
    return result


def resize224(a):
    return F.interpolate(torch.tensor(np.array(a), dtype=torch.float32)[None, None], (224, 224), mode='nearest')[0, 0].numpy()
