import fixes2_guard as g
import sys,io,numpy as np,torch
from PIL import Image
from unittest.mock import patch
sys.path.insert(0,str(g.ROOT/'src'));torch.set_num_threads(2)
import chaos_preprocessing as cp
if not hasattr(cp,'get_chaos_transform'):
    s=(g.ROOT/'src/data.py').read_text();d=(g.ROOT/'src/demo_segmentation.py').read_text()
    assert 'MRI needs a verified' in s and 'ImageOps.equalize' in d and 'preprocessed_v2' not in s
    print('BEFORE: loader uses old family, MRI rejected, demo equalizes image; entrypoint transforms disagree')
    assert '--expect-fixed' not in sys.argv
else:
    from data import CHAOS
    transform=cp.get_chaos_transform();target=cp.get_chaos_transform(is_label=True)
    for mod in cp.MODALITIES:
        a=np.arange(64,dtype=np.float32).reshape(8,8)*10
        quant=np.rint(cp.normalize_pixels(a,mod)*65535).astype(np.uint16)
        b=io.BytesIO();Image.fromarray(quant).save(b,format='PNG');b.seek(0)
        raw=transform(quant);derivative=transform(Image.open(b))
        assert torch.equal(raw,derivative) and raw.shape==(3,224,224)
    q=np.full((3,5),32768,np.uint16);got=transform(q)
    expected=(torch.tensor(32768/65535)-torch.tensor(cp.MEAN))/torch.tensor(cp.STD)
    assert torch.allclose(got[:,0,0],expected,atol=1e-6)
    label=target(Image.fromarray(np.array([[0,1],[3,4]],np.uint8)));assert set(label.unique().tolist())<={0,1,3,4}
    assert len(set(str(getattr(transform,k,None)) for k in ['res']))==1
    # Missing v2 data must fail explicitly rather than discover the old folder.
    try:CHAOS(str(g.ROOT/'src/pytorch_data_dir'),'CT','train',transform,target,2,True)
    except FileNotFoundError as e:assert 'preprocessed_v2' in str(e)
    else:raise AssertionError('Unexpected v2 data/legacy fallback')
    source=(g.ROOT/'src/data.py').read_text();assert 'Mixed preprocessing families' in source
    for path in ['train_segmentation.py','precompute_knns.py','eval_segmentation.py','demo_segmentation.py']:
        assert 'get_chaos_transform' in (g.ROOT/'src'/path).read_text()
    print('AFTER PASS: raw/PNG tensors identical across3 modalities, exact one CLIP normalization, nearest labels, missing-v2 guard, shared entrypoint transforms')
