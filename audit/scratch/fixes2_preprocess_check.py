import fixes2_guard as g
import sys,numpy as np
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(g.ROOT/'src'))
if not (g.ROOT/'src/chaos_preprocessing.py').exists():
    ns=g.definitions('src/preprocess.py',['HU_MIN','HU_MAX','clip_and_normalize'],dict(np=np))
    actual=ns['clip_and_normalize'](np.array([-150.,50.,250.]))
    assert actual.tolist()!=[0,128,255]
    source=(g.ROOT/'src/preprocess.py').read_text();assert 'dict(zip(' in source
    print('BEFORE: CT endpoints/midpoint',actual.tolist(),'; pairing is rank zip; no MRI percentile path or dry-run')
    assert '--expect-fixed' not in sys.argv
else:
    import chaos_preprocessing as cp
    assert np.allclose(cp.normalize_pixels(np.array([-150.,50.,250.]),'CT'),[0,.5,1])
    assert np.allclose(cp.normalize_pixels(np.array([0.,100.,200.]),'CT',slope=2,intercept=-150),[0,.5,1])
    assert np.array_equal(cp.normalize_pixels(np.ones((4,4))*17,'T1DUAL'),np.zeros((4,4)))
    a=np.arange(10000,dtype=np.float32).reshape(100,100);lo,hi=np.percentile(a,[1,99.9])
    assert np.allclose(cp.normalize_pixels(a,'T2SPIR'),np.clip((a-lo)/(hi-lo),0,1))
    try:cp.normalize_pixels(a,'OutPhase')
    except ValueError:pass
    else:raise AssertionError('Excluded sequence accepted')
    # Explicit ID matching must survive reversed image discovery and missing masks.
    from pathlib import Path
    d=Path('/synthetic/DICOM_anon');m=Path('/synthetic/Ground')
    imgs=[d/'b.dcm',d/'a.dcm'];masks=[m/'liver_GT_001.png',m/'liver_GT_000.png']
    headers={imgs[0]:SimpleNamespace(InstanceNumber=2,ImagePositionPatient=[0,0,-1],ImageOrientationPatient=[1,0,0,0,1,0],PixelSpacing=[1,1],Rows=4,Columns=4,SOPInstanceUID='2',SeriesInstanceUID='s'),imgs[1]:SimpleNamespace(InstanceNumber=1,ImagePositionPatient=[0,0,1],ImageOrientationPatient=[1,0,0,0,1,0],PixelSpacing=[1,1],Rows=4,Columns=4,SOPInstanceUID='1',SeriesInstanceUID='s')}
    class Img:
        size=(4,4)
        def __enter__(self):return self
        def __exit__(self,*a):pass
    with patch.object(Path,'glob',lambda p,pat:iter(imgs if p==d else masks)),patch.object(cp.pydicom,'dcmread',lambda p,**k:headers[p]),patch.object(cp.Image,'open',lambda p:Img()):
        records=cp.pair_series(d,m,'CT','1')
        assert [(r['instance'],Path(r['mask']).name) for r in records]==[(2,'liver_GT_001.png'),(1,'liver_GT_000.png')]
        masks.pop()
        try:cp.pair_series(d,m,'CT','1')
        except ValueError:pass
        else:raise AssertionError('Missing mask accepted')
    print('AFTER PASS: CT slope/window, MRI percentile/constant guard, excluded phase, ID association before physical sort, missing-mask rejection')
