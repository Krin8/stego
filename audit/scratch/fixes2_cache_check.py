import fixes2_guard as g
import sys,io,json,numpy as np
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(g.ROOT/'src'))
if not (g.ROOT/'src/knn_cache.py').exists():
    from data import ContrastiveSegDataset
    source=(g.ROOT/'src/data.py').read_text();writer=(g.ROOT/'src/precompute_knns.py').read_text()
    assert 'if "metadata" not in cache' in source and 'json.loads' not in source[source.index('self.nns = None'):]
    assert 'np.savez_compressed(feature_cache_file, nns=' in writer
    b=io.BytesIO();np.savez(b,nns=np.array([[0,999]]),metadata=np.array('wrong-backbone'))
    b.seek(0)
    with np.load(b,allow_pickle=False) as cache:
        assert 'metadata' in cache
        accepted=cache['nns']
    import data
    fake=SimpleNamespace(samples=[('a',None,'CT','1'),('b',None,'CT','1')])
    b.seek(0);original_load=np.load
    with patch.object(data,'CHAOS',return_value=fake),patch.object(data.os.path,'exists',return_value=True),patch.object(data.np,'load',lambda *a,**k:original_load(b,allow_pickle=False)):
        actual=ContrastiveSegDataset('/fake','chaos',None,'train',None,None,SimpleNamespace(),num_neighbors=1)
        assert actual.nns.tolist()==[[0,999]]
    print('BEFORE: current metadata-presence gate accepts arbitrary metadata and invalid index999; writer emits no identity and silently skips existing path')
    assert '--expect-fixed' not in sys.argv
else:
    import knn_cache as kc
    from chaos_preprocessing import get_chaos_transform,canonical_json
    cfg=SimpleNamespace(arch='biomedclip',knn_k=3,num_neighbors=2)
    ds=SimpleNamespace(image_set='train',modality='CT',transform=get_chaos_transform(),sample_ids=['a','b','c','d'],samples=[('img'+str(i),None,'CT','1') for i in range(4)],split_manifest={'fixed':True})
    with patch.object(kc,'file_sha256',lambda p:'hash-'+str(p)),patch.object(kc,'git_commit',lambda:'commit'):
        identity=kc.expected_identity(ds,cfg)
    nns=np.array([[0,1,2],[1,0,2],[2,1,3],[3,2,1]],dtype=np.int64)
    meta=kc.cache_metadata(identity,nns)
    def stream(array=nns,metadata=meta,include=True):
        b=io.BytesIO();values={'nns':array}
        if include:values['metadata']=np.frombuffer(canonical_json(metadata).encode('utf-8'),dtype=np.uint8)
        np.savez_compressed(b,**values);b.seek(0);return b
    assert np.array_equal(kc.read_cache(stream(),identity),nns)
    rejected=[]
    cases=[]
    for field,value in [('ordered_sample_ids',['b','a','c','d']),('backbone',{'weight_sha256':'wrong'}),('preprocessing',{'ct_window':[-400,400]}),('split','val'),('feature_code_sha256',{'modules.py':'other'})]:
        bad=json.loads(json.dumps(identity));bad[field]=value;cases.append((field,stream(),bad))
    cases.append(('missing-metadata',stream(include=False),identity))
    corrupt=nns.copy();corrupt[0,1]=999
    cases.append(('bounds',stream(corrupt,kc.cache_metadata(identity,corrupt)),identity))
    wrong_self=nns[:,::-1].copy();cases.append(('self-slot',stream(wrong_self,kc.cache_metadata(identity,wrong_self)),identity))
    dupe=nns.copy();dupe[0,2]=1;cases.append(('duplicates',stream(dupe,kc.cache_metadata(identity,dupe)),identity))
    cases.append(('digest',stream(nns+1,meta),identity))
    for name,b,expected in cases:
        try:kc.read_cache(b,expected)
        except ValueError:rejected.append(name)
        else:raise AssertionError(name+' accepted')
    from pathlib import Path
    with patch.object(Path,'exists',return_value=True),patch.object(Path,'rename') as rename:
        try:kc.prepare_target(Path('/fake/cache.npz'),False)
        except FileExistsError:pass
        else:raise AssertionError('Existing target accepted')
        rename.assert_not_called()
        kc.prepare_target(Path('/fake/cache.npz'),True);rename.assert_called_once()
    assert kc.cache_path('/root',identity).name!=kc.cache_path('/root',{**identity,'split':'val'}).name
    print('AFTER PASS: UTF-8 JSON roundtrip; rejected',rejected,'; overwrite refused; explicit archival spy; digest filename. No cache files created.')
