"""Synthetic identity/metadata checks; NPZ bytes stay in memory."""
import fixes2_guard as g
import sys,copy,io,json,subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(g.ROOT/'src'))
import knn_cache as kc
from chaos_preprocessing import get_chaos_transform,canonical_json
cfg=SimpleNamespace(arch='biomedclip',knn_k=3)
def dataset(root):
    return SimpleNamespace(image_set='train',modality='CT',transform=get_chaos_transform(),sample_ids=list('abcd'),samples=[(str(Path(root)/f'img{i}.png'),None,'CT','1') for i in range(4)],split_manifest={'fixed':True})
def identity(root,commit,changed=None):
    def content_hash(path):return ('changed-' if Path(path).name==changed else 'same-')+Path(path).name
    with patch.object(kc,'file_sha256',side_effect=content_hash),patch.object(kc,'git_commit',return_value=commit),patch.object(kc,'WEIGHT_FILE',Path(root)/'open_clip_pytorch_model.bin'):
        return kc.expected_identity(dataset(root),cfg)
a=identity('/Users/developer/project','commitA');b=identity('/content/drive/project','docsCommitB')
if '--expect-fixed' not in sys.argv:
    assert a!=b and kc.cache_path('/one',a).name!=kc.cache_path('/two',b).name
    print('BEFORE: changing only git commit invalidates otherwise identical cache content')
else:
    assert a==b and kc.cache_path('/one',a).name==kc.cache_path('/two',b).name
    assert 'git_commit' not in a
    text=canonical_json(a)
    assert not any(x in text for x in ['/Users/','/content/','device','backend','library_version'])
    nns=np.array([[0,1,2],[1,0,2],[2,0,1],[3,0,1]],np.int64)
    with patch.object(kc,'git_commit',return_value='originalCommit'):meta=kc.cache_metadata(a,nns)
    assert meta['provenance']['git_commit']=='originalCommit'
    meta['provenance']['git_commit']='unrelatedDocsCommit'
    def stream():
        f=io.BytesIO();np.savez_compressed(f,nns=nns,metadata=np.frombuffer(canonical_json(meta).encode(),np.uint8));f.seek(0);return f
    assert np.array_equal(kc.read_cache(stream(),b),nns)
    for changed in ['modules.py','chaos_preprocessing.py','data.py','precompute_knns.py','open_clip_pytorch_model.bin','img0.png']:
        wrong=identity('/content/drive/project','commitA',changed)
        assert wrong!=a
        try:kc.read_cache(stream(),wrong)
        except ValueError:pass
        else:raise AssertionError(changed+' mismatch accepted')
    with patch.object(kc.subprocess,'check_output',side_effect=subprocess.CalledProcessError(128,['git'])):
        assert kc.git_commit() is None
    print('AFTER PASS: same identity across checkout roots/docs commits; provenance ignored; code/weights/image changes rejected; git-less provenance allowed')
    print('Identity fields:',sorted(a));print('Machine-specific identity fields: none')
