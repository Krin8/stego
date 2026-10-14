import fixes2_guard as g
import json,sys
sys.path.insert(0,str(g.ROOT/'src'))
path=g.ROOT/'src/configs/chaos_splits_v2.json'
current={}
for mod in ['CT','T1DUAL','T2SPIR']:
    current[mod]={}
    for split in ['train','val']:
        files=(g.ROOT/'src/pytorch_data_dir/preprocessed'/split/'images').glob(mod+'_*.png')
        ids={p.stem.split('_')[1] for p in files}
        if mod=='CT':ids={x for x in ids if not x.startswith('p')}
        current[mod][split]=sorted({str(int(x.removeprefix('p'))) for x in ids},key=int)
if not path.exists():
    ids=sorted(current['CT']['train']+current['CT']['val']);before=ids[-int(.2*len(ids)):]
    ids=sorted(ids+['99']);after=ids[-int(.2*len(ids)):]
    assert set(before)!=set(after)
    print('BEFORE: no frozen manifest; adding patient99 changes validation assignment:',before,'->',after)
    assert '--expect-fixed' not in sys.argv
else:
    from chaos_splits import load_splits,patient_split
    manifest=load_splits();assert manifest['patients']==current
    try:patient_split(manifest,'CT','99')
    except ValueError:pass
    else:raise AssertionError('Unknown patient accepted')
    broken=json.loads(json.dumps(manifest));broken['patients']['CT']['val'].append(current['CT']['train'][0])
    from chaos_splits import validate_splits
    try:validate_splits(broken)
    except ValueError:pass
    else:raise AssertionError('Patient overlap accepted')
    print('AFTER PASS: frozen current assignments, unknown IDs rejected, overlap rejected',json.dumps(current))
