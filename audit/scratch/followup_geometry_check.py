"""Header/array inspection only: no derivative or manifest writes."""
import fixes2_guard as g
import sys,json,collections,numpy as np
sys.path.insert(0,str(g.ROOT/'src'))
from chaos_preprocessing import plan_dataset,preprocessing_contract,read_image
root=g.ROOT/'src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets'
records=plan_dataset(root)
assert preprocessing_contract()['spatial_resampling']=='none'
assert preprocessing_contract()['orientation']=='native; no flip or transpose'
counts=collections.Counter((r['split'],r['modality']) for r in records)
shapes=collections.Counter((r['modality'],tuple(r['shape'])) for r in records)
groups=collections.defaultdict(list)
for r in records:groups[(r['modality'],r['patient_id'])].append(r['physical_position'])
spacing=collections.defaultdict(list)
for (mod,pid),positions in groups.items():spacing[mod].extend(np.diff(positions).tolist())
for mod in ['CT','T1DUAL','T2SPIR']:
    r=next(r for r in records if r['modality']==mod)
    image=read_image(root/r['dicom'],mod)
    assert list(image.shape)==r['shape'] and image.dtype==np.uint16
result=dict(counts={f'{s}/{m}':n for (s,m),n in counts.items()},written_native_shapes={f'{m}/{h}x{w}':n for (m,(h,w)),n in shapes.items()},native_slice_interval_mm={m:sorted(set(round(x,4) for x in v)) for m,v in spacing.items()},preprocessing_contract=preprocessing_contract(),generation=False)
print(json.dumps(result,indent=2))
