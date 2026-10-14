"""Record five fixed reference scores from cached features, then verify live project extraction."""
from study import *
rr=records();fixed=json.loads((OLD/'metadata.json').read_text())['fixed']
# First four pre-existing fixed validation slices + first remaining validation slice by stored order.
fixed=fixed+[next(i for i,r in enumerate(rr) if r['split']=='val' and i not in fixed)]
y=np.load(OLD/'scoring_masks_native.npy',mmap_mode='r');cached=np.load(OLD/'B2_448_features.npy',mmap_mode='r')
reference=dict(producer='package_check.py',selection='Four IMPROVE fixed val slices plus first other validation record; no label-based selection',iou_absolute_tolerance=1e-4,labels='all20 labels map deployment artifact; these validation masks test reproducibility only, not held-out performance',samples=[])
for i in fixed:
    r=rr[i];row=dict(index=i,stem=r['stem'],image=str(Path(r['image']).relative_to(ROOT)),mask=str(Path(r['mask']).relative_to(ROOT)),iou={})
    for seed in [0,1,2]:
        centers,meta=load_artifact(seed);_,pred=predict_features(cached[i],load_plane(r['image']),r,centers,meta)
        row['iou'][str(seed)]=scores(confusion(pred,y[i],2),[1])['iou']
    reference['samples'].append(row)
dump('check_reference.json',reference)
centers,meta=load_artifact(0);net=load_extractor(meta);live=[]
for row in reference['samples']:
    i=row['index'];r=rr[i];im=load_plane(r['image']);f,rec=extract(net,im)
    assert rec['box']==r['box'];err=float(np.abs(f-cached[i]).max());assert err<1e-5,(r['stem'],err)
    _,pred=predict_features(f,im,rec,centers,meta);iou=scores(confusion(pred,y[i],2),[1])['iou']
    diff=abs(iou-row['iou']['0']);assert diff<=reference['iou_absolute_tolerance']
    save_output(HERE/'check_outputs',r['stem'],im,pred)
    live.append(dict(stem=r['stem'],features_max_abs_error=err,live_iou=iou,reference_iou=row['iou']['0'],iou_absolute_error=diff))
# Verify DICOM and derivative equivalence for the same fixed slices.
manifest_path=Path(rr[0]['image']).parents[2]/'manifest.json';manifest=json.loads(manifest_path.read_text());by_stem={r['stem']:r for r in manifest['samples']}
possible=[ROOT/'src/pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets',ROOT/'data/CHAOS_Train_Sets/Train_Sets',ROOT/'CHAOS_Train_Sets/Train_Sets',ROOT/'src/pytorch_data_dir/Train_Sets']
raw=next((p for p in possible if p.is_dir()),None)
dicom=[]
if raw:
    for i in fixed:
        r=rr[i];p=raw/by_stem[r['stem']]['dicom']
        if p.is_file():
            a=load_plane(p);b=load_plane(r['image']);assert np.array_equal(a,b)
            dicom.append(dict(path=str(p),pixel_max_abs_error=float(np.abs(a-b).max())))
dump('package_verification.json',dict(producer='package_check.py',live_feature_checks=live,dicom_checks=dicom,dicom_raw_root=str(raw) if raw else None,backbone_frozen=not any(p.requires_grad for p in net.parameters()),model_eval=not net.training,labels='Validation masks only compare fixed reference scores; all20 artifact is not an inductive test',seconds=time.monotonic()-START))
print('PACKAGE VERIFIED',json.dumps(live),flush=True)
