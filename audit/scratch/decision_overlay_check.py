"""Check physical mask geometry and tissue means on already sampled MRI paths."""
import decision_guard as g
import json,numpy as np,pydicom
from PIL import Image
report=g.REPORT.read_text();section=report.split('### Sample physical geometry evidence',1)[1]
geometry=json.loads(section.split('```json\n',1)[1].split('\n```',1)[0])
rows=[]; phase_checks=[]
for sequence in ['T1-in','T1-out','T2']:
    subset=[r for r in geometry if r['sequence']==sequence]
    r=subset[0];ds=pydicom.dcmread(g.ROOT/r['image']);a=ds.pixel_array.astype(float);mask=np.array(Image.open(g.ROOT/r['mask']).convert('L'))
    iop=np.array(ds.ImageOrientationPatient,float);origin=np.array(ds.ImagePositionPatient,float);spacing=np.array(ds.PixelSpacing,float)
    centroids={}
    for label in np.unique(mask):
        if not label:continue
        rr,cc=np.nonzero(mask==label);pos=origin+iop[:3]*cc.mean()*spacing[1]+iop[3:]*rr.mean()*spacing[0]
        centroids[str(label)]=pos.tolist()
    rows.append(dict(sequence=sequence,image=r['image'],mask=r['mask'],inside_mean=float(a[mask>0].mean()),outside_mean=float(a[mask==0].mean()),foreground_nonzero_image_fraction=float((a[mask>0]>0).mean()),organ_centroid_patient_xyz_mm=centroids))
    if sequence=='T1-out':
        for r in subset:
            out=g.ROOT/r['image'];ind=out.parent.parent/'InPhase'/(__import__('pathlib').Path(r['mask']).stem+'.dcm')
            h=pydicom.dcmread(out,stop_before_pixels=True);ih=pydicom.dcmread(ind,stop_before_pixels=True)
            ok=h.Rows==ih.Rows and h.Columns==ih.Columns and np.allclose(np.array(h.PixelSpacing,float),np.array(ih.PixelSpacing,float),atol=1e-6,rtol=0)
            phase_checks.append(dict(out=str(out.relative_to(g.ROOT)),in_phase=str(ind.relative_to(g.ROOT)),same_matrix_and_pixel_spacing=bool(ok)))
result=dict(overlays=rows,out_phase_pixel_geometry=phase_checks)
g.append('## Additional overlay and phase-geometry check\n\nCommand: `python3 -B audit/scratch/decision_overlay_check.py`; reuses selected slices. Centroids use DICOM row/column direction cosines and pixel spacing. Positive patient-X is left in the DICOM patient coordinate convention; kidney relative lateral order is corroboration, not a complete anatomical audit.\n\n```json\n'+json.dumps(result,indent=2)+'\n```')
print(json.dumps(result))
