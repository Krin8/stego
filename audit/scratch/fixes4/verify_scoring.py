"""Reproduce the existing frozen baseline exactly before scoring new runs."""
from common import *
import ast,json
import torch.nn.functional as F
from train_segmentation import LitUnsupervisedSegmenter
from chaos_preprocessing import get_chaos_transform
from data import ContrastiveSegDataset
FORE=ROOT/'audit/scratch/forensics';BASE=ROOT/'audit/scratch/baseline'
source=(BASE/'common.py').read_text();namespace=dict(globals(),OUT=BASE)
exec(compile(source[source.index('# Execute only the original function definitions'):],str(BASE/'common.py'),'exec'),namespace)
pred=np.load(BASE/'k8_c0_s0_assignments.npy',mmap_mode='r');r=namespace['evaluate'](pred,8);old=json.loads((BASE/'k8_c0_s0.json').read_text())
for key in ['majority','optimal','lopo','lopo_summary','patient_confusions']:assert r[key]==old[key],key
records=json.loads((ROOT/'audit/scratch/selectivity/records.json').read_text());assert all('stem' in x for x in records)
result=dict(script=__file__,baseline_mapping_and_lopo_exact_match=True,majority=r['majority'],lopo_summary=r['lopo_summary'],labels='Only existing472 val annotations; no training labels',training_launched=False)
(OUT/'verify_scoring.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
