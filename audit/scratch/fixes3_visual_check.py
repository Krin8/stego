from fixes3_common import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--fixed',action='store_true');args=p.parse_args()
s=(ROOT/'src/train_segmentation.py').read_text()
if not args.fixed:
 assert 'random.randint(0, len(self.validation_step_outputs) - 1)' in s
 print('REPRODUCED: validation plots choose a random batch');raise SystemExit
from validation_visuals import fixed_validation_indices,plot_fixed_slices
from data import CHAOS
from chaos_preprocessing import get_chaos_transform
from utils import UnsupervisedMetrics
from pathlib import Path
import matplotlib.pyplot as plt
rows=CHAOS(str(ROOT/'src/pytorch_data_dir'),'CT','val',get_chaos_transform(),get_chaos_transform(is_label=True),2,True)
ix=fixed_validation_indices(rows.records);assert len(ix)==4 and len({rows.records[i]['patient_id'] for i in ix})==4
assert ix==fixed_validation_indices(rows.records) and 'random.randint(0, len(self.validation_step_outputs) - 1)' not in s
lookup=torch.tensor([0,0,1,0,1,0,0,0]);outputs=[]
for i in ix:
 x,y,_=rows[i];outputs.append(dict(ind=i,sample_id=rows.records[i]['stem'],img=x,label=y,cluster_preds=torch.arange(224*224).reshape(224,224)%8))
fig=plot_fixed_slices(outputs,lookup);fig.savefig(OUT/'fixed_selection_synthetic.png');plt.close(fig)
print('PASS fixed label-blind indices and distinct patients:',ix,[rows.records[i]['stem'] for i in ix])
print('PASS four-panel types including raw K8 colors; synthetic display only')
