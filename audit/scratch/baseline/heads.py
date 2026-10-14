"""Re-score exact FORENSICS full-sample predictions with identical/LOPO mappings."""
from common import *
old=json.loads((FORE/'task1.json').read_text());inv=json.loads((FORE/'inventory.json').read_text());results={}
for name,r in old.items():
 assert Path(inv[name]['path']).is_file()
 pred=np.load(FORE/f'{name}_assignments.npy',mmap_mode='r');assert pred.shape==(472,224,224)
 result=evaluate(pred,8)
 for key in ['confusion','majority','optimal']:assert result[key]==r[key],(name,key)
 result.update(method={'seed0':'C seed0 batch500','seed1':'C seed1 batch500','seed2':'C seed2 batch500','best600':'C full batch600','final':'C full batch5000'}[name],K=8,seed=int(name[-1]) if name.startswith('seed') else 0,script=__file__,prediction_producer=str(FORE/'task1.py'),checkpoint=inv[name]['path'],prediction_sha256=hashlib.sha256((FORE/f'{name}_assignments.npy').read_bytes()).hexdigest(),labels='Global mappings fit+score all472 validation labels; LOPO mapping fit three validation patients, score held-out patient. No fitting model parameters.',elapsed_seconds=elapsed())
 results[name]=result;dump('heads.json',results)
 print(name,result['majority']['iou'],result['optimal']['iou'],result['lopo_summary'],flush=True)
note('Heads: all five global confusion matrices/scores match FORENSICS exactly; LOPO results in `heads.json`, producer `audit/scratch/baseline/heads.py`. Runtime '+str(elapsed())+'s.')
