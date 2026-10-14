from common import *
import matplotlib.pyplot as plt
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2
root=ROOT/'logs/chaos/chaos_fixes3_full_C_date_Sep30_19-47-58/lightning_logs/version_0'
rows=[json.loads(s) for s in (root/'validation_diagnostics.jsonl').read_text().splitlines()]
wp=ROOT/'logs/wandb/offline-run-20260930_194801-liqimwvz/run-liqimwvz.wandb';import shutil
scan_copy=OUT/'wandb_scan_copy.wandb';shutil.copyfile(wp,scan_copy)
ds=DataStore();ds.open_for_scan(str(scan_copy));hist=[]
while True:
 blob=ds.scan_data()
 if blob is None:break
 rec=wandb_internal_pb2.Record();rec.ParseFromString(blob)
 if rec.WhichOneof('record_type')=='history':
  vals={i.key or '/'.join(i.nested_key):json.loads(i.value_json) for i in rec.history.item}
  if 'test/cluster/liver_IoU' in vals:hist.append({k:v for k,v in vals.items() if k.startswith(('monitor/','test/cluster/')) or k=='trainer/global_step'})
assert len(hist)==len(rows)==85
for r in rows:
 h=next(x for x in hist if x['trainer/global_step']==r['batches']-1)
 assert abs(h['test/cluster/liver_IoU']-r['metrics']['test/cluster/liver_IoU'])<1e-5
log=(ROOT/'audit/run_records/fixes3_full_C/train.log').read_text();assert '`max_steps=15000` reached' in log
expected=[e*117+k for e in range(43) for k in [50,100] if e*117+k<=5000]
assert expected==[r['batches'] for r in rows]
items=[]
for r in rows:
 items.append(dict(batch=r['batches'],optimizer_step=r['optimizer_steps'],liver_iou=r['metrics']['test/cluster/liver_IoU'],liver_dice=r['metrics']['test/cluster/liver_Dice'],cluster_miou=r['metrics']['test/cluster/mIoU'],code_std=r['code_std'],cosine=r['cross_image_cosine'],entropy=r['cluster_entropy'],liver_clusters=[i for i,x in enumerate(r['cluster_to_class']) if x==1]))
fig,axs=plt.subplots(7,1,figsize=(11,16),sharex=True)
for ax,(key,label) in zip(axs,[('liver_iou','Liver IoU (%)'),('liver_dice','Liver Dice (%)'),('cluster_miou','Cluster mIoU (%)'),('code_std','Code std (unit-normalized)'),('cosine','Cross-image cosine'),('entropy','Cluster entropy (nats)'),('liver_clusters','Clusters mapped to liver (count)')]):
 y=[len(x[key]) if key=='liver_clusters' else x[key] for x in items];ax.plot([x['batch'] for x in items],y,lw=1.5);ax.set_ylabel(label);ax.axvline(635,color='red',ls=':',alpha=.5);ax.grid(alpha=.2)
axs[0].set_title('Full C: all85 logged validations; red line = first loss of liver mapping at635')
axs[-1].set_xlabel('Completed training batches (optimizer steps = 3 × batches)');fig.tight_layout();fig.savefig(OUT/'full_run_trajectory.png',dpi=150);plt.close(fig)
result=dict(script=__file__,measurements=items,wandb_history=hist,validation_count=len(rows),batches_per_epoch=117,expected_schedule=expected,batch_gaps=sorted(set(np.diff(expected).tolist())),first_initial_absence=50,first_presence=100,last_presence=568,first_disappearance_after_presence=635,reappears=any(x['liver_clusters'] for x in items if x['batch']>=635),exact_transition='UNVERIFIED within (568,635]; only discrete validation observations',elapsed_seconds=elapsed())
dump('task3.json',result);note('Task3: all85 JSONL and W&B validations cross-checked. First disappearance after liver mapping existed:635; last positive568; no reappearance. Actual intervals50/67 batches across117-batch epochs. Plot`audit/scratch/forensics/full_run_trajectory.png`; producer`task3.py`.');print('DONE',elapsed())
