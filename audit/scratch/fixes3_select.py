"""Select ONLY on predeclared label-free monitors; report labels separately."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'audit/scratch/fixes3'
def select(rows):
    # Only these label-free fields can influence eligibility or ranking.
    free=[dict(run=r['run'],batches=r['batches'],early_stopped=r['early_stopped'],
               **{k:r['monitors'][k] for k in ['code_std','cross_image_cosine','cluster_usage','cluster_entropy']}) for r in rows]
    eligible=[r for r in free if r['batches']==500 and not r['early_stopped'] and r['cross_image_cosine']<=.99 and r['code_std']>=.01 and sum(v>=.01 for v in r['cluster_usage'])>=2]
    if not eligible:return None,free
    best_cos=min(r['cross_image_cosine'] for r in eligible)
    ties=[r for r in eligible if r['cross_image_cosine']<=best_cos+1e-4]
    return max(ties,key=lambda r:r['cluster_entropy'])['run'],free
if '--self-test' in sys.argv:
    base=[dict(run=n,batches=500,early_stopped=False,monitors=dict(code_std=.06,cross_image_cosine=v,cluster_usage=[.5,.5]+[0.]*6,cluster_entropy=.69),cluster_metrics={'IoU':score}) for n,v,score in [('B',.2,0),('C',.3,100)]]
    assert select(base)[0]=='B'
    base[0]['cluster_metrics']['IoU']=100; base[1]['cluster_metrics']['IoU']=0
    assert select(base)[0]=='B'
    base[0]['early_stopped']=True;assert select(base)[0]=='C'
    base[1]['monitors']['cluster_usage']=[1.]+[0.]*7;assert select(base)[0] is None
    print('PASS selection ignores swapped validation scores; rejects early-stop/single-cluster cases');raise SystemExit
paths=[OUT/'fixes3_A_seed0_retry1_result.json',OUT/'fixes3_B_seed0_result.json',OUT/'fixes3_C_seed0_result.json']
rows=[json.loads(p.read_text()) for p in paths]
choice,free=select(rows)
result=dict(selected_noncollapse_candidate=choice,label_free_inputs=free,
            rule='Reach500 without early-stop; cosine<=.99, std>=.01, >=2 clusters at>=1%; lowest cosine, then highest entropy within1e-4 tie',
            label_dependent_reports='Evaluation-only majority map, liver IoU/Dice and GT overlays; reused DIAG linear reference is supervised and has100-slice scope',
            scope='Only4 validation patients; <=500 batches do not establish final quality or generalization')
if choice is not None:
    r=next(r for r in rows if r['run']==choice);s=r['shifts']
    result['full_run_command']='WANDB_MODE=offline python3 -B train_segmentation.py experiment_name=fixes3_full_'+choice+' max_steps=5000 extra_clusters=6 train_linear_probe=False collapse_early_stop=True '+ ' '.join(f'{k}={v!r}' for k,v in s.items())
else:result['full_run_command']=None
(OUT/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
