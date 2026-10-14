"""Summarize completed sealed evaluation, without choosing a seed or tuning."""
import os
os.environ['STEGO_STAGE']='evaluation'
from runtime import *
import csv

r=read(HERE/'metrics/results.json');s=read(HERE/'metrics/across_seeds.json')
lines=['# Completed STEGO transfer experiment','',
       'All runs: frozen BiomedCLIP768×28×28 → STEGO70×28×28, 698,252 trainable parameters. '
       'Three original correspondence terms, no geometric alignment, Adam.0005, batch16,5000 batches per seed.',
       '', 'No CHAOS masks in training, checkpoint selection, naming or inference. '
       'Read logs/*_access.json and LABEL_USAGE.md. Evaluation followed the sealed predictions. '
       'The cohort informed historical choices; this is not untouched testing.', '',
       '## Predeclared N3 results (224, mean of per-patient pooled scores)','',
       '| Seed | Selected N3 cluster | Best train-loss batch | Hard IoU | Hard Dice | F1 IoU | F1 Dice |',
       '|---|---|---|---|---|---|---|']
for seed in range(3):
    run=r['results'][f'seed{seed}'];t=read(HERE/f'logs/seed{seed}_complete.json')
    v=[100*run['rules']['N3'][kind]['224']['summary']['pooled'][metric]['mean']
       for kind in ['hard','F1'] for metric in ['iou','dice']]
    lines.append(f"|{seed}|{run['selected']['N3']}|{t['best_step']}|"+'|'.join(f'{x:.4f}' for x in v)+'|')
for kind in ['hard','F1']:
    lines.append(f"\n{kind} seed mean ± sample SD: "+'; '.join(
        f"{metric} {100*s['N3'][kind][metric]['mean']:.4f} ± {100*s['N3'][kind][metric]['std']:.4f}%"
        for metric in ['iou','dice']))
lines+=['','All N1/N2/N3 seed results and matched baseline: metrics/comparison.csv. '
        'Patient pooled/mean-slice results at224/native512: metrics/results.json. '
        'Seed stability: inter-seed binary Dice, raw-grid ARI/NMI and linear CKA in results.json.',
        '', '## Collapse monitoring','']
for seed in range(3):
    history=[json.loads(x) for x in (HERE/f'logs/seed{seed}_train.jsonl').read_text().splitlines()]
    checks=[x for x in history if 'monitor_loss' in x]
    lines.append(f"Seed{seed}: final {checks[-1]}; collapse flags at batches "+
                 str([x['batch'] for x in checks if x['collapse_detected']])+'.')
lines+=['','## Spatial limitation','',
        'Raw learned grid is still28×28. There is no learned high-resolution decoder. '
        'Native512 masks use the same inverse crop and F1 as baseline. Quantitative scores alone do not establish '
        'less blocky boundaries; inspect the actual arrays/figures.', '',
        'Primary figure (seed0/N3 fixed in advance): visualizations/actual_predictions_five_columns.jpeg '
        '(also PNG/PDF). Every seed/rule has its own figure. Predictions: predictions/seed{0,1,2}/ '
        'raw_clusters.npy, raw_clusters_native.npy, dense_features.npy, ct_native.npy, '
        'N{1,2,3}_hard.npy, N{1,2,3}_F1.npy. No masks are reconstructed from images/PDFs.',
        '', 'See FIDELITY.md for all departures from the pinned original, particularly normalized BiomedCLIP '
        'input, CT crop, posthoc kmeans, label-free naming and F1 in place of online probe/Hungarian/CRF.']
(HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('COMPLETED REPORT',HERE/'RESULTS.md')
