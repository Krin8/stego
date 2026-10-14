"""Pre-authorized exploratory selection, no seed selection or tuning."""
from pipeline import *
variants=['B0','B1','B2_336','B2_448','B3'];means={}
for variant in variants:
 rows=[json.loads((OUT/f'{variant}_s{s}_scores.json').read_text()) for s in [0,1,2]]
 means[variant]=float(np.mean([r['results']['224']['summary']['optimal']['iou']['mean'] for r in rows]))
parent=max(variants,key=lambda v:means[v])
dump('parent_selection.json',dict(script=__file__,parent=parent,means=means,
 criterion='Highest mean224 LOPO optimal IoU across all3 seeds among B0-B3; ties retain earlier listed variant',
 labels_used='All cohort labels indirectly through LOPO scores. B4/B5 exploratory; no untouched outer selection set.'))
print('Parent',parent,means,flush=True)
