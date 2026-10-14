"""Binary liver scores: fractions internally; no label-assisted cluster mapping."""
import numpy as np


def counts(pred, target):
    p, y = np.asarray(pred, bool), np.asarray(target, bool)
    if p.shape != y.shape:
        raise ValueError('Prediction and annotation geometry differ')
    return np.array([(p & y).sum(), (p & ~y).sum(), (~p & y).sum(), (~p & ~y).sum()], np.int64)


def scores(c):
    tp, fp, fn, tn = map(int, c)
    iou = tp/max(1, tp+fp+fn)
    bg = tn/max(1, tn+fp+fn)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, liver_iou=iou,
        liver_dice=2*tp/max(1, 2*tp+fp+fn), mean_iou=(iou+bg)/2,
        pixel_accuracy=(tp+tn)/max(1, tp+fp+fn+tn))


def aggregate(rows):
    patients = {}
    for row in rows:
        patients.setdefault(row['patient'], []).append(row)
    output = {}
    for patient, items in patients.items():
        c = sum((np.array([r[k] for k in ('tp','fp','fn','tn')]) for r in items), start=np.zeros(4, np.int64))
        output[patient] = dict(slices=len(items), pooled=scores(c),
            mean_slice={k:float(np.mean([r[k] for r in items])) for k in ('liver_iou','liver_dice')})
    summary = {}
    for mode in ('pooled', 'mean_slice'):
        summary[mode] = {}
        for metric in ('liver_iou','liver_dice'):
            a = np.array([v[mode][metric] for v in output.values()])
            summary[mode][metric] = dict(mean=float(a.mean()), std=float(a.std(ddof=1)) if len(a)>1 else 0., min=float(a.min()), max=float(a.max()))
    c = sum((np.array([r[k] for k in ('tp','fp','fn','tn')]) for r in rows), start=np.zeros(4, np.int64))
    return dict(patients=output, patient_summary=summary, globally_pooled=scores(c), slices=rows)
