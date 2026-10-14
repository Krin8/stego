"""Seal all validation predictions first, then score N2 hard/refined masks."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import numpy as np
from src2.utils import parser, load_config, setup, resolve_path, new_output, write_json, sha256, contract, labels_closed
from src2.data import FeatureStore
from src2.modules import load_head
from src2.clustering import embed, fields, load_fit
from src2.preprocessing import load_image, inverse_grid, resize224
from src2.postprocessing import refine
from src2.metrics import counts, scores, aggregate


def evaluate(cfg, fit_dir, output):
    setup(cfg)
    with labels_closed():
        store = FeatureStore(cfg); ids = store.indices('val')
        if not ids:
            raise ValueError('No validation slices')
        model, checkpoint_sha = load_head(cfg)
        centers, fitted = load_fit(fit_dir, cfg, checkpoint_sha)
        if fitted['ordered_sample_ids_sha256'] != store.order_sha256 or fitted['feature_cache_sha256'] != sha256(store.path):
            raise ValueError('Feature cache differs from fitted artifact')
        out = new_output(output); write_json(out/'resolved_config.json', cfg)
        raw = np.lib.format.open_memmap(out/'raw_clusters.npy', mode='w+', dtype=np.uint8, shape=(len(ids),28,28))
        hard = np.lib.format.open_memmap(out/'N2_hard.npy', mode='w+', dtype=np.uint8, shape=(len(ids),512,512))
        final = np.lib.format.open_memmap(out/'N2_refined.npy', mode='w+', dtype=np.uint8, shape=(len(ids),512,512))
        selected = fitted['naming']['selected_cluster']
        size = cfg['inference']['batch_size']
        for start in range(0, len(ids), size):
            batch = ids[start:start+size]; z = embed(model, store.batch(batch))
            for offset, i in enumerate(batch):
                j = start+offset; r = store.records[i]
                image = load_image(resolve_path(cfg, r['image']), cfg)
                assignment, field = fields(z[offset], centers, fitted['temperature'], selected)
                raw[j] = assignment
                native = inverse_grid(assignment, r, mode='nearest', outside=8)
                hard[j] = native==selected if selected is not None else np.zeros((512,512),bool)
                final[j] = refine(field, image, r, cfg)
        for a in (raw, hard, final):
            a.flush()
        manifest = dict(contract=contract(cfg), fit_sha256=sha256(Path(fit_dir)/'fit.json'),
            checkpoint_sha256=checkpoint_sha, selected_cluster=selected, abstained=selected is None,
            indices=ids, records=[store.records[i] for i in ids], labels_read=False,
            outputs={p.name:sha256(p) for p in (out/'raw_clusters.npy', out/'N2_hard.npy', out/'N2_refined.npy')})
        write_json(out/'prediction_manifest.json', manifest)
    # Annotation access is after all predictions and their hash manifest exist.
    for name, h in manifest['outputs'].items():
        if sha256(out/name) != h:
            raise ValueError('Prediction changed after sealing')
    labels = np.load(resolve_path(cfg, cfg['paths']['scoring_masks']), mmap_mode='r', allow_pickle=False)
    if labels.shape != (len(store.records),512,512):
        raise ValueError('Annotation-array shape mismatch')
    results = {}
    for kind, masks in [('hard',hard),('refined',final)]:
        results[kind] = {}
        for scale in (512,224):
            rows = []
            for j, i in enumerate(ids):
                pred = masks[j]; target = labels[i]==1
                if scale==224:
                    pred, target = resize224(pred), resize224(target)
                rows.append(dict(index=i, sample_id=store.records[i]['sample_id'], patient=store.records[i]['patient'], **scores(counts(pred,target))))
            results[kind][str(scale)] = aggregate(rows)
    write_json(out/'metrics.json', dict(units='fractions', naming='N2; no label-based mapping',
        labels_use='validation scoring only, after prediction seal', selected_cluster=selected,
        empty_union_policy='IoU and Dice zero, matching legacy max(1, denominator)',
        manifest_sha256=sha256(out/'prediction_manifest.json'), results=results))
    print(out/'metrics.json')


def main():
    p = parser('eval_config.yaml', __doc__); p.add_argument('--fit-dir'); p.add_argument('--output')
    a = p.parse_args(); cfg = load_config(a.config)
    evaluate(cfg, a.fit_dir or resolve_path(cfg,cfg['fit_dir']), a.output or resolve_path(cfg,cfg['output']))


if __name__ == '__main__':
    main()
