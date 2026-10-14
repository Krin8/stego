"""Fit optimized P1 K=8 centres and N2 naming on train images only."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.cluster import KMeans
from src2.utils import parser, load_config, setup, resolve_path, new_output, write_json, sha256, contract, labels_closed
from src2.data import FeatureStore
from src2.modules import load_head
from src2.clustering import embed, distances
from src2.preprocessing import read_hu
from src2.naming import accumulate, select


def fit(cfg, output):
    setup(cfg)
    with labels_closed():
        store = FeatureStore(cfg); ids = store.indices('train')
        model, checkpoint_sha = load_head(cfg)
        out = new_output(output)
        write_json(out/'resolved_config.json', cfg)
        # Full train population, no subsampling or label-based selection.
        z = np.lib.format.open_memmap(out/'train_codes.npy', mode='w+', dtype=np.float32, shape=(len(ids)*784, 70))
        size = cfg['inference']['batch_size']
        for start in range(0, len(ids), size):
            batch = ids[start:start+size]
            z[start*784:(start+len(batch))*784] = embed(model, store.batch(batch)).reshape(-1, 70)
        z.flush()
        p = cfg['clustering']
        km = KMeans(n_clusters=8, random_state=0, n_init=p['n_init'], max_iter=p['max_iter'], tol=p['tol'], algorithm=p['algorithm']).fit(z)
        centers = km.cluster_centers_
        np.save(out/'centers.npy', centers, allow_pickle=False)
        d = distances(z, centers)
        two = np.partition(d, 1, axis=1)[:, :2]
        temperature = max(float(np.median(two[:, 1]-two[:, 0])/np.log(4)), 1e-6)
        del d, two
        sums = np.zeros((8, 4), np.float64)
        for j, i in enumerate(ids):
            record = store.records[i]
            assignment = distances(z[j*784:(j+1)*784], centers).argmin(1)
            accumulate(sums, read_hu(resolve_path(cfg, record['dicom'])), record, assignment, cfg)
        named = select(sums, cfg)
        meta = dict(schema_version=1, contract=contract(cfg), checkpoint_sha256=checkpoint_sha,
            centers_sha256=sha256(out/'centers.npy'), feature_cache_sha256=sha256(store.path),
            ordered_sample_ids_sha256=store.order_sha256, temperature=temperature, naming=named,
            train_samples=len(ids), train_patients=sorted({store.records[i]['patient'] for i in ids}, key=int),
            iterations=int(km.n_iter_), inertia=float(km.inertia_), labels_read=False)
        write_json(out/'fit.json', meta)
        print(out/'fit.json')


def main():
    p = parser('fit_config.yaml', __doc__); p.add_argument('--output')
    a = p.parse_args(); cfg = load_config(a.config)
    fit(cfg, a.output or resolve_path(cfg, cfg['output']))


if __name__ == '__main__':
    main()
