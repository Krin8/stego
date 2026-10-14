"""Read-only full-train neighbour parity; no feature extraction or training."""
import json
import time
import numpy as np
from src2.utils import HERE, ROOT, load_config, setup, labels_closed, read_json
from src2.data import FeatureStore
from src2.correspondence import feature_neighbors


def main():
    cfg=load_config(HERE/'configs/train_knn_config.yaml');setup(cfg)
    start=time.monotonic()
    with labels_closed():
        store=FeatureStore(cfg)
        neighbors,meta=feature_neighbors(store,cfg['training']['knn_k'])
        reference=ROOT/'experiments/stego_biomedclip/logs'
        old=np.load(reference/'train_neighbors.npy',allow_pickle=False)
        old_meta=read_json(reference/'train_neighbors.json')
        assert meta['ordered_sample_ids']==old_meta['ordered_sample_ids']
        np.testing.assert_array_equal(neighbors,old)
    result=dict(training_images=len(neighbors),stored_neighbors=neighbors.shape[1],
        uniform_sampling_topk=cfg['training']['num_neighbors'],
        self_matches=int((neighbors==np.arange(len(neighbors))[:,None]).sum()),
        exact_match_to_existing_448_neighbors=True,labels_read=False,
        training_batches=0,feature_extractions=0,neighbor_cache_files_written=0,
        seconds=time.monotonic()-start,
        reference=str(reference/'train_neighbors.npy'),
        ordered_sample_ids_sha256=meta['ordered_sample_ids_sha256'])
    (HERE/'verification/knn_neighbors.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
