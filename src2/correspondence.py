"""Select P1 adjacency or STEGO image-kNN with full cross-image patch loss."""
import hashlib
import numpy as np
import torch
from .data import train_geometry
from .loss import ColocatedCorrelationLoss, adjacent_partners
from .reference_loss import ContrastiveCorrelationLoss
from .utils import digest


def feature_neighbors(store, k):
    """Train-only cosine neighbours; same pooling/ranking as the sealed 448 run."""
    ids = store.indices('train')
    if not 1 <= k < len(ids):
        raise ValueError('k must be positive and smaller than the training population')
    pooled = []
    for i in ids:
        if store.records[i]['split'] != 'train':
            raise ValueError('Only training images may enter neighbour selection')
        row = np.array(store.array[i])
        if not np.isfinite(row).all() or not np.allclose(np.linalg.norm(row, axis=-1), 1, atol=2e-4):
            raise ValueError('Expected finite L2-normalized patch tokens')
        pooled.append(row.mean(0))
    pooled = np.stack(pooled)
    lengths = np.linalg.norm(pooled, axis=-1, keepdims=True)
    if np.any(lengths <= 1e-12):
        raise ValueError('Cannot rank zero-length pooled image features')
    pooled /= lengths
    similarity = pooled @ pooled.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argsort(-similarity, axis=1, kind='stable')[:, :k]
    if np.any(neighbors == np.arange(len(ids))[:, None]):
        raise ValueError('Self neighbour was not excluded')
    metadata = dict(
        method='L2(mean normalized patch tokens), cosine, stable sort, self excluded',
        index_space='training rows in ordered_sample_ids; not global cache rows',
        ordered_sample_ids=[store.records[i]['sample_id'] for i in ids],
        k=k, labels_read=False,
        pooled_features_sha256=hashlib.sha256(pooled.tobytes()).hexdigest())
    metadata['ordered_sample_ids_sha256'] = digest(metadata['ordered_sample_ids'])
    return neighbors, metadata


def prepare_correspondence(cfg, store, training):
    mode = getattr(training, 'partner_mode', 'colocated')
    if mode == 'colocated':
        geometry = train_geometry(cfg, store)
        criterion = ColocatedCorrelationLoss(training, torch.ones(training.batch_size, dtype=torch.bool))

        def partners(selected, rng):
            positive, valid, _ = adjacent_partners(geometry, selected, rng)
            return positive, torch.tensor(valid)

        return criterion, partners, dict(partner_mode=mode), None
    if mode != 'feature_knn':
        raise ValueError('partner_mode must be colocated or feature_knn')
    if not 1 <= training.num_neighbors <= training.knn_k:
        raise ValueError('num_neighbors must be in [1, knn_k]')
    neighbors, metadata = feature_neighbors(store, training.knn_k)

    def partners(selected, rng):
        return [int(neighbors[int(i), rng.integers(training.num_neighbors)]) for i in selected], None

    metadata.update(partner_mode=mode, sampling_topk=training.num_neighbors,
                    sampling='uniform among top neighbours, per anchor',
                    patch_pairs='independently sampled coordinates; full cross-image correlation')
    # Original helper/forward: independent coordinates and ALL sampled patch pairs.
    # Do not use the P1 diagonal or presume that an image neighbour is all-positive.
    return ContrastiveCorrelationLoss(training), partners, metadata, neighbors
