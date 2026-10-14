"""Identity and validation for CHAOS v2 neighbors; no implicit cache reuse."""
import json
import hashlib
import subprocess
import uuid
from pathlib import Path

import numpy as np

from chaos_preprocessing import (canonical_json, digest, file_sha256,
                                 preprocessing_contract)

WEIGHT_FILE = Path(__file__).resolve().parent / 'BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin'
FEATURE_CODE_FILES = ('modules.py', 'chaos_preprocessing.py', 'data.py', 'precompute_knns.py')


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None  # Descriptive provenance must not require a Git checkout.


def expected_identity(dataset, cfg):
    if cfg.arch != 'biomedclip':
        raise ValueError('CHAOS v2 cache requires the approved BiomedCLIP backbone')
    k = int(getattr(cfg, 'knn_k', 30))
    if not 2 <= k <= len(dataset.sample_ids):
        raise ValueError('Cache k must include self and at least one other sample, without exceeding N')
    if len(set(dataset.sample_ids)) != len(dataset.sample_ids):
        raise ValueError('Duplicate ordered sample IDs')
    return dict(schema_version=3, split=dataset.image_set,
                ordered_sample_ids=list(dataset.sample_ids), ordered_sample_ids_sha256=digest(dataset.sample_ids),
                ordered_image_sha256=[file_sha256(s[0]) for s in dataset.samples],
                split_manifest_sha256=digest(dataset.split_manifest),
                feature_code_sha256={name: file_sha256(Path(__file__).resolve().parent / name)
                                     for name in FEATURE_CODE_FILES},
                preprocessing=dict(preprocessing_contract(), modality=dataset.modality,
                                   **dataset.transform.contract),
                backbone=dict(weight_sha256=file_sha256(WEIGHT_FILE),
                              config_sha256=file_sha256(WEIGHT_FILE.with_name('open_clip_config.json')),
                              layer=11, norm=True, cls_excluded=True, eval=True,
                              dropout_active=False, feature_dim=768, dtype='float32',
                              pooling='spatial mean then L2; cosine similarity'),
                k=k, self_exclusion_rule='column0 is self; remaining ranks unique and non-self; ties by index')


def cache_path(root, identity):
    return Path(root) / 'nns' / f"nns_chaos_v2_{identity['split']}_{digest(identity)}.npz"


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def validate_indices(array, identity):
    n, k = len(identity['ordered_sample_ids']), identity['k']
    if array.dtype != np.int64 or array.shape != (n, k):
        raise ValueError('KNN cache dtype/shape mismatch')
    if (array < 0).any() or (array >= n).any():
        raise ValueError('KNN cache index outside dataset')
    if not np.array_equal(array[:, 0], np.arange(n)):
        raise ValueError('KNN cache self slot mismatch')
    if any(len(set(row.tolist())) != k for row in array):
        raise ValueError('KNN cache contains duplicate/self positive neighbors')


def cache_metadata(identity, array):
    return dict(identity=identity, identity_sha256=digest(identity),
                provenance=dict(git_commit=git_commit()),
                array_sha256=array_digest(array), array_shape=list(array.shape), array_dtype=str(array.dtype))


def read_cache(path, expected):
    with np.load(path, allow_pickle=False) as cache:
        if 'metadata' not in cache:
            raise ValueError('KNN cache missing metadata: verify backbone, preprocessing and ordered samples before regeneration')
        encoded = cache['metadata']
        if encoded.dtype != np.uint8 or encoded.ndim != 1:
            raise ValueError('Cache metadata must be plain UTF-8 JSON bytes')
        metadata = json.loads(encoded.tobytes().decode('utf-8'))
        if metadata.get('identity') != expected or metadata.get('identity_sha256') != digest(expected):
            raise ValueError('KNN cache identity mismatch: samples/preprocessing/backbone/split/feature code')
        array = cache['nns']
        if (metadata.get('array_shape') != list(array.shape) or metadata.get('array_dtype') != str(array.dtype)
                or metadata.get('array_sha256') != array_digest(array)):
            raise ValueError('KNN cache array digest/shape mismatch')
        validate_indices(array, expected)
        return array.copy()


def prepare_target(path, archive_existing=False):
    path = Path(path)
    if path.exists():
        if not archive_existing:
            raise FileExistsError(f'{path} already exists; use archive_existing_cache=true to move it aside explicitly')
        archived = path.with_name(path.name + '.old-' + uuid.uuid4().hex)
        path.rename(archived)
        print(f'Archived existing cache: {archived}')


def write_cache(path, identity, array):
    validate_indices(array, identity)
    metadata = np.frombuffer(canonical_json(cache_metadata(identity, array)).encode('utf-8'), dtype=np.uint8)
    # Exclusive creation also prevents a race from overwriting an existing target.
    with Path(path).open('xb') as f:
        np.savez_compressed(f, nns=array, metadata=metadata)
