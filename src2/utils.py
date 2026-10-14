"""Configuration, artifact identity, deterministic setup and output safety."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def new_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def resolve_path(cfg, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(cfg['_root']) / path


def load_config(path):
    # JSON is a YAML subset: configs remain human-editable without another parser.
    path = Path(path).resolve()
    cfg = read_json(path)
    if 'base' in cfg:
        base = read_json(path.parent / cfg.pop('base'))
        base.update(cfg)
        cfg = base
    cfg['_root'] = str(ROOT)
    cfg['_config'] = str(path)
    if cfg['seed'] != 0 or cfg['inference']['head'] != 'nonlinear_only_unnormalized':
        raise ValueError('This package declares optimized P1 seed 0 only')
    if cfg['clustering']['k'] != 8 or cfg['preprocessing']['resolution'] != 448:
        raise ValueError('The declared pipeline requires K=8 and 448 input')
    if cfg['naming']['rule'] != 'N2':
        raise ValueError('Only N2 naming is implemented')
    # These geometry choices are implemented literally, not configurable alternatives.
    pre = cfg['preprocessing']
    if pre['body_threshold'] != .05 or pre['body_margin_each_side'] != .05 or pre['native_shape'] != [512,512] or pre['normalization_application_count'] != 1:
        raise ValueError('Unsupported feature preprocessing contract')
    if (pre['grid'],pre['patch_size'],pre['feature_dim'],pre['orientation']) != (28,16,768,'native_no_flip'):
        raise ValueError('Unsupported feature geometry')
    if cfg['clustering']['fit_split'] != 'train':
        raise ValueError('Cluster fitting is restricted to training patients')
    if cfg['naming']['selection'] != 'largest_body_patch_area' or cfg['naming']['no_candidate'] != 'empty_prediction':
        raise ValueError('Unsupported N2 selection/abstention rule')
    post = cfg['postprocessing']
    if post['largest_connectivity'] != 8 or post['hole_fill_background_connectivity'] != 4:
        raise ValueError('Unsupported component connectivity')
    if cfg['postprocessing']['closing_radius'] < 1:
        raise ValueError('Closing radius must be positive')
    return cfg


def contract(cfg):
    return {k: cfg[k] for k in ('seed', 'preprocessing', 'inference', 'clustering', 'naming', 'postprocessing')}


def parser(default_config, description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', type=Path, default=HERE / 'configs' / default_config)
    return p


def setup(cfg):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    import numpy as np
    import torch
    torch.set_num_threads(cfg['threads'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)


_labels_closed = False
_hook_installed = False


@contextlib.contextmanager
def labels_closed():
    """Deny annotation-array/image opens during training, fitting and prediction."""
    global _labels_closed, _hook_installed
    if not _hook_installed:
        def guard(event, args):
            if not _labels_closed or event != 'open' or isinstance(args[0], int):
                return
            name = os.fsdecode(args[0]).replace('\\', '/')
            if any(x in name for x in ('/Ground/', '/labels/', '/masks/', 'scoring_masks')):
                raise PermissionError('Labels are closed in this stage: ' + name)
        sys.addaudithook(guard)
        _hook_installed = True
    previous = _labels_closed
    _labels_closed = True
    try:
        yield
    finally:
        _labels_closed = previous
