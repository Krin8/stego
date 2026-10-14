"""Offline, two-thread diagnostic runtime; no annotation access or training."""
import ast
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
START = time.monotonic()
sys.dont_write_bytecode = True
tempfile.tempdir = str(OUT)
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2',
                  OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
READS = set()


def guard(event, args):
    if event == 'open' and not isinstance(args[0], int):
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:]
        write = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if write and not path.is_relative_to(OUT):
            raise PermissionError('Diagnostic writes only under audit/colocated: ' + str(path))
        if not write:
            if any(t in str(path) for t in ['/labels/', '/Ground/', 'scoring_masks', '_labels.npy']):
                raise PermissionError('Labels are closed: ' + str(path))
            READS.add(str(path))
    elif event == 'socket.connect':
        raise PermissionError('No network')


sys.addaudithook(guard)
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(2)
torch.set_grad_enabled(False)
sys.path.insert(0, str(ROOT / 'experiments/stego_biomedclip'))
from reference_loss import ContrastiveCorrelationLoss, sample
from model import objective, StegoHead
sys.path.insert(0, str(ROOT / 'experiments/stego_colocated'))
from loss import ColocatedCorrelationLoss, adjacent_partners


def read(path):
    return json.loads(Path(path).read_text())


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ledger(name):
    dump(name, dict(script=sys.argv[0], seconds=time.monotonic() - START,
                    threads=torch.get_num_threads(), labels_opened=False,
                    opened_read_paths=sorted(READS)))


RR = read(ROOT / 'audit/naming/image_records.json')
TRAIN = np.array([i for i, row in enumerate(RR) if row['split'] == 'train'])
FEATURES = np.load(ROOT / 'audit/improve/B2_448_features.npy', mmap_mode='r', allow_pickle=False)
CFG = read(ROOT / 'experiments/stego_biomedclip/configs/locked.json')
CFG.update(pos_intra_shift=0.794418, pos_inter_shift=0.785981,
           neg_inter_shift=0.797651, threads=2)
assert FEATURES.shape == (2341, 784, 768) and len(TRAIN) == 1869


def feature_batch(global_ids):
    assert all(RR[int(i)]['split'] == 'train' for i in global_ids)
    return torch.from_numpy(np.stack([FEATURES[i] for i in global_ids])).reshape(
        -1, 28, 28, 768).permute(0, 3, 1, 2).contiguous()
