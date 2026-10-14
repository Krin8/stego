"""Frozen patient assignments shared by preprocessing, loaders and kNN input."""
import json
from pathlib import Path

DEFAULT_SPLITS = Path(__file__).resolve().parent / 'configs/chaos_splits_v2.json'


def validate_splits(manifest):
    if manifest.get('schema_version') != 2:
        raise ValueError('Unsupported CHAOS split manifest')
    for modality in ('CT', 'T1DUAL', 'T2SPIR'):
        groups = manifest['patients'][modality]
        train, val = groups['train'], groups['val']
        if len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val):
            raise ValueError(f'Overlapping or duplicate {modality} patients in split manifest')
    if manifest['patients']['T1DUAL'] != manifest['patients']['T2SPIR']:
        raise ValueError('MRI sequences must share patient assignments')
    return manifest


def load_splits(path=DEFAULT_SPLITS):
    return validate_splits(json.loads(Path(path).read_text()))


def patient_split(manifest, modality, patient_id):
    patient_id = str(int(str(patient_id).removeprefix('p')))
    matches = [split for split, ids in manifest['patients'][modality].items() if patient_id in ids]
    if len(matches) != 1:
        raise ValueError(f'Patient {modality}/{patient_id} lacks a unique frozen split')
    return matches[0]
