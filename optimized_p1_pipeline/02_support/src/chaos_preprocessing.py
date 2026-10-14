"""Approved CHAOS v2 DICOM/PNG contract, shared by all active entrypoints."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

from chaos_splits import load_splits, patient_split

MODALITIES = ('CT', 'T1DUAL', 'T2SPIR')
CT_WINDOW = (-150.0, 250.0)
MRI_PERCENTILES = (1.0, 99.9)
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def preprocessing_contract(ct_window=CT_WINDOW):
    return dict(version=2, modalities=list(MODALITIES), ct_window=list(ct_window),
                mri_percentiles=list(MRI_PERCENTILES), mri_scope='whole slice; stored pixels',
                constant_slice='zeros', orientation='native; no flip or transpose',
                storage='uint16 PNG; round(65535*x)', spatial_resampling='none',
                t1_phase='InPhase', mask_interpolation='nearest')


def normalize_pixels(pixels, modality, slope=1.0, intercept=0.0, ct_window=CT_WINDOW):
    values = np.asarray(pixels, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite DICOM pixels')
    if modality == 'CT':
        values = values * float(slope) + float(intercept)
        low, high = map(float, ct_window)
        if not low < high:
            raise ValueError('CT window must have increasing endpoints')
    elif modality in ('T1DUAL', 'T2SPIR'):
        low, high = np.percentile(values, MRI_PERCENTILES)
        if high <= low:
            return np.zeros_like(values)
    else:
        raise ValueError(f'Excluded/unknown CHAOS modality: {modality}')
    return np.clip((values - low) / (high - low), 0, 1).astype(np.float32)


def dicom_modality(path):
    path = Path(path)
    if 'OutPhase' in path.parts:
        raise ValueError('Out-phase images are excluded from the v2 baseline')
    if 'T1DUAL' in path.parts and 'InPhase' in path.parts:
        return 'T1DUAL'
    if 'T2SPIR' in path.parts:
        return 'T2SPIR'
    if pydicom.dcmread(path, stop_before_pixels=True).Modality == 'CT':
        return 'CT'
    raise ValueError(f'Cannot establish an eligible sequence for {path}')


def read_image(path, modality, ct_window=CT_WINDOW):
    if 'OutPhase' in Path(path).parts:
        raise ValueError('Out-phase images are excluded')
    ds = pydicom.dcmread(path)
    values = normalize_pixels(ds.pixel_array, modality,
                              getattr(ds, 'RescaleSlope', 1), getattr(ds, 'RescaleIntercept', 0), ct_window)
    # Identical bounded quantization for raw and derivative routes (error <= 1/65535).
    return np.rint(values * 65535).astype(np.uint16)


def pair_series(dicom_dir, mask_dir, modality, patient_id):
    """Associate by explicit ID first; never zip independently ordered lists."""
    dicom_dir, mask_dir = Path(dicom_dir), Path(mask_dir)
    if modality not in MODALITIES or 'OutPhase' in dicom_dir.parts:
        raise ValueError('Excluded CHAOS sequence')
    masks = {}
    for path in mask_dir.glob('*.png'):
        match = re.fullmatch(r'liver_GT_(\d+)', path.stem) if modality == 'CT' else None
        if modality == 'CT' and match is None:
            raise ValueError(f'Unknown CT mask ID: {path}')
        key = int(match[1]) if modality == 'CT' else path.stem
        if key in masks:
            raise ValueError(f'Duplicate mask ID: {key}')
        masks[key] = path
    records, used = [], set()
    for path in dicom_dir.glob('*.dcm'):
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        key = int(ds.InstanceNumber) - 1 if modality == 'CT' else path.stem
        if key not in masks or key in used:
            raise ValueError(f'Image needs exactly one unused mask: {path}, ID={key}')
        mask = masks[key]
        with Image.open(mask) as im:
            if im.size != (ds.Columns, ds.Rows):
                raise ValueError(f'Image/mask dimensions differ: {path}, {mask}')
        used.add(key)
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
        position = np.asarray(ds.ImagePositionPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        records.append(dict(dicom=str(path), mask=str(mask), modality=modality,
                            patient_id=str(patient_id), instance=int(ds.InstanceNumber),
                            sop_uid=str(ds.SOPInstanceUID), series_uid=str(ds.SeriesInstanceUID),
                            position=position.tolist(), orientation=orientation.tolist(),
                            pixel_spacing=list(map(float, ds.PixelSpacing)),
                            shape=[int(ds.Rows), int(ds.Columns)], physical_position=float(position @ normal)))
    if not records or used != set(masks):
        raise ValueError(f'Missing images or orphan masks in {dicom_dir} / {mask_dir}')
    if len({r['series_uid'] for r in records}) != 1 or len({r['sop_uid'] for r in records}) != len(records):
        raise ValueError(f'Ambiguous DICOM series/instance IDs in {dicom_dir}')
    if any(not np.allclose(r['orientation'], records[0]['orientation'], atol=1e-5, rtol=0) for r in records):
        raise ValueError(f'Inconsistent slice orientation in {dicom_dir}')
    if len({r['physical_position'] for r in records}) != len(records):
        raise ValueError(f'Duplicate physical slice positions in {dicom_dir}')
    return sorted(records, key=lambda r: r['physical_position'])


def plan_dataset(raw_root, modalities=MODALITIES, split_manifest=None):
    raw_root = Path(raw_root)
    split_manifest = load_splits() if split_manifest is None else split_manifest
    records = []
    for mod in modalities:
        if mod not in MODALITIES:
            raise ValueError(f'Excluded modality {mod}')
        base = raw_root / ('CT' if mod == 'CT' else 'MR')
        patient_ids = sorted((p.name for p in base.iterdir() if p.is_dir() and p.name.isdigit()), key=int)
        expected = set(split_manifest['patients'][mod]['train'] + split_manifest['patients'][mod]['val'])
        if set(patient_ids) != expected:
            raise ValueError(f'{mod} discovered patients differ from frozen split: {set(patient_ids) ^ expected}')
        for pid in patient_ids:
            folder = base / pid if mod == 'CT' else base / pid / mod
            dicom_dir = folder / 'DICOM_anon'
            if mod == 'T1DUAL':
                dicom_dir /= 'InPhase'
            for record in pair_series(dicom_dir, folder / 'Ground', mod, pid):
                record['split'] = patient_split(split_manifest, mod, pid)
                record['sample_id'] = f"{mod}/{pid}/{record['sop_uid']}"
                record['stem'] = f"{mod}_{pid}_{record['instance']:04d}"
                record['dicom'] = str(Path(record['dicom']).relative_to(raw_root))
                record['mask'] = str(Path(record['mask']).relative_to(raw_root))
                records.append(record)
    if len({r['sample_id'] for r in records}) != len(records) or len({r['stem'] for r in records}) != len(records):
        raise ValueError('Duplicate derivative IDs or names')
    return records


def generate_derivatives(raw_root, output_dir, records, ct_window=CT_WINDOW, write=False):
    counts = Counter((r['split'], r['modality']) for r in records)
    print(json.dumps({'dry_run': not write, 'counts': {f'{s}/{m}': n for (s, m), n in sorted(counts.items())},
                      'paired_images': len(records), 'pairing_problems': 0, 'output': str(output_dir)}, sort_keys=True))
    if not write:
        return
    output_dir, raw_root = Path(output_dir), Path(raw_root)
    if output_dir.name != 'preprocessed_v2' or output_dir.exists():
        raise FileExistsError('Generation requires a NEW preprocessed_v2 directory; existing files are never overwritten')
    output_dir.mkdir(parents=True)
    manifest_records = []
    for record in records:
        entry = dict(record)
        image = read_image(raw_root / record['dicom'], record['modality'], ct_window)
        with Image.open(raw_root / record['mask']) as mask_image:
            mask = np.asarray(mask_image.convert('L'))
        allowed = {0, 255} if record['modality'] == 'CT' else {0, 63, 126, 189, 252}
        if not set(np.unique(mask)) <= allowed:
            raise ValueError(f'Unknown mask labels: {record["mask"]}')
        for kind, array in [('images', image), ('labels', mask)]:
            rel = Path(record['split']) / kind / (record['stem'] + '.png')
            dest = output_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as f:
                Image.fromarray(array).save(f, format='PNG')
            entry[kind] = str(rel)
            entry[kind + '_sha256'] = file_sha256(dest)
        manifest_records.append(entry)
    manifest = dict(schema_version=2, preprocessing=preprocessing_contract(ct_window),
                    split_manifest_sha256=digest(load_splits()), samples=manifest_records)
    with (output_dir / 'manifest.json').open('x') as f:
        f.write(canonical_json(manifest) + '\n')


class ChaosTransform:
    """Single resize/crop/normalization contract for raw and uint16 derivatives."""
    def __init__(self, res=224, is_label=False, crop_type='center'):
        if res != 224 or crop_type != 'center':
            raise ValueError('CHAOS v2 baseline requires resolution224 and center crop')
        self.res, self.is_label = res, is_label
        self.contract = dict(resize=224, resize_mode='short edge', crop='center224',
                             image_interpolation='bilinear', antialias=True,
                             mask_interpolation='nearest', mean=list(MEAN), std=list(STD),
                             grayscale_channels=3, normalization_application_count=1)

    def __call__(self, image):
        import torch
        from torchvision.transforms import functional as TF, InterpolationMode
        array = np.array(image, copy=True)
        if array.ndim != 2:
            raise ValueError('CHAOS v2 expects a native grayscale plane')
        if not self.is_label and array.dtype not in (np.dtype('uint16'), np.dtype('int32')):
            raise ValueError('CHAOS v2 images must be normalized uint16 PNG values')
        tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
        if self.is_label:
            tensor = TF.resize(tensor, self.res, interpolation=InterpolationMode.NEAREST)
            return TF.center_crop(tensor, [self.res, self.res]).long()
        tensor = (tensor / 65535.0).repeat(3, 1, 1)
        tensor = TF.resize(tensor, self.res, interpolation=InterpolationMode.BILINEAR, antialias=True)
        tensor = TF.center_crop(tensor, [self.res, self.res])
        return TF.normalize(tensor, MEAN, STD)


def get_chaos_transform(res=224, is_label=False, crop_type='center'):
    return ChaosTransform(res, is_label, crop_type)
