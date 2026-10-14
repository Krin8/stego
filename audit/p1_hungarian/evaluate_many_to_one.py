"""Evaluate predeclared many-to-one and disconnected-component variants for P1."""
from pathlib import Path
import json
import sys

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src2.utils import load_config, setup, resolve_path, new_output, write_json, sha256
from src2.data import FeatureStore
from src2.modules import load_head
from src2.clustering import embed, distances, load_fit
from src2.preprocessing import load_image, inverse_grid, resize224
from src2.postprocessing import guided
from src2.metrics import counts, scores, aggregate


FIT = ROOT / "audit/p1_hungarian/p1_fit"
BASELINE = ROOT / "audit/p1_hungarian/n2_baseline"
OUTPUT = ROOT / "audit/p1_hungarian/many_to_one_evaluation"


def best_subset(confusion):
    """Exhaustively maximize pooled validation liver IoU over all 2^8 subsets."""
    liver_total = confusion[:, 1].sum()
    candidates = []
    for bits in range(1 << len(confusion)):
        subset = [i for i in range(len(confusion)) if bits & (1 << i)]
        tp = confusion[subset, 1].sum() if subset else 0
        fp = confusion[subset, 0].sum() if subset else 0
        fn = liver_total - tp
        candidates.append((tp / max(tp + fp + fn, 1), subset))
    return max(candidates, key=lambda x: (x[0], -len(x[1])))[1]


def keep_components(mask, number):
    if number is None:
        return mask
    labels, n = ndi.label(mask, structure=np.ones((3, 3), int))
    if n <= number:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    chosen = np.argpartition(sizes, -number)[-number:]
    return np.isin(labels, chosen)


def refine_variant(field, image, record, cfg, components):
    p = cfg["postprocessing"]
    soft = inverse_grid(field, record)
    filtered = guided(image, soft, p["guided_size"], p["guided_epsilon"])
    pred = keep_components(filtered >= p["threshold"], components)
    pred = ndi.binary_fill_holes(pred)
    rad = p["closing_radius"]
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    pred = ndi.binary_closing(
        np.pad(pred, rad), structure=xx * xx + yy * yy <= rad ** 2
    )[rad:-rad, rad:-rad]
    return pred


def evaluate(masks, labels, ids, store):
    result = {}
    for scale in (512, 224):
        rows = []
        for j, i in enumerate(ids):
            pred, target = masks[j], labels[i] == 1
            if scale == 224:
                pred, target = resize224(pred), resize224(target)
            rows.append(dict(index=i, sample_id=store.records[i]["sample_id"],
                             patient=store.records[i]["patient"],
                             **scores(counts(pred, target))))
        result[str(scale)] = aggregate(rows)
    return result


def main():
    cfg = load_config(ROOT / "src2/configs/eval_config.yaml")
    setup(cfg)
    store = FeatureStore(cfg)
    ids = store.indices("val")
    raw = np.load(BASELINE / "raw_clusters.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(resolve_path(cfg, cfg["paths"]["scoring_masks"]),
                     mmap_mode="r", allow_pickle=False)

    confusion = np.zeros((8, 2), np.int64)
    for j, i in enumerate(ids):
        native = inverse_grid(raw[j], store.records[i], mode="nearest", outside=8).astype(np.int16)
        target = labels[i] == 1
        valid = native < 8
        confusion += np.bincount(
            (native[valid] * 2 + target[valid]).ravel(), minlength=16
        ).reshape(8, 2)
    subset = best_subset(confusion)

    model, checkpoint_sha = load_head(cfg)
    centers, fitted = load_fit(FIT, cfg, checkpoint_sha)
    out = new_output(OUTPUT)
    variants = {
        "largest_2_components": 2,
        "all_components": None,
    }
    predictions = {
        name: np.lib.format.open_memmap(
            out / f"{name}.npy", mode="w+", dtype=np.uint8,
            shape=(len(ids), 512, 512)
        ) for name in variants
    }

    batch_size = cfg["inference"]["batch_size"]
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        codes = embed(model, store.batch(batch))
        for offset, i in enumerate(batch):
            j = start + offset
            record = store.records[i]
            image = load_image(resolve_path(cfg, record["image"]), cfg)
            d = distances(codes[offset], centers)
            weights = np.exp(-(d - d.min(1, keepdims=True)) / fitted["temperature"])
            weights /= weights.sum(1, keepdims=True)
            field = weights[:, subset].sum(1).reshape(28, 28)
            for name, component_count in variants.items():
                predictions[name][j] = refine_variant(
                    field, image, record, cfg, component_count
                )
    for masks in predictions.values():
        masks.flush()

    n2 = np.load(BASELINE / "N2_refined.npy", mmap_mode="r", allow_pickle=False)
    results = {"largest_1_component_current_F1": evaluate(n2, labels, ids, store)}
    results.update({name: evaluate(masks, labels, ids, store)
                    for name, masks in predictions.items()})
    manifest = {
        "method": "validation-label-assisted exhaustive IoU-optimal subset of 8 clusters",
        "label_use": "all 472 validation labels choose the subset and score all variants; optimistic diagnostic",
        "subset": subset,
        "n2_subset": [fitted["naming"]["selected_cluster"]],
        "confusion_rows_clusters_cols_background_liver": confusion.tolist(),
        "postprocessing": {
            "unchanged": ["soft cluster probability sum", "inverse crop", "guided filter size 25 epsilon 0.01", "threshold 0.30", "hole filling", "closing disk radius 3"],
            "varied_only": ["retain largest 1 component", "retain largest 2 components", "retain all components"],
        },
        "checkpoint_sha256": checkpoint_sha,
        "fit_sha256": sha256(FIT / "fit.json"),
        "outputs": {f"{name}.npy": sha256(out / f"{name}.npy") for name in variants},
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "metrics.json", {"units": "fractions", "manifest": manifest,
                                      "results": results})
    print(out / "metrics.json")


if __name__ == "__main__":
    main()
