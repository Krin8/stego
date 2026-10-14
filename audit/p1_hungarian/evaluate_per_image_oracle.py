"""Per-image label oracle for the fixed P1 K=8 predictions."""
from collections import Counter
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src2.utils import load_config, setup, resolve_path, new_output, write_json, sha256
from src2.data import FeatureStore
from src2.modules import load_head
from src2.clustering import embed, distances, load_fit
from src2.preprocessing import load_image, inverse_grid, resize224
from src2.postprocessing import refine
from src2.metrics import counts, scores, aggregate


FIT = ROOT / "audit/p1_hungarian/p1_fit"
BASELINE = ROOT / "audit/p1_hungarian/n2_baseline"
OUTPUT = ROOT / "audit/p1_hungarian/per_image_oracle"


def optimal_subset(assignment, target):
    valid = assignment < 8
    confusion = np.bincount(
        (assignment[valid].astype(np.int16) * 2 + target[valid]).ravel(),
        minlength=16,
    ).reshape(8, 2)
    liver_total = int(target.sum())
    best = (-1.0, 0, ())
    for bits in range(1 << 8):
        subset = tuple(i for i in range(8) if bits & (1 << i))
        tp = int(confusion[list(subset), 1].sum()) if subset else 0
        fp = int(confusion[list(subset), 0].sum()) if subset else 0
        fn = liver_total - tp
        iou = tp / max(tp + fp + fn, 1)
        candidate = (iou, -len(subset), tuple(-i for i in subset))
        if candidate > best:
            best = candidate
            chosen = subset
    return chosen, confusion


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

    selections = []
    native_assignments = []
    for j, i in enumerate(ids):
        native = inverse_grid(raw[j], store.records[i], mode="nearest", outside=8).astype(np.int16)
        subset, confusion = optimal_subset(native, labels[i] == 1)
        selections.append({
            "index": i,
            "sample_id": store.records[i]["sample_id"],
            "patient": store.records[i]["patient"],
            "subset": list(subset),
            "confusion": confusion.tolist(),
        })
        native_assignments.append(native)

    model, checkpoint_sha = load_head(cfg)
    centers, fitted = load_fit(FIT, cfg, checkpoint_sha)
    out = new_output(OUTPUT)
    hard = np.lib.format.open_memmap(out / "hard.npy", mode="w+", dtype=np.uint8,
                                    shape=(len(ids), 512, 512))
    refined = np.lib.format.open_memmap(out / "refined.npy", mode="w+", dtype=np.uint8,
                                       shape=(len(ids), 512, 512))
    batch_size = cfg["inference"]["batch_size"]
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        codes = embed(model, store.batch(batch))
        for offset, i in enumerate(batch):
            j = start + offset
            subset = selections[j]["subset"]
            native = native_assignments[j]
            hard[j] = np.isin(native, subset)
            d = distances(codes[offset], centers)
            weights = np.exp(-(d - d.min(1, keepdims=True)) / fitted["temperature"])
            weights /= weights.sum(1, keepdims=True)
            field = (weights[:, subset].sum(1).reshape(28, 28)
                     if subset else np.zeros((28, 28), np.float32))
            image = load_image(resolve_path(cfg, store.records[i]["image"]), cfg)
            refined[j] = refine(field, image, store.records[i], cfg)
    hard.flush()
    refined.flush()

    frequency = Counter(tuple(x["subset"]) for x in selections)
    manifest = {
        "method": "per-image exhaustive hard-IoU-optimal subset over all 2^8 cluster subsets",
        "label_use": "each validation slice's ground-truth mask chooses that slice's prediction; oracle only",
        "tie_break": "higher hard IoU, then fewer clusters, then lower cluster IDs",
        "refinement": "same selected subset feeds unchanged soft F1 refinement; subset is not re-optimized for refined IoU",
        "slices": len(ids),
        "subset_frequency": {str(list(k)): v for k, v in sorted(frequency.items())},
        "mean_clusters_per_slice": float(np.mean([len(x["subset"]) for x in selections])),
        "multi_cluster_slices": int(sum(len(x["subset"]) > 1 for x in selections)),
        "c2_only_slices": int(sum(x["subset"] == [2] for x in selections)),
        "checkpoint_sha256": checkpoint_sha,
        "fit_sha256": sha256(FIT / "fit.json"),
        "outputs": {
            "hard.npy": sha256(out / "hard.npy"),
            "refined.npy": sha256(out / "refined.npy"),
        },
    }
    write_json(out / "selections.json", selections)
    write_json(out / "manifest.json", manifest)
    write_json(out / "metrics.json", {
        "units": "fractions",
        "manifest": manifest,
        "results": {
            "hard": evaluate(hard, labels, ids, store),
            "refined": evaluate(refined, labels, ids, store),
        },
    })
    print(out / "metrics.json")


if __name__ == "__main__":
    main()
