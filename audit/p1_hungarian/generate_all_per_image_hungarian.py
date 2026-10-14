"""Generate exact per-image Hungarian P1 predictions for all 472 validation slices."""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src2.utils import load_config, setup, resolve_path, new_output, write_json, sha256
from src2.data import FeatureStore
from src2.modules import load_head
from src2.clustering import embed, distances, load_fit
from src2.preprocessing import load_image, inverse_grid
from src2.postprocessing import refine
from src2.metrics import counts, scores

BASELINE = ROOT / "audit/p1_hungarian/n2_baseline"
FIT = ROOT / "audit/p1_hungarian/p1_fit"
OUTPUT = ROOT / "audit/p1_hungarian/per_image_hungarian_all"


def match(native, target):
    valid = native < 8
    confusion = np.bincount(
        (native[valid].astype(np.int16) * 2 + target[valid].astype(np.int16)).ravel(),
        minlength=16,
    ).reshape(8, 2)
    rows, cols = linear_sum_assignment(confusion, maximize=True)
    liver = rows[cols == 1]
    if len(liver) != 1:
        raise ValueError("Expected one liver cluster")
    return int(liver[0]), confusion, rows, cols


def main():
    cfg = load_config(ROOT / "src2/configs/eval_config.yaml")
    setup(cfg)
    store = FeatureStore(cfg)
    ids = store.indices("val")
    if len(ids) != 472:
        raise ValueError(f"Expected 472 validation slices, found {len(ids)}")
    raw = np.load(BASELINE / "raw_clusters.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(resolve_path(cfg, cfg["paths"]["scoring_masks"]),
                     mmap_mode="r", allow_pickle=False)
    model, checkpoint_sha = load_head(cfg)
    centers, fitted = load_fit(FIT, cfg, checkpoint_sha)
    out = new_output(OUTPUT)
    native_raw = np.lib.format.open_memmap(
        out / "raw_clusters_native.npy", mode="w+", dtype=np.uint8,
        shape=(len(ids), 512, 512),
    )
    final = np.lib.format.open_memmap(
        out / "refined_predictions.npy", mode="w+", dtype=np.uint8,
        shape=(len(ids), 512, 512),
    )

    selections = []
    for j, i in enumerate(ids):
        native = inverse_grid(raw[j], store.records[i], mode="nearest", outside=8).astype(np.uint8)
        selected, confusion, rows, cols = match(native, labels[i] == 1)
        native_raw[j] = native
        selections.append({
            "position": j,
            "index": i,
            "sample_id": store.records[i]["sample_id"],
            "patient": store.records[i]["patient"],
            "selected_liver_cluster": selected,
            "confusion_rows_clusters_cols_background_liver": confusion.tolist(),
            "assignment_rows": rows.tolist(),
            "assignment_cols": cols.tolist(),
        })
    native_raw.flush()

    batch_size = cfg["inference"]["batch_size"]
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        codes = embed(model, store.batch(batch))
        for offset, i in enumerate(batch):
            j = start + offset
            selected = selections[j]["selected_liver_cluster"]
            d = distances(codes[offset], centers)
            weights = np.exp(-(d - d.min(1, keepdims=True)) / fitted["temperature"])
            weights /= weights.sum(1, keepdims=True)
            field = weights[:, selected].reshape(28, 28)
            image = load_image(resolve_path(cfg, store.records[i]["image"]), cfg)
            pred = refine(field, image, store.records[i], cfg)
            final[j] = pred
            selections[j]["metrics"] = scores(counts(pred, labels[i] == 1))
    final.flush()

    write_json(out / "selections.json", selections)
    manifest = {
        "method": "overcluster-aware one-to-one 8x2 Hungarian independently for each of 472 validation slices",
        "label_use": "each slice's ground-truth mask selects its liver cluster and scores its final prediction",
        "warning": "per-image label oracle; not label-free inference and not literal upstream C0/C1-only behavior",
        "slices": len(ids),
        "indices": ids,
        "checkpoint_sha256": checkpoint_sha,
        "fit_sha256": sha256(FIT / "fit.json"),
        "outputs": {
            "raw_clusters_native.npy": sha256(out / "raw_clusters_native.npy"),
            "refined_predictions.npy": sha256(out / "refined_predictions.npy"),
            "selections.json": sha256(out / "selections.json"),
        },
    }
    write_json(out / "manifest.json", manifest)
    print(out)


if __name__ == "__main__":
    main()

