"""Evaluate optimized P1 with upstream STEGO's literal Hungarian behavior."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src2.utils import load_config, setup, resolve_path, new_output, write_json, sha256
from src2.data import FeatureStore
from src2.modules import load_head
from src2.clustering import embed, fields, load_fit
from src2.preprocessing import load_image, inverse_grid, resize224
from src2.postprocessing import refine
from src2.metrics import counts, scores, aggregate


BASELINE = ROOT / "audit/p1_hungarian/n2_baseline"
FIT = ROOT / "audit/p1_hungarian/p1_fit"
OUTPUT = ROOT / "audit/p1_hungarian/literal_stego_evaluation"


def main():
    cfg = load_config(ROOT / "src2/configs/eval_config.yaml")
    setup(cfg)
    store = FeatureStore(cfg)
    ids = store.indices("val")
    baseline_manifest = json.loads((BASELINE / "prediction_manifest.json").read_text())
    if ids != baseline_manifest["indices"]:
        raise ValueError("Validation order differs from sealed raw predictions")
    raw = np.load(BASELINE / "raw_clusters.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(resolve_path(cfg, cfg["paths"]["scoring_masks"]), mmap_mode="r", allow_pickle=False)

    # Literal upstream STEGO update(): despite allocating rows for overclusters,
    # its validity mask admits only prediction IDs below n_classes (2 here).
    stats = np.zeros((8, 2), np.int64)
    for j, i in enumerate(ids):
        native = inverse_grid(raw[j], store.records[i], mode="nearest", outside=8).astype(np.int16)
        target = (labels[i] == 1).astype(np.int16)
        valid = native < 2
        stats += np.bincount((native[valid] * 2 + target[valid]).ravel(), minlength=16).reshape(8, 2)
    rows, cols = linear_sum_assignment(stats, maximize=True)
    # Literal upstream map_clusters(extra_clusters>0): unmatched clusters map -1.
    lookup = np.full(8, -1, np.int8)
    lookup[rows] = cols
    liver_rows = rows[cols == 1]
    if len(liver_rows) != 1:
        raise ValueError("Expected exactly one liver cluster from one-to-one Hungarian matching")
    selected = int(liver_rows[0])

    model, checkpoint_sha = load_head(cfg)
    centers, fitted = load_fit(FIT, cfg, checkpoint_sha)
    out = new_output(OUTPUT)
    hard = np.lib.format.open_memmap(out / "hungarian_hard.npy", mode="w+", dtype=np.uint8,
                                    shape=(len(ids), 512, 512))
    refined = np.lib.format.open_memmap(out / "hungarian_refined.npy", mode="w+", dtype=np.uint8,
                                       shape=(len(ids), 512, 512))
    batch_size = cfg["inference"]["batch_size"]
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        codes = embed(model, store.batch(batch))
        for offset, i in enumerate(batch):
            j = start + offset
            record = store.records[i]
            image = load_image(resolve_path(cfg, record["image"]), cfg)
            assignment, field = fields(codes[offset], centers, fitted["temperature"], selected)
            native = inverse_grid(assignment, record, mode="nearest", outside=8).astype(np.int16)
            valid = native < 8
            mapped = np.full((512, 512), -1, np.int8)
            mapped[valid] = lookup[native[valid]]
            hard[j] = mapped == 1
            refined[j] = refine(field, image, record, cfg)
    hard.flush()
    refined.flush()

    results = {}
    for kind, masks in (("hard", hard), ("refined", refined)):
        results[kind] = {}
        for scale in (512, 224):
            score_rows = []
            for j, i in enumerate(ids):
                pred, target = masks[j], labels[i] == 1
                if scale == 224:
                    pred, target = resize224(pred), resize224(target)
                score_rows.append(dict(index=i, sample_id=store.records[i]["sample_id"],
                                       patient=store.records[i]["patient"],
                                       **scores(counts(pred, target))))
            results[kind][str(scale)] = aggregate(score_rows)

    n2_hard = np.load(BASELINE / "N2_hard.npy", mmap_mode="r", allow_pickle=False)
    n2_refined = np.load(BASELINE / "N2_refined.npy", mmap_mode="r", allow_pickle=False)
    manifest = {
        "method": "literal upstream STEGO one-to-one Hungarian update/map behavior on the same 472 validation labels used for scoring",
        "label_use": "validation labels choose the mapping and score it; optimistic label-assisted evaluation",
        "upstream_update_validity_rule": "predicted cluster ID >=0 and < n_classes; n_classes=2, so C2-C7 are excluded",
        "stats_rows_clusters_cols_background_liver": stats.tolist(),
        "assignment_rows": rows.tolist(),
        "assignment_cols": cols.tolist(),
        "lookup_unmatched_value_minus_one": lookup.tolist(),
        "hungarian_liver_cluster": selected,
        "n2_liver_cluster": baseline_manifest["selected_cluster"],
        "hard_identical_to_n2": bool(np.array_equal(hard, n2_hard)),
        "refined_identical_to_n2": bool(np.array_equal(refined, n2_refined)),
        "checkpoint_sha256": checkpoint_sha,
        "fit_sha256": sha256(FIT / "fit.json"),
        "raw_prediction_sha256": sha256(BASELINE / "raw_clusters.npy"),
        "outputs": {
            "hungarian_hard.npy": sha256(out / "hungarian_hard.npy"),
            "hungarian_refined.npy": sha256(out / "hungarian_refined.npy"),
        },
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "metrics.json", {"units": "fractions", "mapping": manifest, "results": results})
    print(out / "metrics.json")


if __name__ == "__main__":
    main()
