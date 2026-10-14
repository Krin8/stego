"""Create actual P1 raw-cluster and per-image Hungarian prediction panels."""
from pathlib import Path
import gc
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
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
SELECTION = ROOT / "audit/improve/figure_selection.json"
OUTPUT = ROOT / "audit/p1_hungarian/per_image_hungarian_visualization"

COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44",
          "#66CCEE", "#AA3377", "#BBBBBB", "#663399", "#111111"]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(np.arange(-0.5, 9.5, 1), 9)


def overlay(image, mask, color, alpha=0.42):
    rgb = np.repeat(image[..., None], 3, axis=2)
    use = np.asarray(mask, bool)
    rgb[use] = (1 - alpha) * rgb[use] + alpha * np.asarray(color)
    return np.clip(rgb, 0, 1)


def per_image_hungarian(native, target):
    """Overcluster-aware 8x2 one-to-one assignment for one labeled slice."""
    valid = native < 8
    confusion = np.bincount(
        (native[valid].astype(np.int16) * 2 + target[valid]).ravel(),
        minlength=16,
    ).reshape(8, 2)
    rows, cols = linear_sum_assignment(confusion, maximize=True)
    liver = rows[cols == 1]
    if len(liver) != 1:
        raise ValueError("Expected exactly one cluster mapped to liver")
    return int(liver[0]), confusion, rows, cols


def main():
    cfg = load_config(ROOT / "src2/configs/eval_config.yaml")
    setup(cfg)
    store = FeatureStore(cfg)
    ids = store.indices("val")
    position = {idx: j for j, idx in enumerate(ids)}
    fixed = json.loads(SELECTION.read_text())["eight_indices"]
    if any(i not in position for i in fixed):
        raise ValueError("A fixed figure index is not in the validation split")

    raw_all = np.load(BASELINE / "raw_clusters.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(resolve_path(cfg, cfg["paths"]["scoring_masks"]),
                     mmap_mode="r", allow_pickle=False)
    model, checkpoint_sha = load_head(cfg)
    centers, fitted = load_fit(FIT, cfg, checkpoint_sha)
    codes = embed(model, store.batch(fixed))

    out = new_output(OUTPUT)
    rows_data = []
    arrays = {"image": [], "label": [], "raw_native": [], "hard": [], "refined": []}
    for offset, i in enumerate(fixed):
        record = store.records[i]
        image = load_image(resolve_path(cfg, record["image"]), cfg)
        raw_grid = np.asarray(raw_all[position[i]])
        native = inverse_grid(raw_grid, record, mode="nearest", outside=8).astype(np.int16)
        target = labels[i] == 1
        selected, confusion, assignment_rows, assignment_cols = per_image_hungarian(native, target)

        d = distances(codes[offset], centers)
        weights = np.exp(-(d - d.min(1, keepdims=True)) / fitted["temperature"])
        weights /= weights.sum(1, keepdims=True)
        field = weights[:, selected].reshape(28, 28)
        hard = native == selected
        final = refine(field, image, record, cfg)
        metric = scores(counts(final, target))

        arrays["image"].append(image)
        arrays["label"].append(target)
        arrays["raw_native"].append(native)
        arrays["hard"].append(hard)
        arrays["refined"].append(final)
        rows_data.append({
            "index": i,
            "sample_id": record["sample_id"],
            "patient": record["patient"],
            "selected_liver_cluster": selected,
            "confusion_rows_clusters_cols_background_liver": confusion.tolist(),
            "assignment_rows": assignment_rows.tolist(),
            "assignment_cols": assignment_cols.tolist(),
            "final_iou": metric["liver_iou"],
            "final_dice": metric["liver_dice"],
        })

    arrays = {name: np.stack(value) for name, value in arrays.items()}
    np.savez_compressed(out / "displayed_predictions.npz", **arrays)
    # Release model/cache objects before rendering high-resolution pages.
    del model, store, codes, raw_all, labels
    gc.collect()
    pdf_path = out / "per_image_hungarian_predictions.pdf"
    png_paths = []
    with PdfPages(pdf_path) as pdf:
        for page in range(4):
            start = page * 2
            fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
            fig.subplots_adjust(left=0.025, right=0.99, bottom=0.15, top=0.88,
                                hspace=0.22, wspace=0.035)
            for local in range(2):
                j = start + local
                row = rows_data[j]
                image = arrays["image"][j]
                target = arrays["label"][j]
                raw = arrays["raw_native"][j]
                final = arrays["refined"][j]

                axes[local, 0].imshow(image, cmap="gray", vmin=0, vmax=1,
                                      interpolation="nearest")
                axes[local, 0].set_title(
                    f"Input CT\nPatient {row['patient']} - index {row['index']}", fontsize=10
                )
                axes[local, 1].imshow(overlay(image, target, [1.0, 0.12, 0.12]),
                                      interpolation="nearest")
                axes[local, 1].set_title("Ground-truth liver overlay", fontsize=10)
                axes[local, 2].imshow(raw, cmap=CMAP, norm=NORM, interpolation="nearest")
                axes[local, 2].set_title("Raw P1 head + K-means K=8", fontsize=10)
                axes[local, 3].imshow(overlay(image, final, [0.10, 0.95, 0.30]),
                                      interpolation="nearest")
                axes[local, 3].set_title(
                    f"Per-image Hungarian: C{row['selected_liver_cluster']}\n"
                    f"F1 IoU {100*row['final_iou']:.2f}% - Dice {100*row['final_dice']:.2f}%",
                    fontsize=10,
                )
                for axis in axes[local]:
                    axis.set_xlim(-0.5, 511.5)
                    axis.set_ylim(511.5, -0.5)
                    axis.set_aspect("equal")
                    axis.axis("off")

            handles = [Patch(facecolor=COLORS[c], label=f"C{c}") for c in range(8)]
            handles.append(Patch(facecolor=COLORS[8], label="Outside crop"))
            fig.legend(handles=handles, loc="lower center", ncol=9, frameon=False,
                       fontsize=10, bbox_to_anchor=(0.5, 0.035))
            fig.suptitle(
                "P1 trained head - per-image label-assisted Hungarian visualization\n"
                "Red = ground truth; green = final F1 prediction; native 512 x 512",
                fontsize=16,
            )
            pdf.savefig(fig, facecolor="white")
            png = out / f"preview_page_{page + 1}.png"
            fig.savefig(png, dpi=150, facecolor="white")
            png_paths.append(png)
            plt.close(fig)

    manifest = {
        "method": "overcluster-aware one-to-one 8x2 Hungarian assignment independently on each displayed slice",
        "label_use": "each displayed slice's ground-truth mask selects its liver cluster and scores the final mask",
        "warning": "oracle visualization; not label-free inference and not literal upstream C0/C1-only overcluster behavior",
        "fixed_indices": fixed,
        "selection_source": str(SELECTION),
        "checkpoint_sha256": checkpoint_sha,
        "fit_sha256": sha256(FIT / "fit.json"),
        "rows": rows_data,
        "outputs": {
            pdf_path.name: sha256(pdf_path),
            "displayed_predictions.npz": sha256(out / "displayed_predictions.npz"),
            **{p.name: sha256(p) for p in png_paths},
        },
    }
    write_json(out / "manifest.json", manifest)
    print(pdf_path)


if __name__ == "__main__":
    main()
