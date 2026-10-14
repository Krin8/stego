"""Render the extracted native arrays without loading the model or feature cache."""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "audit/p1_hungarian/per_image_hungarian_visualization"
NPZ = OUT / "displayed_predictions.npz"
SELECTION = ROOT / "audit/improve/figure_selection.json"
RECORDS = ROOT / "audit/naming/image_records.json"
COLORS = np.array([
    [68, 119, 170], [238, 102, 119], [34, 136, 51], [204, 187, 68],
    [102, 204, 238], [170, 51, 119], [187, 187, 187], [102, 51, 153],
    [17, 17, 17],
], dtype=np.uint8)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def font(size, bold=False):
    choices = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in choices:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def overlay(gray, mask, color, alpha=0.42):
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    use = mask.astype(bool)
    rgb[use] = (1 - alpha) * rgb[use] + alpha * np.asarray(color, np.float32)
    return np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)


def mapping(raw, target):
    valid = raw < 8
    confusion = np.bincount(
        (raw[valid].astype(np.int16) * 2 + target[valid].astype(np.int16)).ravel(),
        minlength=16,
    ).reshape(8, 2)
    rows, cols = linear_sum_assignment(confusion, maximize=True)
    return int(rows[cols == 1][0]), confusion, rows, cols


def metric(pred, target):
    pred, target = pred.astype(bool), target.astype(bool)
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    fn = int((~pred & target).sum())
    return tp / max(tp + fp + fn, 1), 2 * tp / max(2 * tp + fp + fn, 1)


def main():
    data = np.load(NPZ, allow_pickle=False)
    fixed = json.loads(SELECTION.read_text())["eight_indices"]
    records = json.loads(RECORDS.read_text())
    if data["image"].shape != (8, 512, 512):
        raise ValueError("Unexpected extracted image geometry")

    entries = []
    for j, index in enumerate(fixed):
        selected, confusion, rows, cols = mapping(data["raw_native"][j], data["label"][j])
        iou, dice = metric(data["refined"][j], data["label"][j])
        entries.append({
            "index": index,
            "patient": records[index]["patient"],
            "sample_id": records[index]["sample_id"],
            "selected_liver_cluster": selected,
            "iou": iou,
            "dice": dice,
            "confusion_rows_clusters_cols_background_liver": confusion.tolist(),
            "assignment_rows": rows.tolist(),
            "assignment_cols": cols.tolist(),
        })

    page_w, page_h = 2020, 1210
    margin, gap, panel = 20, 20, 480
    header_h, row_title_h = 90, 40
    title_font, header_font, small_font = font(22, True), font(34, True), font(18)
    column_titles = [
        "Input CT", "Ground-truth overlay", "Raw P1 + K-means K=8",
        "Final per-image Hungarian F1 overlay",
    ]
    pages = []
    page_paths = []
    for page_number in range(4):
        page = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(page)
        heading = "P1 head: per-image label-assisted Hungarian predictions"
        draw.text((page_w // 2, 22), heading, fill="black", font=header_font,
                  anchor="ma")
        draw.text((page_w // 2, 62),
                  "Red = ground truth | Green = final prediction | Native 512 x 512",
                  fill=(60, 60, 60), font=small_font, anchor="ma")
        for local in range(2):
            j = page_number * 2 + local
            y_title = header_h + local * (row_title_h + panel + 35)
            y_image = y_title + row_title_h
            image8 = np.rint(np.clip(data["image"][j], 0, 1) * 255).astype(np.uint8)
            target = data["label"][j].astype(bool)
            raw = data["raw_native"][j].astype(np.int16)
            final = data["refined"][j].astype(bool)
            views = [
                np.repeat(image8[..., None], 3, axis=2),
                overlay(image8, target, [255, 31, 31]),
                COLORS[np.clip(raw, 0, 8)],
                overlay(image8, final, [26, 242, 77]),
            ]
            for col, (title, view) in enumerate(zip(column_titles, views)):
                x = margin + col * (panel + gap)
                shown = Image.fromarray(view).resize((panel, panel), Image.Resampling.NEAREST)
                page.paste(shown, (x, y_image))
                if col == 0:
                    subtitle = f"{title} - patient {entries[j]['patient']}, index {entries[j]['index']}"
                elif col == 3:
                    subtitle = (f"C{entries[j]['selected_liver_cluster']} | IoU "
                                f"{100*entries[j]['iou']:.2f}% | Dice {100*entries[j]['dice']:.2f}%")
                else:
                    subtitle = title
                draw.text((x + panel // 2, y_title + 4), subtitle, fill="black",
                          font=title_font, anchor="ma")
        legend_y = page_h - 48
        total_w = 9 * 105
        start_x = (page_w - total_w) // 2
        for cluster in range(9):
            x = start_x + cluster * 105
            draw.rectangle((x, legend_y, x + 24, legend_y + 24),
                           fill=tuple(COLORS[cluster]))
            label = f"C{cluster}" if cluster < 8 else "outside"
            draw.text((x + 31, legend_y + 1), label, fill="black", font=small_font)
        page_path = OUT / f"page_{page_number + 1}.png"
        page.save(page_path, format="PNG")
        pages.append(page)
        page_paths.append(page_path)

    pdf_path = OUT / "per_image_hungarian_predictions.pdf"
    pages[0].save(pdf_path, format="PDF", resolution=150, save_all=True,
                  append_images=pages[1:])
    manifest = {
        "method": "overcluster-aware one-to-one 8x2 Hungarian independently per displayed slice",
        "label_use": "each slice label selects its liver cluster and scores the final prediction",
        "warning": "label oracle visualization; not label-free inference",
        "fixed_indices": fixed,
        "rows": entries,
        "outputs": {
            pdf_path.name: sha(pdf_path),
            NPZ.name: sha(NPZ),
            **{p.name: sha(p) for p in page_paths},
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pdf_path)


if __name__ == "__main__":
    main()
