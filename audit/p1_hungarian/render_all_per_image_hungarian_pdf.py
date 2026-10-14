"""Render all 472 native per-image Hungarian predictions into a 118-page PDF."""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "audit/p1_hungarian/per_image_hungarian_all"
OUTPUT = SOURCE / "all_472_per_image_hungarian_predictions.pdf"
RECORDS = ROOT / "audit/naming/image_records.json"
LABELS = ROOT / "audit/improve/scoring_masks_native.npy"
TEMP = ROOT / "tmp/pdfs/p1_hungarian_all_panels"
COLORS = np.array([
    [68, 119, 170], [238, 102, 119], [34, 136, 51], [204, 187, 68],
    [102, 204, 238], [170, 51, 119], [187, 187, 187], [102, 51, 153],
    [17, 17, 17],
], dtype=np.uint8)
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_ct(path):
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.shape != (512, 512):
        raise ValueError(f"Unexpected CT shape {values.shape}: {path}")
    return np.rint(values.astype(np.float32) / 65535.0 * 255).astype(np.uint8)


def overlay(gray, mask, color, alpha=0.42):
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    use = np.asarray(mask, bool)
    rgb[use] = (1 - alpha) * rgb[use] + alpha * np.asarray(color, np.float32)
    return np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)


def save_panel(array, path, flat=False):
    image = Image.fromarray(array)
    if flat:
        image.save(path, format="PNG", optimize=True)
    else:
        image.save(path, format="JPEG", quality=90, optimize=True)


def raster_text(pdf, temp_paths, name, x, y, width, height, text, size, bold=False):
    scale = 3
    strip = Image.new("RGB", (round(width * scale), round(height * scale)), "white")
    font = ImageFont.truetype(FONT_BOLD_PATH if bold else FONT_PATH, round(size * scale))
    draw = ImageDraw.Draw(strip)
    draw.text((strip.width / 2, strip.height / 2), text, fill="black", font=font,
              anchor="mm", stroke_width=0)
    path = TEMP / f"text_{name}.png"
    strip.save(path, format="PNG", optimize=True)
    temp_paths.append(path)
    pdf.drawImage(str(path), x, y, width=width, height=height, preserveAspectRatio=False)


def main():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    selections = json.loads((SOURCE / "selections.json").read_text())
    records = json.loads(RECORDS.read_text())
    labels = np.load(LABELS, mmap_mode="r", allow_pickle=False)
    raw = np.load(SOURCE / "raw_clusters_native.npy", mmap_mode="r", allow_pickle=False)
    final = np.load(SOURCE / "refined_predictions.npy", mmap_mode="r", allow_pickle=False)
    if len(selections) != 472 or raw.shape != (472, 512, 512) or final.shape != raw.shape:
        raise ValueError("Unexpected all-validation artifact shape")

    TEMP.mkdir(parents=True, exist_ok=True)
    for old in TEMP.iterdir():
        if old.is_file():
            old.unlink()

    page_w, page_h = 969.6, 1104.0
    margin, gap = 15.0, 10.0
    panel = (page_w - 2 * margin - 3 * gap) / 4
    row_height, first_top = 250.0, 1038.0
    total_pages = (len(selections) + 3) // 4
    pdf = canvas.Canvas(str(OUTPUT), pagesize=(page_w, page_h), pageCompression=1)
    pdf.setTitle("All 472 P1 per-image Hungarian predictions")
    temp_paths = []

    for page_number in range(total_pages):
        raster_text(pdf, temp_paths, f"p{page_number:03d}_header", 135, 1073, 700, 22,
                    "P1 head: per-image label-assisted Hungarian predictions", 17, bold=True)
        raster_text(pdf, temp_paths, f"p{page_number:03d}_subtitle", 185, 1058, 600, 14,
                    (f"Page {page_number + 1}/{total_pages} | Red = truth | "
                     "Green = prediction | Native 512 x 512"), 8)

        for local in range(4):
            position = page_number * 4 + local
            if position >= len(selections):
                break
            entry = selections[position]
            index = entry["index"]
            record = records[index]
            image = load_ct(record["image"])
            target = labels[index] == 1
            cluster_map = raw[position]
            prediction = final[position].astype(bool)
            metric = entry["metrics"]
            views = [
                np.repeat(image[..., None], 3, axis=2),
                overlay(image, target, [255, 31, 31]),
                COLORS[np.clip(cluster_map, 0, 8)],
                overlay(image, prediction, [26, 242, 77]),
            ]
            titles = [
                f"Input CT - patient {entry['patient']}, index {index}",
                "Ground-truth liver overlay",
                "Raw P1 head + K-means K=8",
                (f"Hungarian C{entry['selected_liver_cluster']} | IoU "
                 f"{100*metric['liver_iou']:.2f}% | Dice {100*metric['liver_dice']:.2f}%"),
            ]
            row_top = first_top - local * row_height
            image_y = row_top - 20 - panel
            for col, (view, title) in enumerate(zip(views, titles)):
                x = margin + col * (panel + gap)
                suffix = "png" if col == 2 else "jpg"
                path = TEMP / f"p{page_number:03d}_r{local}_c{col}.{suffix}"
                save_panel(view, path, flat=(col == 2))
                temp_paths.append(path)
                raster_text(pdf, temp_paths, f"p{page_number:03d}_r{local}_c{col}_title",
                            x, row_top - 17, panel, 13, title, 7.6, bold=True)
                pdf.drawImage(str(path), x, image_y, width=panel, height=panel,
                              preserveAspectRatio=False)

        legend_y = 13
        legend_width = 72
        legend_start = (page_w - legend_width * 9) / 2
        pdf.setFont("Helvetica", 7)
        for cluster in range(9):
            x = legend_start + cluster * legend_width
            r, g, b = COLORS[cluster] / 255.0
            pdf.setFillColorRGB(float(r), float(g), float(b))
            pdf.rect(x, legend_y, 9, 9, fill=1, stroke=0)
            pdf.setFillColorRGB(0, 0, 0)
            pdf.drawString(x + 12, legend_y + 1,
                           f"C{cluster}" if cluster < 8 else "outside")
        pdf.showPage()
    pdf.save()

    for path in temp_paths:
        path.unlink()
    manifest["visualization"] = {
        "pdf": str(OUTPUT),
        "pages": total_pages,
        "slices_per_page": 4,
        "columns": ["input CT", "ground-truth overlay", "raw K=8 clusters", "final F1 overlay"],
        "pdf_sha256": sha(OUTPUT),
    }
    (SOURCE / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
