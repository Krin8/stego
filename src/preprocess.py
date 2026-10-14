"""CHAOS v2 entrypoint: native paired CT/T1-in/T2 slices, dry-run by default.

Run `python preprocess.py --dry-run` to validate counts and associations.
Only `--write` creates a NEW preprocessed_v2 tree. The active v2 contract is
implemented in chaos_preprocessing.py; older standalone resampling helpers
below are retained for historical experiments and are not used by this CLI.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import pydicom
import SimpleITK as sitk
from PIL import Image

# --------------------------------------------------------------------------- #
# Configuration constants (from the paper's described pipeline)
# --------------------------------------------------------------------------- #
HU_MIN, HU_MAX = -150, 250          # HU clipping window
TARGET_SIZE = (256, 256)            # final 2D slice size (W, H)
DEFAULT_TARGET_SPACING = (1.0, 1.0, 1.0)  # mm, isotropic resample target


# --------------------------------------------------------------------------- #
# 1. Loading
# --------------------------------------------------------------------------- #
def load_dicom_series_as_sitk(dicom_dir: str):
    """Load a directory of DICOM slices belonging to one patient/study into a
    single 3D SimpleITK image, correctly ordered by slice position."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {dicom_dir}")

    # If multiple series exist, pick the one with the most slices (the CT volume).
    best_series, best_len = None, -1
    for sid in series_ids:
        file_names = reader.GetGDCMSeriesFileNames(dicom_dir, sid)
        if len(file_names) > best_len:
            best_series, best_len = sid, len(file_names)

    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, best_series)
    reader.SetFileNames(file_names)
    image = reader.Execute()
    return image, file_names


def load_mask_stack_as_sitk(
    ground_dir: str,
    reference_image: sitk.Image,
    dicom_dir: str = None,
    sitk_file_names: list = None,
) -> sitk.Image:
    """CHAOS ground-truth masks ship as a stack of 2D PNGs (one per slice).
    Aligns PNG slices with the DICOM files loaded by SimpleITK, then stacks them
    into a 3D volume sharing the CT's geometry (spacing/origin/direction)."""
    from chaos_preprocessing import pair_series
    if dicom_dir is None or sitk_file_names is None:
        raise ValueError("Explicit DICOM IDs are required for mask association")
    paired = pair_series(dicom_dir, ground_dir, "CT", os.path.basename(os.path.dirname(dicom_dir)))
    by_image = {os.path.basename(r["dicom"]): r["mask"] for r in paired}
    png_files = [by_image[os.path.basename(path)] for path in sitk_file_names]

    slices = [np.array(Image.open(f).convert("L")) for f in png_files]
    volume = np.stack(slices, axis=0).astype(np.uint8)  # (Z, Y, X)

    mask_img = sitk.GetImageFromArray(volume)
    mask_img.SetSpacing(reference_image.GetSpacing())
    mask_img.SetOrigin(reference_image.GetOrigin())
    mask_img.SetDirection(reference_image.GetDirection())
    return mask_img


# --------------------------------------------------------------------------- #
# 2. Resampling to uniform spatial resolution
# --------------------------------------------------------------------------- #
def resample_volume(
    image: sitk.Image,
    target_spacing=DEFAULT_TARGET_SPACING,
    is_mask: bool = False,
) -> sitk.Image:
    """Resample a SimpleITK image to `target_spacing`, using linear
    interpolation for intensity images and nearest-neighbor for masks."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0 if is_mask else -1000)  # air HU for CT bg
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)

    return resampler.Execute(image)


# --------------------------------------------------------------------------- #
# 3-4. HU clipping + normalization to uint8
# --------------------------------------------------------------------------- #
def clip_and_normalize(hu_slice: np.ndarray, window=(HU_MIN, HU_MAX)) -> np.ndarray:
    """Clip to [HU_MIN, HU_MAX] then linearly map to [0, 255] uint8."""
    low, high = window
    if high <= low:
        raise ValueError("CT window must have increasing endpoints")
    clipped = np.clip(hu_slice, low, high).astype(np.float32)
    normalized = (clipped - low) / (high - low)  # -> [0, 1]
    return (normalized * 255.0).round().astype(np.uint8)


# --------------------------------------------------------------------------- #
# 5. Resize
# --------------------------------------------------------------------------- #
def resize_image(slice_2d: np.ndarray, size=TARGET_SIZE) -> np.ndarray:
    return cv2.resize(slice_2d, size, interpolation=cv2.INTER_LINEAR)


def resize_mask(slice_2d: np.ndarray, size=TARGET_SIZE) -> np.ndarray:
    # Nearest-neighbor for masks so label values are never interpolated.
    return cv2.resize(slice_2d, size, interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------- #
# 6. Full per-patient pipeline
# --------------------------------------------------------------------------- #
def process_patient(
    patient_dir: str,
    output_dir: str,
    target_spacing=DEFAULT_TARGET_SPACING,
    image_size=TARGET_SIZE,
):
    patient_id = os.path.basename(os.path.normpath(patient_dir))
    dicom_dir = os.path.join(patient_dir, "DICOM_anon")
    ground_dir = os.path.join(patient_dir, "Ground")

    img_out_dir = os.path.join(output_dir, "images")
    mask_out_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    # --- load ---
    ct_image, file_names = load_dicom_series_as_sitk(dicom_dir)
    has_mask = os.path.isdir(ground_dir) and len(glob.glob(os.path.join(ground_dir, "*.png"))) > 0
    mask_image = (
        load_mask_stack_as_sitk(ground_dir, ct_image, dicom_dir, file_names)
        if has_mask
        else None
    )

    # --- resample to uniform spacing ---
    ct_resampled = resample_volume(ct_image, target_spacing, is_mask=False)
    ct_array = sitk.GetArrayFromImage(ct_resampled)  # (Z, Y, X), HU values

    if has_mask:
        mask_resampled = resample_volume(mask_image, target_spacing, is_mask=True)
        mask_array = sitk.GetArrayFromImage(mask_resampled)
    else:
        mask_array = None

    # --- per-slice: clip/normalize/resize/save ---
    num_slices = ct_array.shape[0]
    for z in range(num_slices):
        hu_slice = ct_array[z]
        img_uint8 = clip_and_normalize(hu_slice)
        img_resized = resize_image(img_uint8, image_size)

        stem = f"CT_{patient_id}_{z:04d}"
        cv2.imwrite(os.path.join(img_out_dir, f"{stem}.png"), img_resized)

        if has_mask:
            mask_slice = mask_array[z]
            mask_resized = resize_mask(mask_slice, image_size)
            cv2.imwrite(os.path.join(mask_out_dir, f"{stem}.png"), mask_resized)

    print(f"[{patient_id}] processed {num_slices} slices -> {output_dir}")


def process_chaos_dataset(chaos_root: str, output_dir: str, target_spacing, image_size):
    patient_dirs = sorted(
        d for d in glob.glob(os.path.join(chaos_root, "*"))
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, "DICOM_anon"))
    )
    if not patient_dirs:
        raise RuntimeError(f"No valid patient folders with DICOM_anon found under {chaos_root}")

    all_patients = sorted([os.path.basename(os.path.normpath(d)) for d in patient_dirs])
    n_val = max(1, int(0.2 * len(all_patients)))
    val_patients = set(all_patients[-n_val:])

    for patient_dir in patient_dirs:
        patient_id = os.path.basename(os.path.normpath(patient_dir))
        split = "val" if patient_id in val_patients else "train"
        split_output_dir = os.path.join(output_dir, split)
        try:
            process_patient(patient_dir, split_output_dir, target_spacing, image_size)
        except Exception as e:
            print(f"[WARN] Skipping {patient_dir}: {e}")


# --------------------------------------------------------------------------- #
# 7. Data augmentation (rotation +/-15deg, scaling 0.9-1.1) for training
# --------------------------------------------------------------------------- #
def build_train_augmentations():
    """Albumentations pipeline matching the paper's described augmentations.
    Applied jointly to image + mask so spatial correspondence is preserved."""
    try:
        import albumentations as A
    except ImportError:
        raise ImportError(
            "albumentations is required for data augmentation. "
            "Please install it via: pip install albumentations"
        )

    return A.Compose(
        [
            A.Affine(
                rotate=(-15, 15),
                scale=(0.9, 1.1),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                fit_output=False,
                p=0.8,
            ),
        ]
    )


class LiverCTDataset:
    """Minimal torch-style Dataset over the preprocessed PNG pairs, applying
    the rotation/scaling augmentation during training."""

    def __init__(self, images_dir: str, masks_dir: str, train: bool = True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.stems = sorted(
            os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(".png")
        )
        self.train = train
        self.aug = build_train_augmentations() if train else None

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img_path = os.path.join(self.images_dir, f"{stem}.png")
        mask_path = os.path.join(self.masks_dir, f"{stem}.png")

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.train and self.aug is not None:
            augmented = self.aug(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = image.astype(np.float32) / 255.0  # scale to [0, 1] for the network
        return image[None, ...], mask[None, ...]  # add channel dim (1, H, W)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="CHAOS v2 native DICOM/PNG derivatives (dry-run by default)")
    parser.add_argument("--chaos_root", default=os.path.join(script_dir, "pytorch_data_dir/archive/CHAOS_Train_Sets/Train_Sets"))
    parser.add_argument("--output_dir", default=os.path.join(script_dir, "pytorch_data_dir/preprocessed_v2"))
    parser.add_argument("--ct_window", nargs=2, type=float, default=[HU_MIN, HU_MAX])
    parser.add_argument("--modalities", nargs="+", choices=["CT", "T1DUAL", "T2SPIR"], default=["CT", "T1DUAL", "T2SPIR"])
    parser.add_argument("--write", action="store_true", help="Generate a new derivative tree; refuses existing output")
    parser.add_argument("--dry-run", action="store_true", help="Plan/validate only; cannot be combined with --write")
    args = parser.parse_args()
    if args.write and args.dry_run:
        parser.error("--write and --dry-run are mutually exclusive")
    return args


if __name__ == "__main__":
    from chaos_preprocessing import plan_dataset, generate_derivatives
    args = parse_args()
    records = plan_dataset(args.chaos_root, args.modalities)
    generate_derivatives(args.chaos_root, args.output_dir, records, args.ct_window, write=args.write)
