import os
import random
from os.path import join

import numpy as np
import torch
import torch.multiprocessing
from PIL import Image
import pydicom
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from chaos_splits import load_splits, patient_split
from chaos_preprocessing import (plan_dataset, read_image, dicom_modality, ChaosTransform,
                                 preprocessing_contract, digest)
from pathlib import Path
import json


# ---------------------------------------------------------------------------
# Colormap helpers
# ---------------------------------------------------------------------------

def bit_get(val, idx):
    return (val >> idx) & 1


def create_pascal_label_colormap():
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)
    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3
    return colormap


def create_chaos_colormap():
    """
    0 = background  (black)
    1 = liver        (red)
    2 = right kidney (dark blue)
    3 = left kidney  (light blue)
    4 = spleen       (yellow)
    """
    return np.array([
        [0,   0,   0],
        [255,  0,   0],
        [0,   0, 255],
        [0, 128, 255],
        [255, 255,  0],
    ], dtype=np.uint8)


# ---------------------------------------------------------------------------
# CHAOS label pixel → class index mappings
#
# CT  Ground PNGs:  0 = background, 255 = liver
# MRI Ground PNGs:  0 = background, 63 = liver, 126 = right kidney,
#                   189 = left kidney, 252 = spleen
# ---------------------------------------------------------------------------

_CT_PIXEL_TO_CLASS  = {0: 0, 255: 1}
_MRI_PIXEL_TO_CLASS = {0: 0, 63: 1, 126: 2, 189: 3, 252: 4}


# ---------------------------------------------------------------------------
# CHAOS Dataset
# ---------------------------------------------------------------------------
#
# Exact on-disk layout (verified from screenshots):
#
#   <pytorch_data_dir>/
#     archive/
#       CHAOS_Train_Sets/
#         Train_Sets/
#           CT/
#             1/
#               DICOM_anon/   *.dcm
#               Ground/       liver_GT_000.png, liver_GT_001.png ...
#             2/ 5/ 6/ 8/ ...
#           MR/
#             <pid>/
#               T1DUAL/
#                 DICOM_anon/
#                   InPhase/  *.dcm
#                 Ground/     liver_GT_000.png ...
#               T2SPIR/
#                 DICOM_anon/ *.dcm
#                 Ground/     liver_GT_000.png ...
# ---------------------------------------------------------------------------

class CHAOS(Dataset):
    """
    Parameters
    ----------
    root             : str  – pytorch_data_dir (contains 'archive/')
    modality         : str  – 'CT' | 'T1DUAL' | 'T2SPIR' | 'all'
    image_set        : str  – 'train' | 'val' | 'all'
    transform        : PIL Image -> tensor [3, H, W]
    target_transform : PIL Image -> tensor [1, H, W]  (nearest, no interp)
    n_classes        : int – 2 (CT liver only) or 5 (all MRI organs)
    """

    MODALITIES = ("CT", "T1DUAL", "T2SPIR")

    def __init__(self, root, modality, image_set, transform, target_transform,
                 n_classes=5, use_preprocessed_data=False):
        super().__init__()
        assert modality in (*self.MODALITIES, "all"), \
            f"modality must be one of {self.MODALITIES + ('all',)}"
        assert image_set in ("train", "val", "all")

        self.modality         = modality
        self.image_set        = image_set
        if not isinstance(transform, ChaosTransform) or not isinstance(target_transform, ChaosTransform):
            raise ValueError("CHAOS v2 requires the shared image and mask transforms")
        self.transform        = transform
        self.target_transform = target_transform
        self.n_classes        = n_classes
        self.root             = root
        self.use_preprocessed_data = use_preprocessed_data
        self.split_manifest = load_splits()

        # Root of the actual CHAOS train data (verified path)
        self.train_root = join(root, "archive", "CHAOS_Train_Sets", "Train_Sets")

        # Each entry: (dcm_path, mask_path_or_None, modality_tag, patient_id)
        self.samples: list = []
        if self.use_preprocessed_data:
            self._collect_preprocessed_samples()
        else:
            self._collect_samples()

            # Frozen patient assignment, independent of discovered slice counts.
            if image_set != "all":
                keep = [i for i, s in enumerate(self.samples)
                        if patient_split(self.split_manifest, s[2], s[3]) == image_set]
                self.samples = [self.samples[i] for i in keep]
                self.records = [self.records[i] for i in keep]
        self.sample_ids = [r["sample_id"] for r in self.records]

    # ------------------------------------------------------------------
    # Sample collection — exact paths from screenshots
    # ------------------------------------------------------------------

    def _collect_samples(self):
        mods = self.MODALITIES if self.modality == "all" else (self.modality,)
        self.records = plan_dataset(self.train_root, mods, self.split_manifest)
        self.samples = [(join(self.train_root, r["dicom"]), join(self.train_root, r["mask"]),
                         r["modality"], r["patient_id"]) for r in self.records]

    # ------------------------------------------------------------------
    # DICOM → 8-bit RGB PIL Image
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dicom_as_pil(dcm_path: str) -> Image.Image:
        return Image.fromarray(read_image(dcm_path, dicom_modality(dcm_path)))

    @staticmethod
    def _preprocess_ct_slice(dcm_path, mask_path, transform, target_transform, modality):
        image = transform(read_image(dcm_path, modality))
        label = (target_transform(CHAOS._load_mask(mask_path, modality)).squeeze(0).long()
                 if mask_path is not None else torch.full(image.shape[1:], -1, dtype=torch.long))
        label[label == 255] = -1
        return image, label

    # ------------------------------------------------------------------
    # Ground PNG → uint8 class-index PIL Image (mode "L")
    # 255 is used as the ignore sentinel in uint8 space;
    # it is remapped to -1 (int64) inside __getitem__.
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mask(mask_path: str, modality: str) -> Image.Image:
        raw = np.array(Image.open(mask_path).convert("L"))   # uint8 grayscale
        lut = _CT_PIXEL_TO_CLASS if modality == "CT" else _MRI_PIXEL_TO_CLASS
        label = np.full(raw.shape, 255, dtype=np.uint8)      # 255 = ignore
        for px, cls in lut.items():
            label[raw == px] = cls
        return Image.fromarray(label, mode="L")

    def _collect_preprocessed_samples(self):
        preprocessed_root = Path(self.root) / "preprocessed_v2"
        manifest_path = preprocessed_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Generate approved derivatives first: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("schema_version") != 2 or
                manifest.get("preprocessing") != preprocessing_contract() or
                manifest.get("split_manifest_sha256") != digest(self.split_manifest)):
            raise ValueError("Derivative preprocessing/split manifest mismatch")
        records = manifest["samples"]
        families = {"legacy" if r["patient_id"].startswith("p") else "current" for r in records}
        if len(families) > 1:
            raise ValueError("Mixed preprocessing families require an explicit verified manifest")
        if families != {"current"}:
            raise ValueError("Legacy derivatives are excluded from CHAOS v2")
        mods = self.MODALITIES if self.modality == "all" else (self.modality,)
        self.records = []
        for record in records:
            mod, pid, split = record["modality"], record["patient_id"], record["split"]
            if mod not in self.MODALITIES or "OutPhase" in Path(record["dicom"]).parts:
                raise ValueError("Excluded modality/phase in derivative manifest")
            if patient_split(self.split_manifest, mod, pid) != split:
                raise ValueError(f"Patient {mod}/{pid} is in the wrong split")
            if mod not in mods or (self.image_set != "all" and split != self.image_set):
                continue
            image, mask = (preprocessed_root / record[k] for k in ("images", "labels"))
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(f"Missing derivative pair: {image}, {mask}")
            self.samples.append((str(image), str(mask), mod, pid))
            self.records.append(record)
        if not self.records or len({r["sample_id"] for r in self.records}) != len(self.records):
            raise ValueError("Empty or duplicate derivative sample IDs")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        dcm_path, mask_path, mod, _ = self.samples[index]
        if dcm_path.lower().endswith((".nii", ".nii.gz")):
            raise NotImplementedError("Paired NIfTI orientation/slice/affine workflow is not implemented")

        if self.use_preprocessed_data:
            with Image.open(dcm_path) as image:
                img = self.transform(image)
        else:
            img = self.transform(read_image(dcm_path, mod))
        label = self.target_transform(self._load_mask(mask_path, mod)).squeeze(0).long()
        label[label == 255] = -1
        img_sum = img.sum(0)
        valid_mask = (img_sum > (img_sum.min() + 0.01)).float()
        return img, label, valid_mask


# ---------------------------------------------------------------------------
# MaterializedDataset
# ---------------------------------------------------------------------------

class MaterializedDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.materialized = []
        loader = DataLoader(ds, num_workers=4, collate_fn=lambda l: l[0])
        for batch in tqdm(loader):
            self.materialized.append(batch)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ind):
        return self.materialized[ind]


# ---------------------------------------------------------------------------
# ContrastiveSegDataset — EAGLE-compatible wrapper
# ---------------------------------------------------------------------------
#
# In your Hydra / train config set:
#   dataset_name    : chaos
#   chaos_modality  : CT | T1DUAL | T2SPIR | all   (default: all)
#   chaos_n_classes : 5                             (default: 5)
#
# pytorch_data_dir must point to the folder that contains "archive/", i.e.:
#   /content/drive/MyDrive/EAGLE/src_EAGLE/pytorch_data_dir
# ---------------------------------------------------------------------------

class ContrastiveSegDataset(Dataset):

    def __init__(
        self,
        pytorch_data_dir,
        dataset_name,
        crop_type,              # ignored — kept for API compatibility
        image_set,
        transform,
        target_transform,
        cfg,
        aug_geometric_transform=None,
        aug_photometric_transform=None,
        mask=False,
        extra_transform=None,
        model_type_override=None,
        num_neighbors=None,
        pos_images=False,
        pos_labels=False,
    ):
        super().__init__()

        # assert dataset_name == "chaos", \
        #     f"This datasets.py only supports dataset_name='chaos', got '{dataset_name}'"

        self.mask                      = mask
        self.extra_transform           = extra_transform
        self.aug_geometric_transform   = aug_geometric_transform
        self.aug_photometric_transform = aug_photometric_transform
        self.pos_images                = pos_images
        self.pos_labels                = pos_labels

        modality  = getattr(cfg, "chaos_modality",  "all")
        n_classes = getattr(cfg, "chaos_n_classes", 5)
        use_preprocessed_data = getattr(cfg, "use_preprocessed_data", False)
        self.n_classes = n_classes

        self.dataset = CHAOS(
            root             = pytorch_data_dir,
            modality         = modality,
            image_set        = image_set,
            transform        = transform,
            target_transform = target_transform,
            n_classes        = n_classes,
            use_preprocessed_data = use_preprocessed_data,
        )

        # Build patient → indices map for positive pair sampling
        self._patient_to_indices = {}
        for idx, sample in enumerate(self.dataset.samples):
            pid = sample[3]  # patient_id
            if pid not in self._patient_to_indices:
                self._patient_to_indices[pid] = []
            self._patient_to_indices[pid].append(idx)
        # Pre-compute per-index lists for O(1) lookup
        self._pos_candidates = []
        for idx, sample in enumerate(self.dataset.samples):
            pid = sample[3]
            siblings = [i for i in self._patient_to_indices[pid] if i != idx]
            if len(siblings) == 0:
                siblings = [idx]  # fallback: use self if only one slice
            self._pos_candidates.append(siblings)

        # Load precomputed KNN neighbors if available
        self.num_neighbors = num_neighbors
        self.nns = None
        if num_neighbors is not None and num_neighbors > 0:
            from knn_cache import expected_identity, cache_path, read_cache
            identity = expected_identity(self.dataset, cfg)
            if num_neighbors >= identity["k"]:
                raise ValueError("Requested positive-neighbor count exceeds cached non-self ranks")
            nns_file = cache_path(pytorch_data_dir, identity)
            if not nns_file.is_file():
                raise FileNotFoundError(f"Verified KNN cache missing: {nns_file}; legacy/unverified caches cannot be reused")
            self.nns = torch.from_numpy(read_cache(nns_file, identity))
            print(f"Loaded verified KNN neighbors from {nns_file}, shape={self.nns.shape}")

    def __len__(self):
        return len(self.dataset)

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, ind):
        pack = self.dataset[ind]   # (img, label, valid_mask)

        seed = np.random.randint(2147483647)
        self._set_seed(seed)

        # Sample a positive pair using KNN neighbors or patient-based fallback
        if self.pos_images:
            if self.nns is not None:
                # Use precomputed KNN: pick a random neighbor from top-k
                # Skip index 0 (self), pick from indices 1..num_neighbors
                knn_neighbors = self.nns[ind]
                k = min(self.num_neighbors, len(knn_neighbors) - 1)
                assert k >= 1, "KNN cache must contain a non-self neighbor"
                pos_idx = knn_neighbors[random.randint(1, k)].item()
                assert 0 <= pos_idx < len(self.dataset), "KNN cache index outside dataset"
            else:
                # Fallback: patient-based sampling
                pos_idx = random.choice(self._pos_candidates[ind])
            pos_pack = self.dataset[pos_idx]
        else:
            pos_pack = pack

        # Spatial coordinate grid — used by some EAGLE loss terms
        coord_entries = torch.meshgrid(
            [torch.linspace(-1, 1, pack[0].shape[1]),
             torch.linspace(-1, 1, pack[0].shape[2])],
            indexing="ij"
        )
        coord = torch.cat([t.unsqueeze(-1) for t in coord_entries], -1)

        extra = self.extra_transform \
                if self.extra_transform is not None else lambda i, x: x

        ret = {
            "ind":       ind,
            "img":       extra(ind, pack[0]),
            "label":     extra(ind, pack[1]),
            "img_pos":   extra(ind, pos_pack[0]),
            "label_pos": extra(ind, pos_pack[1]),
        }

        # Apply geometric augmentation to get img_aug + coord_aug
        if self.aug_geometric_transform is not None:
            aug_seed = np.random.randint(2147483647)
            # Apply same random transform to both image and coordinate grid
            self._set_seed(aug_seed)
            ret["img_aug"] = self.aug_geometric_transform(extra(ind, pack[0]))
            self._set_seed(aug_seed)
            coord_aug = self.aug_geometric_transform(
                coord.permute(2, 0, 1)  # [2, H, W] for torchvision transforms
            ).permute(1, 2, 0)          # back to [H, W, 2]
            ret["coord_aug"] = coord_aug
        else:
            ret["img_aug"] = extra(ind, pack[0])
            ret["coord_aug"] = coord

        ret["coord"] = coord

        if self.mask:
            ret["mask"] = pack[2]
            ret["mask_pos"] = pos_pack[2]

        if self.aug_photometric_transform is not None:
            ret["img_pos_aug"] = self.aug_photometric_transform(ret["img_pos"])

        return ret
class NegativeImageDataset(Dataset):
    """
    A simple dataset to load images from a directory recursively.
    Used to supply completely different "negative" images to the contrastive loss.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.dcm')):
                    self.image_paths.append(os.path.join(root, file))
        # Sort or shuffle if necessary, but dataloader shuffle will handle randomizing batches
        self.image_paths = sorted(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            import pydicom
            from PIL import Image
            import numpy as np
            ds = pydicom.dcmread(img_path, force=True)
            arr = ds.pixel_array
            if arr.max() > arr.min():
                arr = (arr - arr.min()) / (arr.max() - arr.min()) * 65535.0
            else:
                arr = np.zeros_like(arr)
            img = Image.fromarray(arr.astype(np.uint16))
        except Exception:
            from PIL import Image
            import numpy as np
            img = Image.fromarray(np.zeros((256, 256), dtype=np.uint16))
        if self.transform:
            img = self.transform(img)
        # We don't have labels or masks, just return the image
        return {"img": img}
