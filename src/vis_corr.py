"""
Accurate Visualization of Self & KNN Correspondence
=====================================================
Shows exactly what happens during STEGO/EAGLE training:

Row 1: Inputs – anchor image + self-pair (same patient, different slice)
                              + KNN-pair (different patient, same modality)
Row 2: Self-correspondence – feature correlation between anchor & self-pair
Row 3: KNN correspondence  – feature correlation between anchor & KNN-pair
Row 4: Contrastive loss mechanism – pos_intra / pos_inter / neg_inter shifts

Uses the actual BiomedCLIP backbone to extract 768-d features, then
`tensor_correlation(norm(f1), norm(f2))` exactly as ContrastiveCorrelationLoss does.
"""

import sys
import os

# Force UTF-8 on Windows console to avoid cp1252 encoding errors
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure the src directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torchvision import transforms as T

# ── Local imports (from the project) ─────────────────────────────────
from utils import prep_for_plot
from data import ContrastiveSegDataset, create_chaos_colormap


# ── Re-implement the exact functions from modules.py ─────────────────
def norm(t):
    """Exactly as in modules.py"""
    return F.normalize(t, dim=1, eps=1e-10)


def tensor_correlation(a, b):
    """Exactly as in modules.py — einsum 'nchw,ncij->nhwij'"""
    return torch.einsum("nchw,ncij->nhwij", a, b)


def sample(t: torch.Tensor, coords: torch.Tensor):
    """Exactly as in modules.py"""
    return F.grid_sample(t, coords.permute(0, 2, 1, 3),
                         padding_mode='reflection', align_corners=True)


# ── Build the BiomedCLIP featurizer just like train_segmentation.py ──
def build_featurizer(cfg):
    """Build the same featurizer used during training."""
    from modules import BiomedCLIPFeaturizer
    dim = cfg.dim
    net = BiomedCLIPFeaturizer(dim, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.eval()
    return net, device


# ── Minimal config object so we don't need Hydra ────────────────────
class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def make_cfg():
    """Load config from train_config.yaml."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/train_config.yaml")
    
    # Override for cleaner visualization output
    cfg.feature_samples = 11
    
    return cfg


def get_transform(res, is_label, crop_type):
    """Same as utils.get_transform."""
    from utils import normalize

    class ToTargetTensorLocal:
        def __call__(self, pic):
            return torch.as_tensor(np.array(pic), dtype=torch.int64).unsqueeze(0)

    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        cropper = T.CenterCrop(res)

    if is_label:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          ToTargetTensorLocal()])
    else:
        normalize_t = T.Normalize([0.48145466, 0.4578275, 0.40821073],
                                  [0.26862954, 0.26130258, 0.27577711])
        return T.Compose([T.Resize(res, T.InterpolationMode.BILINEAR),
                          cropper,
                          T.ToTensor(),
                          normalize_t])


def find_sample_indices(dataset):
    """
    Find three indices that demonstrate the correspondence pairs:
      1) anchor_idx         – an arbitrary slice with visible organ
      2) self_idx           – same patient, different slice
      3) knn_idx            – different patient, same modality
    Also return a negative (random permutation) index.
    """
    import random
    inner = dataset.dataset  # the CHAOS dataset

    # Pick a slice roughly in the middle of a patient so organ is visible
    patients = {}
    for idx, s in enumerate(inner.samples):
        pid = s[3]
        if pid not in patients:
            patients[pid] = []
        patients[pid].append(idx)

    # Pick two patients with enough slices
    good_patients = {p: idxs for p, idxs in patients.items() if len(idxs) >= 5}
    pids = sorted(good_patients.keys())
    if len(pids) < 2:
        raise RuntimeError("Need at least 2 patients with ≥5 slices each")

    # Randomly select Patient A and Patient B
    pid_a, pid_b = random.sample(pids, 2)

    # Anchor = random slice from middle 50% of Patient A
    slices_a = good_patients[pid_a]
    q1 = len(slices_a) // 4
    q3 = q1 * 3
    a_idx = random.randint(q1, q3 - 2) # ensure room for +2
    anchor_idx = slices_a[a_idx]

    # Self = different slice, same patient (2 slices away)
    self_idx = slices_a[a_idx + 2] 

    # KNN = random slice from middle 50% of Patient B (different patient, same modality)
    slices_b = good_patients[pid_b]
    b_idx = random.randint(len(slices_b) // 4, (len(slices_b) // 4) * 3)
    knn_idx = slices_b[b_idx]

    # Negative = a distant random patient (different from A and B if possible)
    remaining_pids = [p for p in pids if p not in (pid_a, pid_b)]
    neg_pid = random.choice(remaining_pids) if remaining_pids else pid_b
    slices_neg = good_patients[neg_pid]
    neg_idx = random.choice(slices_neg)

    return anchor_idx, self_idx, knn_idx, neg_idx, pid_a, pid_b, neg_pid


def extract_features(net, img_tensor, device):
    """Run image through BiomedCLIP and return (backbone_feats, projected_code)."""
    img = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        out = net(img)
        if len(out) == 3:
            feats, code, _ = out
        else:
            feats, code = out
    return feats, code


def compute_correlation_map(feats_a, feats_b, query_h, query_w):
    """
    Compute the feature-space correlation between a single query position
    in feats_a and ALL positions in feats_b.

    Returns a 2D similarity map of shape (H_b, W_b).
    """
    # feats_a: (1, C, H, W),  feats_b: (1, C, H, W)
    fa = norm(feats_a)
    fb = norm(feats_b)

    # Query vector at (query_h, query_w) from feats_a
    query = fa[:, :, query_h, query_w]  # (1, C)

    # Dot product with every position in feats_b
    sim = torch.einsum("bc,bchw->bhw", query, fb)  # (1, H, W)
    return sim.squeeze(0).cpu().numpy()


def compute_full_correlation(feats_a, feats_b):
    """
    Compute the full tensor correlation as used in ContrastiveCorrelationLoss.
    Returns shape (H_a, W_a, H_b, W_b) — correlation from every position in A
    to every position in B.
    """
    corr = tensor_correlation(norm(feats_a), norm(feats_b))  # (1, H, W, H, W)
    return corr.squeeze(0).cpu()


def main():
    print("=" * 70)
    print("ACCURATE CORRESPONDENCE VISUALIZATION")
    print("Using actual BiomedCLIP features, not pixel-space similarity")
    print("=" * 70)

    cfg = make_cfg()

    # ── 1. Build dataset (same as training) ──────────────────────────
    print("\n[1/5] Loading CHAOS dataset...")
    transform = get_transform(cfg.res, False, cfg.loader_crop_type)
    target_transform = get_transform(cfg.res, True, cfg.loader_crop_type)

    dataset = ContrastiveSegDataset(
        pytorch_data_dir=cfg.pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=transform,
        target_transform=target_transform,
        cfg=cfg,
        num_neighbors=cfg.num_neighbors,
        mask=True,
        pos_images=True,
        pos_labels=True,
    )
    print(f"   Dataset has {len(dataset)} samples, {dataset.n_classes} classes")

    # ── 2. Pick representative indices ───────────────────────────────
    print("\n[2/5] Selecting anchor, self-pair, KNN-pair, and negative...")
    anchor_idx, self_idx, knn_idx, neg_idx, pid_a, pid_b, neg_pid = \
        find_sample_indices(dataset)

    inner = dataset.dataset
    print(f"   Anchor : idx={anchor_idx}  patient={pid_a}  "
          f"file={Path(inner.samples[anchor_idx][0]).name}")
    print(f"   Self   : idx={self_idx}  patient={pid_a}  "
          f"file={Path(inner.samples[self_idx][0]).name}")
    print(f"   KNN    : idx={knn_idx}  patient={pid_b}  "
          f"file={Path(inner.samples[knn_idx][0]).name}")
    print(f"   Neg    : idx={neg_idx}  patient={neg_pid}  "
          f"file={Path(inner.samples[neg_idx][0]).name}")

    # Load raw data (img, label, mask)
    anchor_pack = inner[anchor_idx]
    self_pack   = inner[self_idx]
    knn_pack    = inner[knn_idx]
    neg_pack    = inner[neg_idx]

    # ── 3. Build featurizer ──────────────────────────────────────────
    print("\n[3/5] Loading BiomedCLIP backbone...")
    net, device = build_featurizer(cfg)
    print(f"   Running on: {device}")

    # ── 4. Extract features ──────────────────────────────────────────
    print("\n[4/5] Extracting features...")
    feats_anchor, code_anchor = extract_features(net, anchor_pack[0], device)
    feats_self,   code_self   = extract_features(net, self_pack[0], device)
    feats_knn,    code_knn    = extract_features(net, knn_pack[0], device)
    feats_neg,    code_neg    = extract_features(net, neg_pack[0], device)

    feat_h, feat_w = feats_anchor.shape[2], feats_anchor.shape[3]
    print(f"   Feature map size: {feat_h}×{feat_w}  "
          f"(from {cfg.res}×{cfg.res} input, patch_size=16)")

    # ── 5. Compute correlations for 25 points ───────────────────────
    num_points = 25
    print(f"\n[5/5] Computing feature correlations for {num_points} points...")
    
    import random
    anchor_mask = anchor_pack[1].unsqueeze(0).unsqueeze(0).to(torch.float32)
    anchor_mask_feat = F.interpolate(anchor_mask, size=(feat_h, feat_w), mode='nearest').squeeze()
    
    query_points = []
    
    # Initialize aggregators
    self_corr_map_agg = np.zeros((feat_h, feat_w))
    knn_corr_map_agg = np.zeros((feat_h, feat_w))
    neg_corr_map_agg = np.zeros((feat_h, feat_w))
    self_self_map_agg = np.zeros((feat_h, feat_w))
    
    code_self_corr_agg = np.zeros((feat_h, feat_w))
    code_knn_corr_agg = np.zeros((feat_h, feat_w))
    code_neg_corr_agg = np.zeros((feat_h, feat_w))
    
    self_best_pixels = []
    knn_best_pixels = []
    neg_best_pixels = []

    patch_size = 16

    for i in range(num_points):
        # 100% chance to pick from the salient (non-zero) region for visualization
        if cfg.use_salience:
            nonzeros = torch.nonzero(anchor_mask_feat)
            if len(nonzeros) > 0:
                idx = torch.randint(len(nonzeros), size=(1,)).item()
                query_h, query_w = nonzeros[idx][0].item(), nonzeros[idx][1].item()
            else:
                query_h, query_w = random.randint(0, feat_h - 1), random.randint(0, feat_w - 1)
        else:
            query_h, query_w = random.randint(0, feat_h - 1), random.randint(0, feat_w - 1)
            
        query_points.append((query_h, query_w))
        
        sc_map = compute_correlation_map(feats_anchor, feats_self, query_h, query_w)
        kc_map = compute_correlation_map(feats_anchor, feats_knn, query_h, query_w)
        nc_map = compute_correlation_map(feats_anchor, feats_neg, query_h, query_w)
        
        def get_best_px(cmap):
            idx = np.unravel_index(np.argmax(cmap), cmap.shape)
            return idx[1] * patch_size + patch_size // 2, idx[0] * patch_size + patch_size // 2
            
        self_best_pixels.append(get_best_px(sc_map))
        knn_best_pixels.append(get_best_px(kc_map))
        neg_best_pixels.append(get_best_px(nc_map))
        
        self_corr_map_agg += sc_map
        knn_corr_map_agg += kc_map
        neg_corr_map_agg += nc_map
        self_self_map_agg += compute_correlation_map(feats_anchor, feats_anchor, query_h, query_w)
        
        code_self_corr_agg += compute_correlation_map(code_anchor, code_self, query_h, query_w)
        code_knn_corr_agg += compute_correlation_map(code_anchor, code_knn, query_h, query_w)
        code_neg_corr_agg += compute_correlation_map(code_anchor, code_neg, query_h, query_w)

    # Average them
    self_corr_map = self_corr_map_agg / num_points
    knn_corr_map = knn_corr_map_agg / num_points
    neg_corr_map = neg_corr_map_agg / num_points
    self_self_map = self_self_map_agg / num_points
    
    code_self_corr = code_self_corr_agg / num_points
    code_knn_corr = code_knn_corr_agg / num_points
    code_neg_corr = code_neg_corr_agg / num_points

    # Map back to pixel coordinates for visualization
    patch_size = 16
    query_pixels = [(qw * patch_size + patch_size // 2, qh * patch_size + patch_size // 2) for qh, qw in query_points]

    def count_hits(target_markers, label_tensor):
        lbl = label_tensor.numpy()
        if len(lbl.shape) > 2:
            lbl = lbl.squeeze()
        hits = 0
        for px, py in target_markers:
            if py < lbl.shape[0] and px < lbl.shape[1]:
                if lbl[py, px] > 0:
                    hits += 1
        return hits
        
    self_hits = count_hits(self_best_pixels, self_pack[1])
    knn_hits = count_hits(knn_best_pixels, knn_pack[1])
    neg_hits = count_hits(neg_best_pixels, neg_pack[1])

    # ══════════════════════════════════════════════════════════════════
    #  VISUALIZATION
    # ══════════════════════════════════════════════════════════════════
    print("\nGenerating visualization...")

    label_cmap = create_chaos_colormap()

    def show_img(ax, img_tensor, title, query_marker=False, target_markers=None):
        """Display a normalized image tensor."""
        plot = prep_for_plot(img_tensor)
        ax.imshow(plot)
        ax.set_title(title, fontsize=9, fontweight='bold')
        if query_marker:
            for qpx, qpy in query_pixels:
                ax.plot(qpx, qpy, 'r+', markersize=8, markeredgewidth=2)
        if target_markers is not None:
            for tpx, tpy in target_markers:
                ax.plot(tpx, tpy, 'g+', markersize=8, markeredgewidth=2)
        ax.axis('off')

    def show_label(ax, label_tensor, title, target_markers=None, marker_color='b+'):
        """Display a label map using the CHAOS colormap."""
        lbl = label_tensor.numpy()
        lbl_vis = np.zeros((*lbl.shape, 3), dtype=np.uint8)
        for c in range(len(label_cmap)):
            lbl_vis[lbl == c] = label_cmap[c]
        ax.imshow(lbl_vis)
        ax.set_title(title, fontsize=9, fontweight='bold')
        if target_markers is not None:
            for tpx, tpy in target_markers:
                ax.plot(tpx, tpy, marker_color, markersize=8, markeredgewidth=2)
        ax.axis('off')

    def show_corr(ax, corr_map, title, vmin=None, vmax=None):
        """Display a correlation map."""
        im = ax.imshow(corr_map, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                       interpolation='nearest')
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.axis('off')
        return im

    def find_best_match(corr_map):
        """Find the position of highest correlation."""
        idx = np.unravel_index(np.argmax(corr_map), corr_map.shape)
        return idx  # (h, w) in feature-map space

    # Global vmin/vmax for consistent color scales
    all_corr = np.concatenate([self_corr_map.ravel(), knn_corr_map.ravel(),
                                neg_corr_map.ravel()])
    global_vmin = np.percentile(all_corr, 2)
    global_vmax = np.percentile(all_corr, 98)

    fig = plt.figure(figsize=(24, 20))
    fig.suptitle("Sampled-pair correspondence (not cached kNN)\n"
                 "(Using BiomedCLIP 768-d Features, Exactly as in Training)",
                 fontsize=14, fontweight='bold', y=0.98)

    # ── ROW 1: Input images and labels ───────────────────────────────
    gs = fig.add_gridspec(5, 6, hspace=0.35, wspace=0.25,
                          top=0.93, bottom=0.03, left=0.03, right=0.97)

    ax = fig.add_subplot(gs[0, 0])
    show_img(ax, anchor_pack[0], f"Anchor Image\n(Patient {pid_a})", query_marker=True)

    ax = fig.add_subplot(gs[0, 1])
    show_label(ax, anchor_pack[1], f"Anchor Label\n(Patient {pid_a})")

    ax = fig.add_subplot(gs[0, 2])
    show_img(ax, self_pack[0], f"Same-patient slice Image\n(Patient {pid_a}, diff slice)")

    ax = fig.add_subplot(gs[0, 3])
    show_label(ax, self_pack[1], f"Same-patient slice Label\n(Patient {pid_a})", target_markers=self_best_pixels, marker_color='b+')

    ax = fig.add_subplot(gs[0, 4])
    show_img(ax, knn_pack[0], f"Random other-patient slice Image\n(Patient {pid_b})")

    ax = fig.add_subplot(gs[0, 5])
    show_label(ax, knn_pack[1], f"Random other-patient slice Label\n(Patient {pid_b})", target_markers=knn_best_pixels, marker_color='b+')

    # ── ROW 2: Self-Correspondence (same patient, diff slice) ────────
    # Title for the row
    ax = fig.add_subplot(gs[1, 0])
    show_img(ax, anchor_pack[0], "Anchor\n(query patch marked)", query_marker=True)

    ax = fig.add_subplot(gs[1, 1])
    show_img(ax, self_pack[0], f"Same-patient slice\n(same patient {pid_a})", target_markers=self_best_pixels)
    # Get the best match for the aggregate map for stats
    best_h, best_w = find_best_match(self_corr_map)

    ax = fig.add_subplot(gs[1, 2])
    im = show_corr(ax, self_corr_map,
                   "Same-patient slice: Backbone Features\n(anchor query → self-pair)",
                   global_vmin, global_vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 3])
    im = show_corr(ax, code_self_corr,
                   "Same-patient slice: Projected Code\n(what loss optimizes)",
                   None, None)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 4])
    im = show_corr(ax, self_self_map,
                   "Self-Self: Anchor → Anchor\n(intra-image reference)",
                   global_vmin, global_vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Stats
    ax = fig.add_subplot(gs[1, 5])
    ax.axis('off')
    stats_text = (
        f"Self-Correspondence Stats\n"
        f"{'─' * 30}\n"
        f"Pair: same patient, diff slice\n"
        f"Backbone corr mean: {self_corr_map.mean():.3f}\n"
        f"Backbone corr std:  {self_corr_map.std():.3f}\n"
        f"Backbone corr max:  {self_corr_map.max():.3f}\n"
        f"Code corr mean:     {code_self_corr.mean():.3f}\n"
        f"Best match at:      ({best_h}, {best_w})\n"
        f"GT Organ Hits:      {self_hits} / {num_points}\n"
        f"\nUsed in training as:\n"
        f"  pos_intra_loss (shift={cfg.pos_intra_shift})\n"
        f"  pos_inter_loss (shift={cfg.pos_inter_shift})\n"
        f"  Weight: intra={0.5}, inter={0.15}"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ── ROW 3: KNN Correspondence (different patient) ────────────────
    ax = fig.add_subplot(gs[2, 0])
    show_img(ax, anchor_pack[0], "Anchor\n(query patch marked)", query_marker=True)

    ax = fig.add_subplot(gs[2, 1])
    show_img(ax, knn_pack[0], f"Random other-patient slice\n(patient {pid_b}, same modality)", target_markers=knn_best_pixels)
    best_h_k, best_w_k = find_best_match(knn_corr_map)

    ax = fig.add_subplot(gs[2, 2])
    im = show_corr(ax, knn_corr_map,
                   "Random other-patient slice: Backbone Features\n(anchor query → KNN-pair)",
                   global_vmin, global_vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[2, 3])
    im = show_corr(ax, code_knn_corr,
                   "Random other-patient slice: Projected Code\n(what loss optimizes)",
                   None, None)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[2, 4])
    im = show_corr(ax, neg_corr_map,
                   f"Negative: Anchor → Patient {neg_pid}\n(should be low)",
                   global_vmin, global_vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Stats
    ax = fig.add_subplot(gs[2, 5])
    ax.axis('off')
    stats_text = (
        f"KNN-Correspondence Stats\n"
        f"{'─' * 30}\n"
        f"Pair: diff patient, same modality\n"
        f"Backbone corr mean: {knn_corr_map.mean():.3f}\n"
        f"Backbone corr std:  {knn_corr_map.std():.3f}\n"
        f"Backbone corr max:  {knn_corr_map.max():.3f}\n"
        f"Code corr mean:     {code_knn_corr.mean():.3f}\n"
        f"Best match at:      ({best_h_k}, {best_w_k})\n"
        f"GT Organ Hits:      {knn_hits} / {num_points}\n"
        f"\nNegative corr mean: {neg_corr_map.mean():.3f}\n"
        f"  (neg_inter shift={cfg.neg_inter_shift})\n"
        f"  Weight: neg={0.1}"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    # ── ROW 4: Contrastive Loss Mechanism ────────────────────────────
    # Show how pos_intra, pos_inter, neg_inter losses work
    # by sampling random coordinate pairs (just like the actual loss)

    coord_shape = [1, cfg.feature_samples, cfg.feature_samples, 2]
    coords1 = torch.rand(coord_shape, device=device) * 2 - 1
    coords2 = torch.rand(coord_shape, device=device) * 2 - 1

    # Sample features at random coordinates
    sampled_feats_a = sample(feats_anchor, coords1)
    sampled_code_a = sample(code_anchor, coords1)
    sampled_feats_s = sample(feats_self, coords2)
    sampled_code_s = sample(code_self, coords2)
    sampled_feats_k = sample(feats_knn, coords2)
    sampled_code_k = sample(code_knn, coords2)

    # Feature-space correlations (frozen backbone — this is fd)
    fd_self = tensor_correlation(norm(sampled_feats_a), norm(sampled_feats_s))
    fd_knn  = tensor_correlation(norm(sampled_feats_a), norm(sampled_feats_k))

    # Code-space correlations (learned projections — this is cd)
    cd_self = tensor_correlation(norm(sampled_code_a), norm(sampled_code_s))
    cd_knn  = tensor_correlation(norm(sampled_code_a), norm(sampled_code_k))

    # Reshape for visualization
    n = cfg.feature_samples
    fd_self_2d = fd_self.squeeze(0).reshape(n * n, n * n).cpu().numpy()
    fd_knn_2d  = fd_knn.squeeze(0).reshape(n * n, n * n).cpu().numpy()
    cd_self_2d = cd_self.squeeze(0).reshape(n * n, n * n).cpu().numpy()
    cd_knn_2d  = cd_knn.squeeze(0).reshape(n * n, n * n).cpu().numpy()

    ax = fig.add_subplot(gs[3, 0])
    im = ax.imshow(fd_self_2d, cmap='RdBu_r', aspect='auto')
    ax.set_title("fd (backbone corr)\nAnchor ↔ Self-pair", fontsize=8, fontweight='bold')
    ax.set_xlabel(f"{n}×{n} sampled positions", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[3, 1])
    im = ax.imshow(cd_self_2d, cmap='RdBu_r', aspect='auto')
    ax.set_title("cd (code corr)\nAnchor ↔ Self-pair", fontsize=8, fontweight='bold')
    ax.set_xlabel(f"{n}×{n} sampled positions", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[3, 2])
    im = ax.imshow(fd_knn_2d, cmap='RdBu_r', aspect='auto')
    ax.set_title("fd (backbone corr)\nAnchor ↔ KNN-pair", fontsize=8, fontweight='bold')
    ax.set_xlabel(f"{n}×{n} sampled positions", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[3, 3])
    im = ax.imshow(cd_knn_2d, cmap='RdBu_r', aspect='auto')
    ax.set_title("cd (code corr)\nAnchor ↔ KNN-pair", fontsize=8, fontweight='bold')
    ax.set_xlabel(f"{n}×{n} sampled positions", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Loss component visualization
    ax = fig.add_subplot(gs[3, 4:])
    ax.axis('off')

    # Compute actual loss values
    pos_intra_loss_val = -(cd_self.clamp(0) * (fd_self - cfg.pos_intra_shift)).mean().item()
    pos_inter_loss_val = -(cd_self.clamp(0) * (fd_self - cfg.pos_inter_shift)).mean().item()

    # For neg, use a permuted sample
    perm = torch.randperm(sampled_feats_a.shape[0], device=device)
    fd_neg = tensor_correlation(norm(sampled_feats_a), norm(sampled_feats_k[perm]))
    cd_neg = tensor_correlation(norm(sampled_code_a), norm(sampled_code_k[perm]))
    neg_inter_loss_val = -(cd_neg.clamp(0) * (fd_neg - cfg.neg_inter_shift)).mean().item()

    loss_text = (
        f"Contrastive Correlation Loss Mechanism\n"
        f"{'═' * 45}\n\n"
        f"loss = -cd.clamp(0) × (fd - shift)\n\n"
        f"Where:\n"
        f"  fd = backbone feature correlation (frozen)\n"
        f"  cd = projected code correlation (learned)\n"
        f"  shift = threshold separating pos/neg signal\n\n"
        f"{'─' * 45}\n"
        f"pos_intra (self-pair, shift={cfg.pos_intra_shift}):\n"
        f"  loss = {pos_intra_loss_val:+.4f}  ×  weight={0.5}\n"
        f"  → encourages cd>0 where fd>{cfg.pos_intra_shift}\n\n"
        f"pos_inter (self-pair, shift={cfg.pos_inter_shift}):\n"
        f"  loss = {pos_inter_loss_val:+.4f}  ×  weight={0.15}\n"
        f"  → wider positive region (lower threshold)\n\n"
        f"neg_inter (random pair, shift={cfg.neg_inter_shift}):\n"
        f"  loss = {neg_inter_loss_val:+.4f}  ×  weight={0.1}\n"
        f"  → pushes apart unrelated images\n"
        f"{'─' * 45}\n"
        f"Both pos_intra and pos_inter use SAME pair!\n"
        f"Only the shift differs (not self vs KNN)."
    )
    ax.text(0.05, 0.95, loss_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))

    # ── ROW 5: Summary diagram ───────────────────────────────────────
    ax = fig.add_subplot(gs[4, :])
    ax.axis('off')

    summary = (
        "TRAINING DATA FLOW SUMMARY\n"
        "═══════════════════════════════════════════════════════════════════════════════════════════════════════════════\n"
        "\n"
        "  ┌─────────────────┐     Self-pair: same patient, diff slice      ┌─────────────────┐\n"
        "  │  Anchor Image   │ ──────────────────────────────────────────── │  Self Image     │\n"
        "  │  (Patient A,    │     Used for pos_intra & pos_inter losses    │  (Patient A,    │\n"
        "  │   slice k)      │                                              │   slice k±Δ)    │\n"
        "  └────────┬────────┘                                              └─────────────────┘\n"
        "           │\n"
        "           │  KNN-pair: same modality, diff patient\n"
        "           │  (This is what img_pos actually is in the dataloader)\n"
        "           ▼\n"
        "  ┌─────────────────┐     Negative: random permutation within batch ┌─────────────────┐\n"
        "  │  KNN Image      │ ────────────────────────────────────────────  │  Neg Image      │\n"
        "  │  (Patient B,    │     Used for neg_inter loss                   │  (Patient C,    │\n"
        "  │   same modality)│                                               │   any slice)    │\n"
        "  └─────────────────┘                                               └─────────────────┘\n"
    )
    ax.text(0.02, 0.95, summary, transform=ax.transAxes,
            fontsize=7.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#fffff0', alpha=0.9))

    # ── Save ─────────────────────────────────────────────────────────
    output_dir = Path("/Users/navneetbavineni/STEGO/correspondence_visualization")
    output_dir.mkdir(exist_ok=True)

    save_path = output_dir / "accurate_correspondence.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n[OK] Saved to: {save_path}")

    plt.close(fig)

    # ── Also save a compact version with just the key comparisons ────
    fig2, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig2.suptitle("Same-patient vs random other-patient vs negative slices\n"
                  "(BiomedCLIP backbone features, query = red cross)",
                  fontsize=13, fontweight='bold')

    # Row 1: Self-correspondence
    show_img(axes[0, 0], anchor_pack[0], f"Anchor (Patient {pid_a})", query_marker=True)
    show_img(axes[0, 1], self_pack[0], f"Self: Same Patient {pid_a}\n(different slice)", target_markers=self_best_pixels)
    im = show_corr(axes[0, 2], self_corr_map, "Feature Correlation\n(Anchor → Self)",
                   global_vmin, global_vmax)
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    show_label(axes[0, 3], self_pack[1], f"Same-patient slice Ground Truth\n(Hits: {self_hits} | Misses: {num_points - self_hits})", target_markers=self_best_pixels, marker_color='b+')

    # Row 2: KNN-correspondence
    show_img(axes[1, 0], anchor_pack[0], f"Anchor (Patient {pid_a})", query_marker=True)
    show_img(axes[1, 1], knn_pack[0], f"KNN: Patient {pid_b}\n(same modality)", target_markers=knn_best_pixels)
    im = show_corr(axes[1, 2], knn_corr_map, "Feature Correlation\n(Anchor → KNN)",
                   global_vmin, global_vmax)
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    show_label(axes[1, 3], knn_pack[1], f"Random other-patient slice Ground Truth\n(Hits: {knn_hits} | Misses: {num_points - knn_hits})", target_markers=knn_best_pixels, marker_color='b+')

    axes[0, 0].set_ylabel("SELF\n(same patient,\ndiff slice)", fontsize=11,
                          fontweight='bold', color='blue')
    axes[1, 0].set_ylabel("KNN\n(diff patient,\nsame modality)", fontsize=11,
                          fontweight='bold', color='green')

    plt.tight_layout()
    save_path2 = output_dir / "correspondence_compact.png"
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved to: {save_path2}")

    plt.close(fig2)

    # ── Print explanation ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HOW CORRESPONDENCE WORKS IN YOUR MODEL")
    print("=" * 70)
    print("""
DATA PAIRING (in ContrastiveSegDataset.__getitem__):
  - img       = anchor slice (e.g., Patient A, slice k)
  - img_pos   = KNN positive (same modality, DIFFERENT patient)
  - img_aug   = augmented version of a SELF slice (same patient, diff slice)

CONTRASTIVE CORRELATION LOSS (in ContrastiveCorrelationLoss.forward):
  Uses (img, img_pos) pair for BOTH pos_intra and pos_inter:
  
  1. pos_intra_loss: -cd * (fd - 0.18)  <-- strict threshold
     -> Only positions with backbone similarity > 0.18 get positive signal
     -> Focuses on EXACT anatomical matches
     
  2. pos_inter_loss: -cd * (fd - 0.12)  <-- looser threshold
     -> Positions with backbone similarity > 0.12 get positive signal  
     -> Captures broader structural correspondence
     
  3. neg_inter_loss: -cd * (fd - 0.46)  <-- high threshold on RANDOM pair
     -> Since random pairs rarely have fd > 0.46, this term
       pushes cd NEGATIVE for unrelated positions
       -> Prevents the code from collapsing to uniform similarity

WHERE:
  fd = tensor_correlation(norm(backbone_features_A), norm(backbone_features_B))
       ^ FROZEN -- comes from BiomedCLIP, provides the "teacher" signal
  cd = tensor_correlation(norm(projected_code_A), norm(projected_code_B))  
       ^ LEARNED -- this is what the model optimizes
""")


if __name__ == "__main__":
    main()