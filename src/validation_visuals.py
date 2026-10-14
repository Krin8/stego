"""Label-blind fixed slice selection and evaluation-only segmentation panels."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from chaos_preprocessing import MEAN, STD


def fixed_validation_indices(records, count=4):
    patients = sorted({r['patient_id'] for r in records}, key=int)
    if len(patients) < count:
        raise ValueError(f'Fixed panel needs {count} patients, found {len(patients)}')
    chosen = [patients[i] for i in np.linspace(0, len(patients) - 1, count, dtype=int)]
    return [indices[len(indices) // 2] for patient in chosen
            for indices in [[i for i, r in enumerate(records) if r['patient_id'] == patient]]]


def plot_fixed_slices(outputs, lookup):
    outputs = sorted(outputs, key=lambda x: x['ind'])
    colors = np.asarray(plt.get_cmap('tab10').colors)
    lookup = lookup.detach().cpu().numpy()
    fig, axes = plt.subplots(4, len(outputs), figsize=(12, 12), squeeze=False)
    for col, entry in enumerate(outputs):
        rgb = np.clip(entry['img'].permute(1, 2, 0).numpy() * np.asarray(STD) + np.asarray(MEAN), 0, 1)
        raw = entry['cluster_preds'].numpy()
        truth = entry['label'].numpy() == 1
        mapped = lookup[raw] == 1
        gt_overlay, pred_overlay = rgb.copy(), rgb.copy()
        gt_overlay[truth] = .55 * rgb[truth] + .45 * np.array([1, 0, 0])
        pred_overlay[mapped] = .55 * rgb[mapped] + .45 * np.array([1, 0, 0])
        for row, image in enumerate([rgb, gt_overlay, colors[raw % len(colors)], pred_overlay]):
            axes[row, col].imshow(image)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        axes[0, col].set_title(entry['sample_id'])
    for row, label in enumerate(['CT', 'Ground truth overlay', f'Raw clusters (K={len(lookup)})', 'Mapped liver overlay']):
        axes[row, 0].set_ylabel(label)
    fig.legend(handles=[Patch(color=colors[i % len(colors)], label=str(i)) for i in range(len(lookup))],
               loc='lower center', ncol=len(lookup), title='Raw cluster IDs')
    fig.tight_layout(rect=(0, .05, 1, 1))
    return fig
