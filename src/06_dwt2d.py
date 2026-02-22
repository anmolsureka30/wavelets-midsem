#!/usr/bin/env python3
"""
06_dwt2d.py — Run all 2D DWT experiments for Q2.
Generates Plot Sets F (decompositions), G (comparisons), H (energy), I (sparsity).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pywt
import matplotlib.pyplot as plt
from utils import (
    setup_plotting, save_figure, normalize_for_display,
    DATA_DIR, FIGURES_DIR, FILTER_NAMES, CMAP_DETAIL, CMAP_GRAY,
)

setup_plotting()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load images
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("LOADING IMAGES")
print("=" * 70)

img1 = np.load(os.path.join(DATA_DIR, 'image_cameraman.npy'))
img2 = np.load(os.path.join(DATA_DIR, 'image_exponential.npy'))

images = {
    'cameraman': {'data': img1, 'label': 'Cameraman (edge-rich)'},
    'exponential': {'data': img2, 'label': 'Exponential Surface (smooth)'},
}

for k, v in images.items():
    print(f"  {k}: shape={v['data'].shape}, range=[{v['data'].min():.1f}, {v['data'].max():.1f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Compute 2-level DWT for all (image, filter) combinations
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("COMPUTING 2-LEVEL DWT DECOMPOSITIONS")
print("=" * 70)

# decomp[img_name][filter_name] = dict with all sub-bands
decomp = {img_name: {} for img_name in images}

for img_name, img_info in images.items():
    image = img_info['data']
    for filt_name in FILTER_NAMES:
        coeffs = pywt.wavedec2(image, filt_name, level=2, mode='periodization')
        # coeffs = [LL2, (LH2, HL2, HH2), (LH1, HL1, HH1)]
        LL2 = coeffs[0]
        LH2, HL2, HH2 = coeffs[1]  # pywt: (cH, cV, cD) = (LH, HL, HH)
        LH1, HL1, HH1 = coeffs[2]

        decomp[img_name][filt_name] = {
            'LL2': LL2, 'LH2': LH2, 'HL2': HL2, 'HH2': HH2,
            'LH1': LH1, 'HL1': HL1, 'HH1': HH1,
        }
        print(f"  {img_name} + {filt_name}: LL2={LL2.shape}, detail1={LH1.shape}")


def compose_dwt_display(d):
    """Compose standard 2-level DWT display layout."""
    LL2 = d['LL2']
    LH2, HL2, HH2 = d['LH2'], d['HL2'], d['HH2']
    LH1, HL1, HH1 = d['LH1'], d['HL1'], d['HH1']

    h2, w2 = LL2.shape
    h1, w1 = LH1.shape

    display = np.zeros((2 * h1, 2 * w1))

    # Level 2 sub-bands (top-left quadrant)
    display[:h2, :w2] = normalize_for_display(LL2, is_ll=True)
    display[:h2, w2:2*w2] = normalize_for_display(HL2)
    display[h2:2*h2, :w2] = normalize_for_display(LH2)
    display[h2:2*h2, w2:2*w2] = normalize_for_display(HH2)

    # Level 1 sub-bands
    display[:h1, w1:] = normalize_for_display(HL1)
    display[h1:, :w1] = normalize_for_display(LH1)
    display[h1:, w1:] = normalize_for_display(HH1)

    return display


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Plot Set F: Individual 2-level DWT decompositions (10 figures)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING PLOT SET F: DWT DECOMPOSITION FIGURES")
print("=" * 70)

for img_name, img_info in images.items():
    for filt_name in FILTER_NAMES:
        d = decomp[img_name][filt_name]
        display = compose_dwt_display(d)

        h2, w2 = d['LL2'].shape
        h1, w1 = d['LH1'].shape

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(display, cmap='gray', aspect='equal')

        N_vm = int(filt_name[2:])
        ax.set_title(f'2-Level DWT: {img_info["label"]} — {filt_name.upper()} '
                      f'(Length {2*N_vm}, {N_vm} VM)')

        # Draw sub-band boundary lines
        ax.axhline(y=h1 - 0.5, color='yellow', lw=1, alpha=0.7)
        ax.axvline(x=w1 - 0.5, color='yellow', lw=1, alpha=0.7)
        ax.axhline(y=h2 - 0.5, color='cyan', lw=0.8, alpha=0.7,
                   xmin=0, xmax=w1 / (2*w1))
        ax.axvline(x=w2 - 0.5, color='cyan', lw=0.8, alpha=0.7,
                   ymin=1 - h1 / (2*h1), ymax=1)

        # Sub-band labels
        label_kwargs = dict(fontsize=8, color='yellow', ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5))
        ax.text(w2//2, h2//2, 'LL2', **label_kwargs)
        ax.text(w2 + w2//2, h2//2, 'HL2', **label_kwargs)
        ax.text(w2//2, h2 + h2//2, 'LH2', **label_kwargs)
        ax.text(w2 + w2//2, h2 + h2//2, 'HH2', **label_kwargs)
        ax.text(w1 + w1//2, h1//2, 'HL1', **label_kwargs)
        ax.text(w1//2, h1 + h1//2, 'LH1', **label_kwargs)
        ax.text(w1 + w1//2, h1 + h1//2, 'HH1', **label_kwargs)

        ax.set_xticks([])
        ax.set_yticks([])

        save_figure(fig, f'dwt2d_{img_name}_{filt_name}.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Plot Set G: Comparative sub-band analysis (2 figures)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING PLOT SET G: COMPARATIVE SUB-BAND GRIDS")
print("=" * 70)

subband_names = ['LH1', 'HL1', 'HH1', 'LH2', 'HL2', 'HH2']
subband_labels = [
    '$LH_1$ (Horiz Detail L1)', '$HL_1$ (Vert Detail L1)',
    '$HH_1$ (Diag Detail L1)', '$LH_2$ (Horiz Detail L2)',
    '$HL_2$ (Vert Detail L2)', '$HH_2$ (Diag Detail L2)',
]

for img_name, img_info in images.items():
    fig, axes = plt.subplots(6, 5, figsize=(18, 20))

    for row_idx, (sb_name, sb_label) in enumerate(zip(subband_names, subband_labels)):
        for col_idx, filt_name in enumerate(FILTER_NAMES):
            ax = axes[row_idx, col_idx]
            d = decomp[img_name][filt_name]
            sb = d[sb_name]

            vmax = np.max(np.abs(sb))
            if vmax > 0:
                im = ax.imshow(sb, cmap=CMAP_DETAIL, vmin=-vmax, vmax=vmax, aspect='equal')
            else:
                im = ax.imshow(sb, cmap=CMAP_GRAY, aspect='equal')

            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                N_vm = int(filt_name[2:])
                ax.set_title(f'{filt_name.upper()} ({N_vm} VM)', fontsize=10, fontweight='bold')

            if col_idx == 0:
                ax.set_ylabel(sb_label, fontsize=9)

    fig.suptitle(f'Sub-band Comparison: {img_info["label"]}',
                 fontsize=14, fontweight='bold')
    save_figure(fig, f'dwt2d_comparison_{img_name}.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Plot Set H: Energy distribution (2 figures)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING PLOT SET H: 2D ENERGY DISTRIBUTION")
print("=" * 70)

for img_name, img_info in images.items():
    fig, ax = plt.subplots(figsize=(10, 6))

    e_ll2 = []
    e_det2 = []
    e_det1 = []

    for filt_name in FILTER_NAMES:
        d = decomp[img_name][filt_name]
        total_energy = sum(np.sum(d[k]**2) for k in d)

        ell2 = np.sum(d['LL2']**2) / total_energy * 100
        edet2 = sum(np.sum(d[k]**2) for k in ['LH2', 'HL2', 'HH2']) / total_energy * 100
        edet1 = sum(np.sum(d[k]**2) for k in ['LH1', 'HL1', 'HH1']) / total_energy * 100

        e_ll2.append(ell2)
        e_det2.append(edet2)
        e_det1.append(edet1)

    x_pos = np.arange(len(FILTER_NAMES))
    bar_width = 0.5

    ax.bar(x_pos, e_ll2, bar_width, label='$LL_2$ (Approximation)',
           color='#2166ac')
    ax.bar(x_pos, e_det2, bar_width, bottom=e_ll2,
           label='Level 2 Details', color='#f4a582')
    bottom2 = [a + b for a, b in zip(e_ll2, e_det2)]
    ax.bar(x_pos, e_det1, bar_width, bottom=bottom2,
           label='Level 1 Details', color='#d6604d')

    ax.set_xlabel('Filter Bank')
    ax.set_ylabel('Energy (%)')
    ax.set_title(f'Energy Distribution: {img_info["label"]}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.upper() for n in FILTER_NAMES])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim([0, 105])

    save_figure(fig, f'energy_2d_{img_name}.pdf')

    # Print summary
    print(f"\n  {img_name}:")
    for i, filt_name in enumerate(FILTER_NAMES):
        print(f"    {filt_name}: LL2={e_ll2[i]:.1f}%, Det2={e_det2[i]:.1f}%, Det1={e_det1[i]:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Plot Set I: Sparsity analysis (1 figure)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING PLOT SET I: SPARSITY ANALYSIS")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(FILTER_NAMES))
bar_width = 0.35
colors_img = ['#2166ac', '#d6604d']

for img_idx, (img_name, img_info) in enumerate(images.items()):
    counts = []
    for filt_name in FILTER_NAMES:
        d = decomp[img_name][filt_name]
        all_detail = np.concatenate([
            d['LH1'].ravel(), d['HL1'].ravel(), d['HH1'].ravel(),
            d['LH2'].ravel(), d['HL2'].ravel(), d['HH2'].ravel(),
        ])
        threshold = 0.01 * np.max(np.abs(all_detail))
        n_significant = np.sum(np.abs(all_detail) > threshold)
        counts.append(n_significant)

    offset = (img_idx - 0.5) * bar_width
    ax.bar(x_pos + offset, counts, bar_width,
           label=img_info['label'], color=colors_img[img_idx], alpha=0.85)

ax.set_xlabel('Filter Bank')
ax.set_ylabel('Number of Significant Detail Coefficients')
ax.set_title('Sparsity Analysis: Detail Coefficients > 1% of Max')
ax.set_xticks(x_pos)
ax.set_xticklabels([n.upper() for n in FILTER_NAMES])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

save_figure(fig, 'sparsity_2d.pdf')

# Print summary
for img_name, img_info in images.items():
    print(f"\n  {img_name}:")
    for filt_name in FILTER_NAMES:
        d = decomp[img_name][filt_name]
        all_detail = np.concatenate([
            d['LH1'].ravel(), d['HL1'].ravel(), d['HH1'].ravel(),
            d['LH2'].ravel(), d['HL2'].ravel(), d['HH2'].ravel(),
        ])
        threshold = 0.01 * np.max(np.abs(all_detail))
        n_significant = np.sum(np.abs(all_detail) > threshold)
        n_total = len(all_detail)
        print(f"    {filt_name}: {n_significant}/{n_total} significant ({100*n_significant/n_total:.1f}%)")

print("\n06_dwt2d.py completed successfully.")
