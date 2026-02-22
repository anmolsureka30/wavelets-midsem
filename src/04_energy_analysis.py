#!/usr/bin/env python3
"""
04_energy_analysis.py — Compute energy distributions, generate tables and
Plot Set B (energy bar chart).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    setup_plotting, save_figure,
    DATA_DIR, TABLES_DIR,
    A_VALUES, TAYLOR_ORDERS, TAYLOR_LABELS, FILTER_NAMES,
)

setup_plotting()

ALL_ORDERS = TAYLOR_ORDERS + ['full']
ORDER_LABELS_SHORT = ['Ord 0', 'Ord 1', 'Ord 2', 'Ord 3', 'Ord 4', 'Full']

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load results and compute energy metrics
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("COMPUTING ENERGY DISTRIBUTIONS")
print("=" * 70)

# energy[filter][a][order] = dict with E_input, E_low, E_high, ratio, max_detail
energy = {name: {} for name in FILTER_NAMES}

for name in FILTER_NAMES:
    for a in A_VALUES:
        energy[name][a] = {}
        for order in ALL_ORDERS:
            order_str = str(order)
            path = os.path.join(DATA_DIR, f'results_{name}_a{a}_{order_str}.npz')
            data = np.load(path)

            x = data['x']
            # Use pywt coefficients for energy (ensures Parseval's theorem)
            cA = data['cA_pywt']
            cD = data['cD_pywt']

            E_input = np.sum(x ** 2)
            E_low = np.sum(cA ** 2)
            E_high = np.sum(cD ** 2)
            E_total = E_low + E_high
            ratio = E_high / E_total if E_total > 0 else 0.0
            max_detail = np.max(np.abs(cD))

            # Interior max detail (skip boundary coefficients)
            N_vm = int(name[2:])
            skip = max(N_vm + 1, 3)
            detail_manual = data['detail']
            # Use first half to avoid Taylor divergence at end
            d_first_half = detail_manual[:len(detail_manual) // 2]
            interior = d_first_half[skip:] if skip < len(d_first_half) else d_first_half
            max_interior = np.max(np.abs(interior)) if len(interior) > 0 else 0.0

            energy[name][a][order] = {
                'E_input': E_input,
                'E_low': E_low,
                'E_high': E_high,
                'E_total': E_total,
                'ratio': ratio,
                'max_detail': max_detail,
                'max_interior': max_interior,
            }

# Print summary for a = 0.5
print("\nEnergy ratio E_high/E_total (a = 0.5):")
print(f"{'Filter':<8}", end='')
for lbl in ORDER_LABELS_SHORT:
    print(f'{lbl:>12}', end='')
print()
for name in FILTER_NAMES:
    print(f'{name:<8}', end='')
    for order in ALL_ORDERS:
        r = energy[name][0.5][order]['ratio']
        print(f'{r:12.6f}', end='')
    print()

print("\nMax |interior detail coeff| (a = 0.5) — staircase pattern:")
print(f"{'Filter':<8}", end='')
for lbl in ORDER_LABELS_SHORT:
    print(f'{lbl:>12}', end='')
print()
for name in FILTER_NAMES:
    print(f'{name:<8}', end='')
    for order in ALL_ORDERS:
        md = energy[name][0.5][order]['max_interior']
        print(f'{md:12.6f}', end='')
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Generate tables/energy_distribution.tex
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING ENERGY DISTRIBUTION TABLE")
print("=" * 70)

lines = []
lines.append(r'\begin{table}[H]')
lines.append(r'\centering')
lines.append(r'\caption{Highpass energy ratio $E_{\mathrm{high}}/E_{\mathrm{total}}$ for $a = 0.5$}')
lines.append(r'\label{tab:energy_dist}')
lines.append(r'\small')
cols = 'l' + 'r' * 6
lines.append(r'\begin{tabular}{%s}' % cols)
lines.append(r'\toprule')
header = r'Filter & Ord 0 & Ord 1 & Ord 2 & Ord 3 & Ord 4 & Full \\'
lines.append(header)
lines.append(r'\midrule')
for name in FILTER_NAMES:
    vals = []
    for order in ALL_ORDERS:
        r = energy[name][0.5][order]['ratio']
        vals.append(f'{r:.6f}')
    lines.append(f'{name.upper()} & ' + ' & '.join(vals) + r' \\')
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')

tex_path = os.path.join(TABLES_DIR, 'energy_distribution.tex')
with open(tex_path, 'w') as fh:
    fh.write('\n'.join(lines))
print(f"  Saved: {tex_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Generate tables/max_detail_coeff.tex
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING MAX DETAIL COEFFICIENT TABLE")
print("=" * 70)

lines = []
lines.append(r'\begin{table}[H]')
lines.append(r'\centering')
lines.append(r'\caption{Maximum absolute interior detail coefficient $\max|d[n]|$ for $a = 0.5$ (boundary coefficients excluded)}')
lines.append(r'\label{tab:max_detail}')
lines.append(r'\small')
lines.append(r'\begin{tabular}{%s}' % cols)
lines.append(r'\toprule')
lines.append(header)
lines.append(r'\midrule')
for name in FILTER_NAMES:
    vals = []
    for order in ALL_ORDERS:
        md = energy[name][0.5][order]['max_interior']
        vals.append(f'{md:.6f}')
    lines.append(f'{name.upper()} & ' + ' & '.join(vals) + r' \\')
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')

tex_path = os.path.join(TABLES_DIR, 'max_detail_coeff.tex')
with open(tex_path, 'w') as fh:
    fh.write('\n'.join(lines))
print(f"  Saved: {tex_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Generate figures/energy_bars_a05.pdf (Plot Set B)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING PLOT SET B: ENERGY BAR CHART")
print("=" * 70)

a_plot = 0.5
n_filters = len(FILTER_NAMES)
n_orders = len(ALL_ORDERS)
x_pos = np.arange(n_filters)
bar_width = 0.12
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#000000']

fig, ax = plt.subplots(figsize=(12, 6))

for i, order in enumerate(ALL_ORDERS):
    ratios = [energy[name][a_plot][order]['ratio'] for name in FILTER_NAMES]
    offset = (i - n_orders / 2 + 0.5) * bar_width
    ax.bar(x_pos + offset, ratios, bar_width, label=ORDER_LABELS_SHORT[i],
           color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)

ax.set_xlabel('Filter Bank')
ax.set_ylabel('$E_{\\mathrm{high}} / E_{\\mathrm{total}}$')
ax.set_title(f'Highpass Energy Ratio by Filter and Taylor Order ($a = {a_plot}$)')
ax.set_xticks(x_pos)
ax.set_xticklabels([n.upper() for n in FILTER_NAMES])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

save_figure(fig, 'energy_bars_a05.pdf')

print("\n04_energy_analysis.py completed successfully.")
