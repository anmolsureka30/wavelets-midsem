#!/usr/bin/env python3
"""
02_signals.py — Generate all input signals (exponential + Taylor truncations)
and save them to data/ directory. Generate input signal visualization.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from utils import (
    setup_plotting, save_figure,
    DATA_DIR, N_SAMPLES, A_VALUES, TAYLOR_ORDERS, TAYLOR_LABELS,
)

setup_plotting()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Signal generation functions
# ═══════════════════════════════════════════════════════════════════════════════

def make_exponential(a, N):
    """Generate causal exponential signal x[n] = e^{-an} for n >= 0."""
    n = np.arange(N, dtype=np.float64)
    return np.exp(-a * n)


def make_taylor(a, N, order):
    """Generate Taylor-truncated approximation of e^{-an} up to given order.

    x_k[n] = Σ_{i=0}^{k} (-a*n)^i / i!
    """
    n = np.arange(N, dtype=np.float64)
    x = np.zeros(N, dtype=np.float64)
    for k in range(order + 1):
        x += ((-a * n) ** k) / factorial(k)
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Generate and save all signals
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("GENERATING INPUT SIGNALS")
print("=" * 70)

for a in A_VALUES:
    # Full exponential
    x_full = make_exponential(a, N_SAMPLES)
    path = os.path.join(DATA_DIR, f'signal_a{a}_full.npy')
    np.save(path, x_full)
    print(f"  a={a}, full exp: range [{x_full.min():.6f}, {x_full.max():.6f}]")

    # Taylor truncations
    for order in TAYLOR_ORDERS:
        x_taylor = make_taylor(a, N_SAMPLES, order)
        path = os.path.join(DATA_DIR, f'signal_a{a}_order{order}.npy')
        np.save(path, x_taylor)

print(f"\n  Saved {len(A_VALUES) * (len(TAYLOR_ORDERS) + 1)} signal files to {DATA_DIR}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Generate figures/input_signals.pdf
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING INPUT SIGNAL PLOTS")
print("=" * 70)

a_demo = 0.5
n = np.arange(N_SAMPLES)
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#000000']

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Top: Full range (clipped y-axis)
ax = axes[0]
for idx, order in enumerate(TAYLOR_ORDERS):
    x_t = make_taylor(a_demo, N_SAMPLES, order)
    ax.plot(n, x_t, color=colors[idx], lw=1.0, alpha=0.8,
            label=TAYLOR_LABELS[idx])
x_full = make_exponential(a_demo, N_SAMPLES)
ax.plot(n, x_full, color=colors[5], lw=2.0, ls='--',
        label=TAYLOR_LABELS[5])
ax.set_ylim([-3, 3])
ax.set_xlabel('$n$ (sample index)')
ax.set_ylabel('$x[n]$')
ax.set_title(f'Taylor Approximations of $e^{{-{a_demo}n}}$ — Full Range (clipped)')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', lw=0.5)

# Bottom: Zoomed view (first 25 samples)
ax = axes[1]
n_zoom = 25
for idx, order in enumerate(TAYLOR_ORDERS):
    x_t = make_taylor(a_demo, N_SAMPLES, order)
    ax.stem(n[:n_zoom], x_t[:n_zoom], linefmt=colors[idx], markerfmt='o',
            basefmt=' ', label=TAYLOR_LABELS[idx])
    # Make stem markers smaller
    ax.get_children()
x_full = make_exponential(a_demo, N_SAMPLES)
ax.plot(n[:n_zoom], x_full[:n_zoom], color=colors[5], lw=2.5, ls='--',
        label=TAYLOR_LABELS[5], zorder=10)
ax.set_xlabel('$n$ (sample index)')
ax.set_ylabel('$x[n]$')
ax.set_title(f'Taylor Approximations of $e^{{-{a_demo}n}}$ — Zoomed ($n = 0$ to ${n_zoom-1}$)')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

save_figure(fig, 'input_signals.pdf')

print("\n02_signals.py completed successfully.")
