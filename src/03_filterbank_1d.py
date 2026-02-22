#!/usr/bin/env python3
"""
03_filterbank_1d.py — Run all 1D filter bank experiments for Q1.
Generates the centerpiece vanishing moments grid, signal flow, and effect-of-a figures.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pywt
import matplotlib.pyplot as plt
from math import factorial
from utils import (
    setup_plotting, save_figure,
    DATA_DIR, FIGURES_DIR,
    N_SAMPLES, A_VALUES, TAYLOR_ORDERS, TAYLOR_LABELS, FILTER_NAMES,
)

setup_plotting()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load filters
# ═══════════════════════════════════════════════════════════════════════════════
ALL_FILTERS = {}
for name in FILTER_NAMES:
    w = pywt.Wavelet(name)
    ALL_FILTERS[name] = {
        'h0': np.array(w.dec_lo, dtype=np.float64),
        'h1': np.array(w.dec_hi, dtype=np.float64),
        'g0': np.array(w.rec_lo, dtype=np.float64),
        'g1': np.array(w.rec_hi, dtype=np.float64),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Signal generation (self-contained)
# ═══════════════════════════════════════════════════════════════════════════════
def make_exponential(a, N):
    return np.exp(-a * np.arange(N, dtype=np.float64))

def make_taylor(a, N, order):
    n = np.arange(N, dtype=np.float64)
    x = np.zeros(N, dtype=np.float64)
    for k in range(order + 1):
        x += ((-a * n) ** k) / factorial(k)
    return x

def get_signal(a, order):
    """Get signal for given a and order ('full' or integer)."""
    if order == 'full':
        return make_exponential(a, N_SAMPLES)
    else:
        return make_taylor(a, N_SAMPLES, order)

# Order labels for iteration: 0,1,2,3,4,'full'
ALL_ORDERS = TAYLOR_ORDERS + ['full']


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Analysis and synthesis functions (manual convolution)
# ═══════════════════════════════════════════════════════════════════════════════
def analysis(x, h0, h1):
    """Analysis side: convolve + downsample by 2."""
    low_full = np.convolve(x, h0, mode='full')
    high_full = np.convolve(x, h1, mode='full')
    approx = low_full[::2]
    detail = high_full[::2]
    return approx, detail, low_full, high_full

def synthesis(approx, detail, g0, g1):
    """Synthesis side: upsample + filter + add."""
    up_approx = np.zeros(2 * len(approx))
    up_approx[::2] = approx
    up_detail = np.zeros(2 * len(detail))
    up_detail[::2] = detail
    low_recon = np.convolve(up_approx, g0, mode='full')
    high_recon = np.convolve(up_detail, g1, mode='full')
    min_len = min(len(low_recon), len(high_recon))
    return low_recon[:min_len] + high_recon[:min_len]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Run all filter bank experiments
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("RUNNING ALL 1D FILTER BANK EXPERIMENTS")
print("=" * 70)

# Storage: results[filter_name][a][order] = {approx, detail, x, recon_error, ...}
results = {name: {} for name in FILTER_NAMES}

for name in FILTER_NAMES:
    f = ALL_FILTERS[name]
    L = len(f['h0'])
    N_vm = int(name[2:])

    for a in A_VALUES:
        results[name][a] = {}

        for order in ALL_ORDERS:
            x = get_signal(a, order)

            # Manual analysis
            approx, detail, low_full, high_full = analysis(x, f['h0'], f['h1'])

            # Manual synthesis + perfect reconstruction check
            recon = synthesis(approx, detail, f['g0'], f['g1'])
            # The reconstructed signal has delay = L-1
            delay = L - 1
            if delay + len(x) <= len(recon):
                recon_segment = recon[delay:delay + len(x)]
                recon_error = np.max(np.abs(x - recon_segment))
            else:
                recon_segment = recon[delay:]
                trunc = min(len(x), len(recon_segment))
                recon_error = np.max(np.abs(x[:trunc] - recon_segment[:trunc]))

            # Also compute via pywt for energy analysis
            cA_pywt, cD_pywt = pywt.dwt(x, name, mode='periodization')

            results[name][a][order] = {
                'x': x,
                'approx': approx,
                'detail': detail,
                'low_full': low_full,
                'high_full': high_full,
                'recon': recon,
                'recon_error': recon_error,
                'cA_pywt': cA_pywt,
                'cD_pywt': cD_pywt,
            }

            # Save for energy analysis script
            order_str = str(order)
            np.savez(os.path.join(DATA_DIR, f'results_{name}_a{a}_{order_str}.npz'),
                     x=x, approx=approx, detail=detail,
                     cA_pywt=cA_pywt, cD_pywt=cD_pywt,
                     recon_error=np.array([recon_error]))

print(f"\n  Completed {len(FILTER_NAMES) * len(A_VALUES) * len(ALL_ORDERS)} filter bank runs")

# Print reconstruction error summary
print("\nPerfect Reconstruction Verification (manual convolution):")
for name in FILTER_NAMES:
    max_err = max(results[name][a][order]['recon_error']
                  for a in A_VALUES for order in ALL_ORDERS)
    print(f"  {name}: max reconstruction error = {max_err:.2e}")

# Also verify with pywt
print("\nPerfect Reconstruction Verification (pywt):")
for name in FILTER_NAMES:
    max_err_pywt = 0.0
    for a in A_VALUES:
        x = get_signal(a, 'full')
        cA, cD = pywt.dwt(x, name, mode='periodization')
        recon_pywt = pywt.idwt(cA, cD, name, mode='periodization')
        err = np.max(np.abs(x - recon_pywt[:len(x)]))
        max_err_pywt = max(max_err_pywt, err)
    print(f"  {name}: max reconstruction error = {max_err_pywt:.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Plot Set A: Vanishing Moments Grid (CENTERPIECE)
#    Uses manual convolution detail coefficients, showing only the first half
#    to avoid end-of-signal Taylor divergence boundary effects.
#    The onset boundary (n=0 jump) and interior pattern are clearly visible.
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING PLOT SET A: VANISHING MOMENTS GRIDS")
print("=" * 70)

for a in A_VALUES:
    fig, axes = plt.subplots(6, 5, figsize=(22, 18))

    for row_idx, order in enumerate(ALL_ORDERS):
        for col_idx, name in enumerate(FILTER_NAMES):
            ax = axes[row_idx, col_idx]
            r = results[name][a][order]
            N_vm = int(name[2:])

            # Use manual convolution detail coefficients (mode='full')
            # Show first half to avoid Taylor divergence at signal end
            d_full = r['detail']
            n_show = min(len(d_full), N_SAMPLES // 2 + 5)
            d = d_full[:n_show]
            n_d = np.arange(len(d))

            # Determine if this should be ~0 (order < VM count)
            is_annihilated = (order != 'full' and order < N_vm)

            color = '#2166ac' if not is_annihilated else '#4daf4a'
            markerline, stemlines, baseline = ax.stem(
                n_d, d, linefmt=f'{color}', markerfmt='.', basefmt='k-')
            plt.setp(stemlines, linewidth=0.6, color=color)
            plt.setp(markerline, markersize=2, color=color)

            # Compute interior max (skip boundary region)
            skip = max(N_vm + 1, 3)
            interior = d[skip:] if skip < len(d) else d
            max_int = np.max(np.abs(interior)) if len(interior) > 0 else 0
            max_all = np.max(np.abs(d)) if len(d) > 0 else 0

            # Annotate
            ann = f'int:{max_int:.2e}'
            ax.text(0.98, 0.95, ann, transform=ax.transAxes,
                    fontsize=7, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=6)

            # Row labels
            if col_idx == 0:
                label = TAYLOR_LABELS[row_idx]
                ax.set_ylabel(label, fontsize=9)

            # Column headers
            if row_idx == 0:
                ax.set_title(f'{name.upper()} ({N_vm} VM)',
                             fontsize=10, fontweight='bold', pad=4)
            # Bottom row x-label
            if row_idx == 5:
                ax.set_xlabel('$n$', fontsize=8)

    fig.suptitle(
        f'Detail Coefficients $d[n]$ — Vanishing Moments Demonstration ($a = {a}$)\n'
        f'Green stems: annihilated (order < VM count). Blue stems: non-zero.',
        fontsize=14, fontweight='bold', y=1.02)

    save_figure(fig, f'vanishing_moments_grid_a{a}.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Plot Set C: Effect of parameter a
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING PLOT SET C: EFFECT OF PARAMETER a")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes_flat = axes.flatten()

for idx, a in enumerate(A_VALUES):
    ax = axes_flat[idx]
    r = results['db3'][a]['full']
    d = r['cD_pywt']
    n_d = np.arange(len(d))

    markerline, stemlines, baseline = ax.stem(
        n_d, d, linefmt='b-', markerfmt='bo', basefmt='k-')
    plt.setp(stemlines, linewidth=0.8)
    plt.setp(markerline, markersize=3)

    max_d = np.max(np.abs(d))
    energy_d = np.sum(d ** 2)
    ax.set_title(f'$a = {a}$  |  max|d| = {max_d:.4f}  |  $E_d$ = {energy_d:.4f}')
    ax.set_xlabel('$n$')
    ax.set_ylabel('$d[n]$')
    ax.grid(True, alpha=0.3)

fig.suptitle('Detail Coefficients for DB3, Full Exponential — Effect of $a$',
             fontsize=14, fontweight='bold')
save_figure(fig, 'effect_of_a_db3.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Plot Set D: Complete signal flow (db3, a=0.5, full exp)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING PLOT SET D: SIGNAL FLOW")
print("=" * 70)

r = results['db3'][0.5]['full']
x = r['x']
low_full = r['low_full']
high_full = r['high_full']
approx = r['approx']
detail = r['detail']
recon = r['recon']
L = len(ALL_FILTERS['db3']['h0'])
delay = L - 1

fig, axes = plt.subplots(6, 1, figsize=(14, 20))

# 1. Input x[n]
ax = axes[0]
n_x = np.arange(len(x))
markerline, stemlines, baseline = ax.stem(n_x, x, linefmt='b-', markerfmt='b.', basefmt='k-')
plt.setp(stemlines, linewidth=0.5)
plt.setp(markerline, markersize=2)
ax.set_title('Input Signal $x[n] = e^{-0.5n}$')
ax.set_ylabel('$x[n]$')
ax.grid(True, alpha=0.3)

# 2. Lowpass filter output (before downsampling)
ax = axes[1]
n_lf = np.arange(len(low_full))
markerline, stemlines, baseline = ax.stem(n_lf, low_full, linefmt='g-', markerfmt='g.', basefmt='k-')
plt.setp(stemlines, linewidth=0.5)
plt.setp(markerline, markersize=2)
ax.set_title('Lowpass Filter Output $(h_0 * x)[n]$ (before downsampling)')
ax.set_ylabel('Amplitude')
ax.grid(True, alpha=0.3)

# 3. Highpass filter output (before downsampling)
ax = axes[2]
n_hf = np.arange(len(high_full))
markerline, stemlines, baseline = ax.stem(n_hf, high_full, linefmt='r-', markerfmt='r.', basefmt='k-')
plt.setp(stemlines, linewidth=0.5)
plt.setp(markerline, markersize=2)
ax.set_title('Highpass Filter Output $(h_1 * x)[n]$ (before downsampling)')
ax.set_ylabel('Amplitude')
ax.grid(True, alpha=0.3)

# 4. Downsampled approximation a[n]
ax = axes[3]
n_a = np.arange(len(approx))
markerline, stemlines, baseline = ax.stem(n_a, approx, linefmt='g-', markerfmt='go', basefmt='k-')
plt.setp(stemlines, linewidth=0.8)
plt.setp(markerline, markersize=3)
ax.set_title('Approximation Coefficients $a[n]$ (downsampled lowpass)')
ax.set_ylabel('$a[n]$')
ax.grid(True, alpha=0.3)

# 5. Downsampled detail d[n]
ax = axes[4]
n_d = np.arange(len(detail))
markerline, stemlines, baseline = ax.stem(n_d, detail, linefmt='r-', markerfmt='ro', basefmt='k-')
plt.setp(stemlines, linewidth=0.8)
plt.setp(markerline, markersize=3)
ax.set_title('Detail Coefficients $d[n]$ (downsampled highpass)')
ax.set_ylabel('$d[n]$')
ax.grid(True, alpha=0.3)

# 6. Reconstructed vs original
ax = axes[5]
recon_segment = recon[delay:delay + len(x)]
n_r = np.arange(len(x))
ax.plot(n_r, x, 'b-', lw=1.5, label='Original $x[n]$', alpha=0.7)
ax.plot(n_r, recon_segment, 'r--', lw=1.5, label='Reconstructed $\\hat{x}[n]$', alpha=0.7)
err = np.max(np.abs(x - recon_segment))
ax.set_title(f'Reconstruction: max error = {err:.2e}')
ax.set_xlabel('$n$')
ax.set_ylabel('Amplitude')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

fig.suptitle('Complete Signal Flow — DB3, $a = 0.5$, Full Exponential',
             fontsize=14, fontweight='bold')
save_figure(fig, 'signal_flow_db3_a05.pdf')

print("\n03_filterbank_1d.py completed successfully.")
