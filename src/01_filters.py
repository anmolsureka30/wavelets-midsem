#!/usr/bin/env python3
"""
01_filters.py — Define Daubechies filter banks (db1–db5), verify CQF relations,
generate coefficient tables and frequency response plots.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pywt
import matplotlib.pyplot as plt
from utils import (
    setup_plotting, save_figure,
    FIGURES_DIR, TABLES_DIR, DATA_DIR, FILTER_NAMES,
)

setup_plotting()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Get filter coefficients from PyWavelets (authoritative source)
# ═══════════════════════════════════════════════════════════════════════════════
def derive_cqf_filters(h0):
    """Derive h₁, g₀, g₁ from h₀ using CQF relations (PyWavelets convention).

    CQF Relations (PyWavelets sign convention):
        h₁[n] = (-1)^(n+1) · h₀[L-1-n]  (analysis highpass)
        g₀[n] = h₀[L-1-n]                (synthesis lowpass = time-reverse of h₀)
        g₁[n] = (-1)^n · h₀[n]           (synthesis highpass)
    """
    h0 = np.array(h0, dtype=np.float64)
    L = len(h0)
    h1 = np.array([(-1)**(n + 1) * h0[L - 1 - n] for n in range(L)])
    g0 = h0[::-1].copy()
    g1 = np.array([(-1)**n * h0[n] for n in range(L)])
    return h0, h1, g0, g1


ALL_FILTERS = {}

print("=" * 70)
print("FILTER BANK DEFINITIONS AND VERIFICATION")
print("=" * 70)

for name in FILTER_NAMES:
    N = int(name[2:])
    w = pywt.Wavelet(name)

    # Use PyWavelets as the authoritative source
    h0 = np.array(w.dec_lo, dtype=np.float64)
    h1 = np.array(w.dec_hi, dtype=np.float64)
    g0 = np.array(w.rec_lo, dtype=np.float64)
    g1 = np.array(w.rec_hi, dtype=np.float64)
    L = len(h0)

    # Verify CQF relations hold for pywt's coefficients
    _, h1_cqf, g0_cqf, g1_cqf = derive_cqf_filters(h0)
    cqf_h1_match = np.allclose(h1, h1_cqf, atol=1e-12)
    cqf_g0_match = np.allclose(g0, g0_cqf, atol=1e-12)
    cqf_g1_match = np.allclose(g1, g1_cqf, atol=1e-12)

    # Verify filter properties
    sum_h0 = np.sum(h0)
    norm_h0 = np.sum(h0 ** 2)

    # Power complementary: |H₀(e^{jω})|² + |H₀(e^{j(ω+π)})|² = 2
    omega = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    H0_freq = np.zeros(len(omega), dtype=complex)
    H0_shifted = np.zeros(len(omega), dtype=complex)
    for k in range(L):
        H0_freq += h0[k] * np.exp(-1j * omega * k)
        H0_shifted += h0[k] * np.exp(-1j * (omega + np.pi) * k)
    pc_val = np.abs(H0_freq) ** 2 + np.abs(H0_shifted) ** 2
    pc_dev = np.max(np.abs(pc_val - 2.0))

    print(f"\n{name} (Length {L}, {N} vanishing moment{'s' if N > 1 else ''}):")
    print(f"  CQF h₁ relation holds: {cqf_h1_match}")
    print(f"  CQF g₀ relation holds: {cqf_g0_match}")
    print(f"  CQF g₁ relation holds: {cqf_g1_match}")
    print(f"  Σh₀ = {sum_h0:.16f}  (√2 = {np.sqrt(2):.16f})")
    print(f"  Σh₀² = {norm_h0:.16f}  (expect 1.0)")
    print(f"  Power complementary max deviation: {pc_dev:.2e}")

    ALL_FILTERS[name] = {'h0': h0, 'h1': h1, 'g0': g0, 'g1': g1}

# Save filter data for downstream scripts
np.savez(os.path.join(DATA_DIR, 'filters.npz'),
         **{f'{name}_{ftype}': ALL_FILTERS[name][ftype]
            for name in FILTER_NAMES for ftype in ['h0', 'h1', 'g0', 'g1']})
print(f"\n  Saved filter data to {os.path.join(DATA_DIR, 'filters.npz')}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Generate tables/filter_coefficients.tex
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING FILTER COEFFICIENT TABLES")
print("=" * 70)

lines = []
for name in FILTER_NAMES:
    N = int(name[2:])
    f = ALL_FILTERS[name]
    L = len(f['h0'])

    lines.append(r'\begin{table}[H]')
    lines.append(r'\centering')
    lines.append(r'\caption{Filter coefficients for %s (length %d, %d vanishing moment%s)}' % (
        name.upper(), L, N, 's' if N > 1 else ''))
    lines.append(r'\label{tab:filters_%s}' % name)
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{c r r r r}')
    lines.append(r'\toprule')
    lines.append(r'$n$ & $h_0[n]$ (Dec LP) & $h_1[n]$ (Dec HP) & $g_0[n]$ (Rec LP) & $g_1[n]$ (Rec HP) \\')
    lines.append(r'\midrule')
    for i in range(L):
        lines.append(r'%d & %+.10f & %+.10f & %+.10f & %+.10f \\' % (
            i, f['h0'][i], f['h1'][i], f['g0'][i], f['g1'][i]))
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

tex_path = os.path.join(TABLES_DIR, 'filter_coefficients.tex')
with open(tex_path, 'w') as fh:
    fh.write('\n'.join(lines))
print(f"  Saved: {tex_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Generate figures/freq_responses.pdf
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING FREQUENCY RESPONSE PLOTS")
print("=" * 70)

NFFT = 1024
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes_flat = axes.flatten()

for idx, name in enumerate(FILTER_NAMES):
    ax = axes_flat[idx]
    f = ALL_FILTERS[name]
    N = int(name[2:])
    L = len(f['h0'])

    H0 = np.fft.fft(f['h0'], NFFT)
    H1 = np.fft.fft(f['h1'], NFFT)

    freq = np.linspace(0, 1, NFFT // 2 + 1)
    ax.plot(freq, np.abs(H0[:NFFT // 2 + 1]), 'b-', lw=1.5,
            label=r'$|H_0(e^{j\omega})|$')
    ax.plot(freq, np.abs(H1[:NFFT // 2 + 1]), 'r-', lw=1.5,
            label=r'$|H_1(e^{j\omega})|$')
    ax.set_title(f'{name.upper()} (Length {L}, {N} VM)')
    ax.set_xlabel(r'$\omega / \pi$')
    ax.set_ylabel(r'$|H(e^{j\omega})|$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.8])

# Hide empty 6th subplot
axes_flat[5].set_visible(False)

save_figure(fig, 'freq_responses.pdf')

print("\n01_filters.py completed successfully.")
