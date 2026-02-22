"""
Shared constants, plotting style, and utility functions for EE678 Wavelets project.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# ── Directory paths ──────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
TABLES_DIR = os.path.join(PROJECT_DIR, 'tables')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')

# ── Plotting style constants ────────────────────────────────────────────────
FIG_SINGLE = (8, 5)
FIG_GRID = (16, 12)
DPI = 150
FONT_SIZE = 11
LABEL_SIZE = 12
TITLE_SIZE = 13

CMAP_DETAIL = 'RdBu_r'
CMAP_GRAY = 'gray'

# ── Signal / experiment constants ────────────────────────────────────────────
N_SAMPLES = 128
A_VALUES = [0.1, 0.5, 1.0, 2.0]
TAYLOR_ORDERS = [0, 1, 2, 3, 4]
FILTER_NAMES = ['db1', 'db2', 'db3', 'db4', 'db5']
TAYLOR_LABELS = [
    'Order 0 (const)',
    'Order 1 (linear)',
    'Order 2 (quadratic)',
    'Order 3 (cubic)',
    'Order 4 (quartic)',
    'Full exponential',
]


def setup_plotting():
    """Configure matplotlib rcParams for consistent styling."""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': LABEL_SIZE,
        'axes.titlesize': TITLE_SIZE,
        'axes.titleweight': 'bold',
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'figure.figsize': FIG_SINGLE,
    })


def save_figure(fig, name):
    """Save a figure to the figures/ directory and close it."""
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def normalize_for_display(subband, is_ll=False):
    """Normalize a sub-band for image display."""
    if is_ll:
        s = subband.copy()
        smin, smax = s.min(), s.max()
        if smax - smin > 0:
            s = (s - smin) / (smax - smin) * 255
        return s
    else:
        s = np.abs(subband)
        if s.max() > 0:
            s = s / s.max() * 255
        return s
