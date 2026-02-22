#!/usr/bin/env python3
"""
05_images.py — Load/generate the two images for Q2 (2D DWT analysis).
Image 1: Edge-rich (cameraman or synthetic)
Image 2: Smooth exponential surface (2D analog of Q1 input)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from utils import setup_plotting, save_figure, DATA_DIR

setup_plotting()

IMG_SIZE = 256

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Image 1: Edge-rich image
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("GENERATING / LOADING IMAGES")
print("=" * 70)

try:
    from skimage.data import camera
    from PIL import Image as PILImage
    img_raw = camera()  # 512x512 uint8
    img1_pil = PILImage.fromarray(img_raw).resize((IMG_SIZE, IMG_SIZE), PILImage.LANCZOS)
    img1 = np.array(img1_pil, dtype=np.float64)
    img1_name = 'Cameraman'
    print(f"  Image 1: Cameraman from scikit-image ({IMG_SIZE}x{IMG_SIZE})")
except ImportError:
    print("  scikit-image not available, generating synthetic edge-rich image")
    img1 = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float64)
    # Smooth background gradient
    x = np.arange(IMG_SIZE, dtype=np.float64)
    y = np.arange(IMG_SIZE, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    img1 = 128 + 40 * np.sin(2 * np.pi * X / IMG_SIZE)
    # Add rectangles
    img1[30:100, 40:120] = 220
    img1[140:200, 60:180] = 50
    img1[50:80, 150:230] = 200
    # Add diagonal line
    for i in range(IMG_SIZE):
        j = min(i + 10, IMG_SIZE - 1)
        img1[i, j] = 255
    # Add circle
    cx, cy, r = 180, 80, 30
    mask = (X - cx)**2 + (Y - cy)**2 < r**2
    img1[mask] = 240
    img1_name = 'Synthetic Edges'
    print(f"  Image 1: Synthetic edge-rich image ({IMG_SIZE}x{IMG_SIZE})")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Image 2: Smooth exponential surface
# ═══════════════════════════════════════════════════════════════════════════════
x = np.arange(IMG_SIZE, dtype=np.float64)
y = np.arange(IMG_SIZE, dtype=np.float64)
X, Y = np.meshgrid(x, y)
img2 = 255.0 * np.exp(-0.02 * X) * np.exp(-0.02 * Y)
img2_name = 'Exponential Surface'
print(f"  Image 2: Smooth exponential surface ({IMG_SIZE}x{IMG_SIZE})")
print(f"    I(x,y) = 255 * exp(-0.02x) * exp(-0.02y)")
print(f"    Range: [{img2.min():.2f}, {img2.max():.2f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Save images
# ═══════════════════════════════════════════════════════════════════════════════
np.save(os.path.join(DATA_DIR, 'image_cameraman.npy'), img1)
np.save(os.path.join(DATA_DIR, 'image_exponential.npy'), img2)
# Save metadata
np.savez(os.path.join(DATA_DIR, 'image_metadata.npz'),
         img1_name=np.array(img1_name), img2_name=np.array(img2_name))
print(f"\n  Saved image data to {DATA_DIR}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Generate display figures
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING IMAGE DISPLAY FIGURES")
print("=" * 70)

# Image 1
fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(img1, cmap='gray', aspect='equal')
ax.set_title(f'Image 1: {img1_name} ({IMG_SIZE}x{IMG_SIZE})')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
save_figure(fig, 'input_image1.pdf')

# Image 2
fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(img2, cmap='gray', aspect='equal')
ax.set_title(f'Image 2: {img2_name} ({IMG_SIZE}x{IMG_SIZE})\n$I(x,y) = 255 \\cdot e^{{-0.02x}} \\cdot e^{{-0.02y}}$')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
save_figure(fig, 'input_image2.pdf')

print("\n05_images.py completed successfully.")
