"""
_background.py — Corner-based background processing.

Samples two left-side corners of the stack to estimate the background
intensity distribution.

Corner regions sampled (both on left side, X=0..corner_xy):
  Top-left    : Z=0..corner_z,  Y=0..corner_xy,        X=0..corner_xy
  Bottom-left : Z=0..corner_z,  Y=(H-corner_xy)..H,    X=0..corner_xy

Three processing modes:
  1. remove_outside_brain   — inference-guided: zero background pixels
                              outside the brain mask only
  2. remove_global          — zero all pixels in the full stack that fall
                              within the background range ± tolerance
  3. fill_random_background — fill all sub-background pixels everywhere
                              with random samples drawn from the corner pixels
"""

import numpy as np


def _sample_corners(volume, corner_xy=50, corner_z=101):
    """
    Return flat array of all pixel values from the two left-side corners.

    Top-left    : Z=0..corner_z,  Y=0..corner_xy,      X=0..corner_xy
    Bottom-left : Z=0..corner_z,  Y=H-corner_xy..H,    X=0..corner_xy
    """
    Z, Y, X = volume.shape
    z_end        = min(corner_z, Z)
    y_end        = min(corner_xy, Y)
    x_end        = min(corner_xy, X)
    y_start_bot  = max(0, Y - corner_xy)

    corner_tl = volume[:z_end, :y_end,       :x_end]
    corner_bl = volume[:z_end, y_start_bot:, :x_end]
    return np.concatenate([corner_tl.ravel(), corner_bl.ravel()])


def _bg_bounds(volume, corner_xy=50, corner_z=101, tolerance_pct=0.05):
    """Return (bg_values, bg_min, bg_max, low, high) for the given tolerance."""
    bg_values  = _sample_corners(volume, corner_xy, corner_z)
    bg_min     = float(bg_values.min())
    bg_max     = float(bg_values.max())
    data_range = float(volume.max()) - float(volume.min())
    tol        = data_range * (tolerance_pct / 100.0)
    return bg_values, bg_min, bg_max, bg_min - tol, bg_max + tol


# ------------------------------------------------------------------ #
# Mode 1 — inference-guided removal (outside brain only)
# ------------------------------------------------------------------ #

def remove_outside_brain(
    volume: np.ndarray,
    brain_mask: np.ndarray,
    corner_xy: int = 50,
    corner_z: int = 101,
    tolerance_pct: float = 0.05,
):
    """
    Zero background pixels that are OUTSIDE the brain mask.
    Pixels inside the brain are always preserved.

    Returns
    -------
    result    : copy of volume with outside-brain background pixels zeroed
    bg_min, bg_max, n_removed
    """
    _, bg_min, bg_max, low, high = _bg_bounds(volume, corner_xy, corner_z, tolerance_pct)

    print(f"   Background range (corners): [{bg_min:.1f}, {bg_max:.1f}]")
    print(f"   Tolerance: {tolerance_pct:+.3f}%  →  zeroing [{low:.1f}, {high:.1f}]")

    outside  = ~brain_mask.astype(bool)
    bg_mask  = (volume >= low) & (volume <= high)
    to_zero  = outside & bg_mask
    n_removed = int(to_zero.sum())
    print(f"   Removed {n_removed:,} outside-brain background voxels"
          f"  ({100.*n_removed/volume.size:.1f}% of stack)")

    result = volume.copy()
    result[to_zero] = 0
    return result, bg_min, bg_max, n_removed


# ------------------------------------------------------------------ #
# Mode 2 — global threshold removal (whole stack)
# ------------------------------------------------------------------ #

def remove_global(
    volume: np.ndarray,
    corner_xy: int = 50,
    corner_z: int = 101,
    tolerance_pct: float = 0.05,
):
    """
    Zero ALL pixels in the full stack within the background range ± tolerance.
    tolerance_pct can be negative (shrinks the range) or positive (expands it).
    Useful range: -1.0% to +1.0%.

    Returns
    -------
    result    : copy of volume with matching pixels set to 0
    bg_min, bg_max, n_removed
    """
    _, bg_min, bg_max, low, high = _bg_bounds(volume, corner_xy, corner_z, tolerance_pct)

    print(f"   Background range (corners): [{bg_min:.1f}, {bg_max:.1f}]")
    print(f"   Tolerance: {tolerance_pct:+.3f}%  →  zeroing [{low:.1f}, {high:.1f}]")

    mask = (volume >= low) & (volume <= high)
    n_removed = int(mask.sum())
    print(f"   Removed {n_removed:,} voxels globally"
          f"  ({100.*n_removed/volume.size:.1f}% of stack)")

    result = volume.copy()
    result[mask] = 0
    return result, bg_min, bg_max, n_removed


# ------------------------------------------------------------------ #
# Mode 3 — random background fill (whole stack)
# ------------------------------------------------------------------ #

def fill_random_background(
    volume: np.ndarray,
    corner_xy: int = 50,
    corner_z: int = 101,
):
    """
    Fill all pixels below the background minimum with random samples
    drawn from the actual corner pixel distribution.

    Uses random sampling (not mean) so the filled regions match the
    natural texture/noise of the scanner background.

    Returns
    -------
    result   : copy of volume with sub-background pixels randomly filled
    bg_min   : threshold used to detect sub-background pixels
    n_filled : number of voxels filled
    """
    bg_values = _sample_corners(volume, corner_xy, corner_z)
    bg_min    = float(bg_values.min())

    empty_mask = volume < bg_min
    n_filled   = int(empty_mask.sum())

    print(f"   Background — min: {bg_min:.2f}  (corners, {len(bg_values):,} samples)")
    print(f"   Filling {n_filled:,} sub-background voxels with random noise"
          f"  ({100.*n_filled/volume.size:.3f}% of stack)")

    result = volume.copy()
    if n_filled > 0:
        random_fill = np.random.choice(bg_values, size=n_filled, replace=True)
        result[empty_mask] = random_fill.astype(volume.dtype)
    return result, bg_min, n_filled
