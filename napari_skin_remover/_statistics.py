"""
_statistics.py — Per-label morphological statistics + natural-language descriptions.

Statistics computed per label
------------------------------
  label               : label ID
  volume_vox          : voxel count
  volume_um3          : physical volume (µm³)
  centroid_z/y/x_vox  : centroid in voxels
  centroid_z/y/x_um   : centroid in µm
  bbox_dz/dy/dx_um    : bounding box size (µm)
  eq_diam_um          : equivalent sphere diameter (µm)
  axis1/2/3_um        : principal axis lengths (µm), axis1 = longest
  elongation          : axis1 / axis3  (1 = sphere, > 1 = elongated)
  principal_axis_dir  : dominant axis of elongation ('Z', 'Y', or 'X')
  solidity            : volume / convex_hull_volume  (1 = convex, < 1 = lobulated)
  extent              : volume / bounding_box_volume
  surface_area_um2    : mesh surface area via marching cubes (µm²)
  sphericity          : π^(1/3) × (6V)^(2/3) / A  (1 = perfect sphere)
  n_branches          : skeleton branch count  (requires skan)
  n_endpoints         : number of free-end endpoints
  mean_branch_len_um  : mean branch length (µm)
  description         : natural-language description

Description backends
--------------------
  rule   – built-in rule-based templates (always available, fully offline)
  ollama – local Ollama LLM (free, no API key; install from https://ollama.com)
  openai – OpenAI API (paid; requires API key from https://platform.openai.com)
  claude – Anthropic Claude API (paid; requires API key from https://console.anthropic.com)

Ollama setup (quick guide)
--------------------------
  1. Install Ollama:  https://ollama.com/download
  2. Pull a model:    ollama pull llama3
  3. It starts automatically; default endpoint is http://localhost:11434
  4. Select "Ollama (local)" in the plugin and enter the model name (e.g. llama3)

OpenAI setup
------------
  1. Create account at https://platform.openai.com
  2. Generate an API key under API Keys
  3. Select "OpenAI API" in the plugin, paste the key and choose a model
     (e.g. gpt-4o-mini for low cost, gpt-4o for best quality)

Anthropic Claude setup
----------------------
  1. Create account at https://console.anthropic.com
  2. Generate an API key under API Keys
  3. Select "Claude API" in the plugin, paste the key and choose a model
     (e.g. claude-haiku-4-5-20251001 for low cost, claude-sonnet-4-6 for quality)
"""

from __future__ import annotations

import json
import math
import urllib.request
import urllib.error
from typing import Any

import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────────

def _principal_axis_dir(inertia_tensor: np.ndarray) -> tuple[str, np.ndarray]:
    """
    Return (dominant_direction_label, eigenvectors) from the 3×3 inertia tensor.

    The eigenvector with the SMALLEST eigenvalue corresponds to the longest
    principal axis (least resistance to rotation around it).
    Returns the axis label ('Z', 'Y', 'X') whose component is largest in that
    eigenvector.
    """
    eigvals, eigvecs = np.linalg.eigh(inertia_tensor)
    # Smallest eigenvalue → longest axis
    longest_vec = eigvecs[:, 0]
    labels = ['Z', 'Y', 'X']
    dominant = labels[int(np.argmax(np.abs(longest_vec)))]
    return dominant, eigvecs


def _surface_area(binary: np.ndarray, scale_zyx: tuple) -> float:
    """Mesh surface area in µm² via marching cubes. Returns 0.0 on failure."""
    try:
        from skimage.measure import marching_cubes, mesh_surface_area
        # Pad by 1 so marching cubes can close open surfaces at boundaries
        padded = np.pad(binary.astype(np.uint8), 1)
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=scale_zyx)
        return float(mesh_surface_area(verts, faces))
    except Exception:
        return 0.0


def _skeleton_stats(binary: np.ndarray, scale_zyx: tuple) -> tuple[int, int, float]:
    """
    Return (n_branches, n_endpoints, mean_branch_len_um).
    Uses skan if available, otherwise returns (0, 0, 0.0).
    """
    try:
        from skimage.morphology import skeletonize
        import skan
        skeleton = skeletonize(binary)
        if not skeleton.any():
            return 0, 0, 0.0
        sk = skan.Skeleton(skeleton, spacing=scale_zyx, source_image=binary.astype(np.float32))
        branch_data = skan.summarize(sk, separator='-')
        if len(branch_data) == 0:
            return 0, 0, 0.0
        n_branches   = len(branch_data)
        n_endpoints  = int((branch_data['branch-type'] == 1).sum())
        mean_len     = float(branch_data['euclidean-distance'].mean())
        return n_branches, n_endpoints, mean_len
    except Exception:
        return 0, 0, 0.0


# ── Rule-based description ────────────────────────────────────────────────────

def _rule_based_description(row: dict) -> str:
    lbl        = row['label']
    vol        = row['volume_um3']
    elong      = row['elongation']
    axis_dir   = row['principal_axis_dir']
    spher      = row['sphericity']
    solid      = row['solidity']
    n_br       = row['n_branches']
    n_ep       = row['n_endpoints']
    br_len     = row['mean_branch_len_um']
    cz         = row['centroid_z_um']
    cy         = row['centroid_y_um']
    cx         = row['centroid_x_um']

    # Shape
    if spher > 0.85:
        shape = "spherical"
    elif spher > 0.70:
        if elong > 1.8:
            shape = f"rounded, elongated along {axis_dir}-axis ({elong:.1f}:1)"
        else:
            shape = "rounded"
    elif elong > 2.5:
        shape = f"elongated along {axis_dir}-axis ({elong:.1f}:1)"
    elif elong > 1.5:
        shape = f"moderately elongated along {axis_dir}-axis"
    else:
        shape = "compact, irregular"

    # Surface
    if solid > 0.90:
        surface = "smooth surface"
    elif solid > 0.75:
        surface = "slightly lobulated surface"
    else:
        surface = "lobulated/irregular surface"

    # Branches
    if n_br == 0:
        branch_str = "no branching detected"
    elif n_br <= 2:
        branch_str = f"{n_ep} protrusion(s)"
        if br_len > 0:
            branch_str += f" (mean {br_len:.1f} µm)"
    else:
        branch_str = f"{n_br} branches, {n_ep} endpoints"
        if br_len > 0:
            branch_str += f" (mean {br_len:.1f} µm)"

    return (
        f"Label {lbl}: {shape.capitalize()}, volume {vol:,.0f} µm³, "
        f"centroid Z={cz:.1f} Y={cy:.1f} X={cx:.1f} µm. "
        f"{surface.capitalize()}, sphericity {spher:.2f}, solidity {solid:.2f}. "
        f"{branch_str.capitalize()}."
    )


# ── Ollama backend ────────────────────────────────────────────────────────────

_STATS_PROMPT_TEMPLATE = """\
You are analyzing a single 3D cell from a zebrafish brain confocal microscopy image.
Given the morphological statistics below, write ONE concise sentence (max 40 words) \
describing the cell's shape, size, and notable features.

Label: {label}
Volume: {volume_um3:.0f} µm³
Elongation (longest/shortest principal axis): {elongation:.2f}
Dominant elongation axis: {principal_axis_dir}
Sphericity (1=sphere): {sphericity:.3f}
Solidity (1=convex): {solidity:.3f}
Surface area: {surface_area_um2:.0f} µm²
Branches: {n_branches}  |  Endpoints: {n_endpoints}  |  Mean branch length: {mean_branch_len_um:.1f} µm
Centroid: Z={centroid_z_um:.1f} Y={centroid_y_um:.1f} X={centroid_x_um:.1f} µm

Description:"""


def _ollama_description(row: dict, endpoint: str, model: str) -> str:
    prompt  = _STATS_PROMPT_TEMPLATE.format(**row)
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data.get("response", "").strip()
    except urllib.error.URLError as exc:
        return f"[Ollama error: {exc.reason}]"
    except Exception as exc:
        return f"[Ollama error: {exc}]"


# ── Remote API backend (OpenAI / Claude) ─────────────────────────────────────

def _openai_description(row: dict, api_key: str, model: str, api_url: str) -> str:
    url     = (api_url or "https://api.openai.com").rstrip("/") + "/v1/chat/completions"
    prompt  = _STATS_PROMPT_TEMPLATE.format(**row)
    payload = json.dumps({
        "model": model or "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 80,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"[OpenAI error: {exc}]"


def _claude_description(row: dict, api_key: str, model: str) -> str:
    prompt  = _STATS_PROMPT_TEMPLATE.format(**row)
    payload = json.dumps({
        "model": model or "claude-haiku-4-5-20251001",
        "max_tokens": 80,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["content"][0]["text"].strip()
    except Exception as exc:
        return f"[Claude error: {exc}]"


def _make_desc_fn(backend_config: dict | None):
    """Return a callable row_dict → description_str for the chosen backend."""
    if not backend_config or backend_config.get("backend", "rule") == "rule":
        return _rule_based_description

    backend = backend_config["backend"]

    if backend == "ollama":
        endpoint = backend_config.get("ollama_endpoint", "http://localhost:11434")
        model    = backend_config.get("ollama_model", "llama3")
        return lambda row: _ollama_description(row, endpoint, model)

    if backend == "openai":
        api_key = backend_config.get("api_key", "")
        model   = backend_config.get("api_model", "gpt-4o-mini")
        api_url = backend_config.get("api_url", "")
        return lambda row: _openai_description(row, api_key, model, api_url)

    if backend == "claude":
        api_key = backend_config.get("api_key", "")
        model   = backend_config.get("api_model", "claude-haiku-4-5-20251001")
        return lambda row: _claude_description(row, api_key, model)

    return _rule_based_description


# ── Main entry point ─────────────────────────────────────────────────────────

def compute_stats(
    labels: np.ndarray,
    scale_zyx: tuple,
    backend_config: dict | None = None,
) -> "Any":  # returns pd.DataFrame
    """
    Compute per-label morphological statistics and generate descriptions.

    Parameters
    ----------
    labels         : (Z, Y, X) int32 ndarray, 0 = background
    scale_zyx      : (z_um, y_um, x_um) — physical voxel size in µm, read from
                     the napari layer scale — never hardcoded
    backend_config : dict with keys:
                       backend  : 'rule' | 'ollama' | 'openai' | 'claude'
                       ollama_endpoint, ollama_model  (for 'ollama')
                       api_key, api_model, api_url    (for 'openai' / 'claude')

    Returns
    -------
    pd.DataFrame — one row per label, columns as documented at top of module
    """
    import pandas as pd
    from skimage.measure import regionprops

    sz, sy, sx = float(scale_zyx[0]), float(scale_zyx[1]), float(scale_zyx[2])
    voxel_vol  = sz * sy * sx          # µm³ per voxel

    desc_fn    = _make_desc_fn(backend_config)
    rows: list[dict] = []

    label_ids = [int(v) for v in np.unique(labels) if v > 0]
    n_total   = len(label_ids)
    print(f"   Computing statistics for {n_total} labels...")

    for idx, prop in enumerate(regionprops(labels), start=1):
        lbl = prop.label
        print(f"   [{idx}/{n_total}] label {lbl}...", end="\r")

        binary = (labels == lbl)

        # ── Basic regionprops ─────────────────────────────────────────────
        vol_vox = int(prop.area)
        vol_um3 = vol_vox * voxel_vol
        cz_vox, cy_vox, cx_vox = prop.centroid
        cz_um   = cz_vox * sz
        cy_um   = cy_vox * sy
        cx_um   = cx_vox * sx

        # Bounding box
        min_z, min_y, min_x, max_z, max_y, max_x = prop.bbox
        dz_um = (max_z - min_z) * sz
        dy_um = (max_y - min_y) * sy
        dx_um = (max_x - min_x) * sx

        # Equivalent sphere diameter: (6V/π)^(1/3)
        eq_diam_um = (6.0 * vol_um3 / math.pi) ** (1.0 / 3.0)

        # ── Principal axes via inertia tensor ─────────────────────────────
        try:
            it = prop.inertia_tensor
            eigvals, _ = np.linalg.eigh(it)
            eigvals = np.maximum(eigvals, 1e-10)
            # axis length ∝ 1/sqrt(eigenvalue) — normalised to physical units
            # Use the formula: axis_length ≈ sqrt(5/2 * (sum_others - self) / mass)
            # Simpler: use region's bounding dimensions as a proxy, eigvals for ratio
            axis_lengths_px = np.sqrt(5.0 / eigvals)
            axis_lengths_um = axis_lengths_px * np.array([sz, sy, sx])
            a1, a2, a3 = sorted(axis_lengths_um, reverse=True)
            elongation = float(a1 / max(a3, 1e-10))
            axis_dir, _ = _principal_axis_dir(it)
        except Exception:
            a1 = a2 = a3 = eq_diam_um / 2.0
            elongation = 1.0
            axis_dir   = 'Z'

        # ── Solidity + extent ─────────────────────────────────────────────
        try:
            solidity = float(prop.solidity)
        except Exception:
            solidity = 1.0

        bbox_vol = dz_um * dy_um * dx_um
        extent   = (vol_um3 / bbox_vol) if bbox_vol > 0 else 0.0

        # ── Surface area + sphericity ─────────────────────────────────────
        sa_um2    = _surface_area(binary, (sz, sy, sx))
        if sa_um2 > 0:
            sphericity = (math.pi ** (1.0 / 3.0)) * ((6.0 * vol_um3) ** (2.0 / 3.0)) / sa_um2
            sphericity = min(float(sphericity), 1.0)
        else:
            sphericity = 0.0

        # ── Skeleton / branch stats ───────────────────────────────────────
        n_branches, n_endpoints, mean_br_len = _skeleton_stats(binary, (sz, sy, sx))

        row = {
            "label":               lbl,
            "volume_vox":          vol_vox,
            "volume_um3":          round(vol_um3, 2),
            "centroid_z_vox":      round(cz_vox, 2),
            "centroid_y_vox":      round(cy_vox, 2),
            "centroid_x_vox":      round(cx_vox, 2),
            "centroid_z_um":       round(cz_um, 2),
            "centroid_y_um":       round(cy_um, 2),
            "centroid_x_um":       round(cx_um, 2),
            "bbox_dz_um":          round(dz_um, 2),
            "bbox_dy_um":          round(dy_um, 2),
            "bbox_dx_um":          round(dx_um, 2),
            "eq_diam_um":          round(eq_diam_um, 2),
            "axis1_um":            round(a1, 2),
            "axis2_um":            round(a2, 2),
            "axis3_um":            round(a3, 2),
            "elongation":          round(elongation, 3),
            "principal_axis_dir":  axis_dir,
            "solidity":            round(solidity, 4),
            "extent":              round(extent, 4),
            "surface_area_um2":    round(sa_um2, 2),
            "sphericity":          round(sphericity, 4),
            "n_branches":          n_branches,
            "n_endpoints":         n_endpoints,
            "mean_branch_len_um":  round(mean_br_len, 2),
        }
        rows.append(row)

    print(f"\n   Generating descriptions ({backend_config.get('backend', 'rule') if backend_config else 'rule'} backend)...")
    for row in rows:
        row["description"] = desc_fn(row)

    df = pd.DataFrame(rows)
    print(f"   Statistics complete — {len(df)} labels.")
    return df
