# napari-skin-remover

A [napari](https://napari.org) plugin for automated 3D brain extraction and microglia labelling from zebrafish confocal stacks using a MONAI 3D U-Net.

Developed at **FH Technikum Wien** — Artificial Intelligence & Data Science.

---

## What it does

Given a 3D confocal volume (TIF or IMS), the plugin provides three tabs:

- **Tab 1 — Skin Remover:** runs a trained MONAI U-Net to predict the brain mask, removes the skin, and saves `brain_mask.tif` + `brain_only.tif`
- **Tab 2 — Create Labels:** detects and labels individual microglia in 3D (Gaussian smooth → threshold → overlap-based 3D stitching → volume filter)
- **Tab 3 — Statistics:** computes up to 51 morphological, spatial, and intensity features per labelled cell and exports a CSV

---

## Environment setup (first time)

### 1. Clone the repository

```bash
git clone https://github.com/CTichy/napari-skin-remover.git
cd napari-skin-remover
```

### 2. Create the environment

**Linux (CUDA GPU):**
```bash
conda env create -f environment.yml
```

**Mac (CPU / Apple MPS):**
```bash
conda env create -f environment-mac.yml
```

### 3. Activate and launch

```bash
conda activate skin-seg
napari
```

Then: **Plugins → MONAI Skin-Remover**

---

## Updating (subsequent runs)

```bash
cd napari-skin-remover
git pull
conda env update --name skin-seg -f environment.yml --prune   # or environment-mac.yml on Mac
```

> **Linux only:** after every `conda env update`, restore the correct torch:
> ```bash
> /home/carlos-eduardo-tichy/anaconda3/envs/skin-seg/bin/pip install \
>   "torch==2.7.0+cu126" "torchvision==0.22.0+cu126" \
>   --index-url https://download.pytorch.org/whl/cu126
> ```
> (`conda env update` ignores `--extra-index-url` in environment.yml and resets torch to the wrong version.)

---

## Model file

The plugin requires a trained `.pth` checkpoint — **not included in this repo** (~220 MB).

**Download:**
[best_model_fullstack_v1_epoch460_dice9573.pth](https://cloud.technikum-wien.at/s/kYQ4qq3Jsn4xEyY)

Save it anywhere and point the plugin to it using the **Browse (...)** button in Tab 1. The path is remembered across sessions.

---

## Tab 1 — Skin Remover

### Workflow

1. **Open a file** — click "Open TIF / IMS file". All channels load as separate layers.
2. **Select the channel** to process by clicking its layer in the Layers panel.
3. **Browse to the model** `.pth` file if not auto-detected.
4. **Adjust MONAI Threshold** (default 0.30).
5. **Choose Background mode** (recommended: Option 2 — Remove globally, BG Threshold 0.60).
6. Click **Run Skin-Remover**.

### Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| MONAI Threshold | 0.30 | Sigmoid cutoff. Keep low — post-processing cleans the rest. |
| Erosion | 0 vox | Strips skin rim from `brain_only`. `brain_mask` always saved un-eroded. |
| Background mode | Off | Option 2 recommended before labelling |
| BG Threshold | 0.50 | Use 0.60 for microglia stacks |

### Output files

Saved in `<source_folder>/<source_stem>/`:

| File | Content |
|------|---------|
| `*_brain_only.tif` | Volume with everything outside the brain zeroed |
| `*_brain_only_NoBG.tif` | Same with global background also removed (Mode 2) |
| `*_brain_mask.tif` | Binary mask (0/255 uint8), un-eroded |

---

## Tab 2 — Create Labels

Detects individual microglia in 3D from a `brain_only` layer.

### Parameters

| Parameter | Default | Recommended |
|-----------|---------|-------------|
| Smooth σ XY | 1.0 | 1.5 |
| Smooth σ Z | 0.5 | 3.0 |
| Min overlap (%) | 10 | 10 |
| Min volume (vox) | 7500 | 7500 |

### Additional tools

- **Resort Labels** — renumber 1…N by size, centroid Z/Y/X
- **Split Label** — watershed split of a merged blob into N parts
- **Save Labels** — explicit file dialog (edit labels in napari before saving)

---

## Tab 3 — Statistics

Computes up to 51 features per labelled cell and exports a CSV.

- Select a Labels layer, optionally an Image layer (intensity stats) and a Shapes layer (brain region assignment)
- Choose output columns via the per-column checklist
- Select a description backend (Rule-based / Ollama / OpenAI / Claude API)
- Click **Generate Statistics**

CSV saved as `<stem>_statistics.csv` in the output folder.

---

## Typical voxel dimensions (zebrafish 4 dpf, 25× objective)

| Axis | Size |
|------|------|
| Z | 1.0 µm |
| X, Y | 0.174 µm |
| Anisotropy | ~5.75:1 |

---

## File format support

| Format | Channels | Metadata source |
|--------|----------|----------------|
| `.tif` / `.tiff` | single or multi-channel (C,Z,Y,X) | ImageJ tags or `*_metadata.txt` |
| `.ims` (Imaris) | all channels | embedded or `*_metadata.txt` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "no default model found" | Use Browse to select your `.pth` file |
| CUDA out of memory | Plugin falls back to CPU automatically |
| `.ims` files fail to open | `pip install imaris_ims_file_reader` |
| `EnvironmentFileNotFound` on `conda env update` | You must `cd` into the repo folder first |

---

## Contact

Carlos Tichy — ai24m016@technikum-wien.at  
FH Technikum Wien — Artificial Intelligence & Data Science
