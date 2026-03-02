"""
__main__.py — CLI entry point for napari_brain_peel.

Usage:
    python -m napari_brain_peel [file.tif]
    brain-peel [file.tif]
"""

import sys
from pathlib import Path


def main():
    print("=" * 70)
    print("MONAI BRAIN PEEL — Napari Plugin")
    print("=" * 70)

    # Defer heavy imports — only pay cost when actually launched
    from ._inference import DEFAULT_MODEL

    if not DEFAULT_MODEL.exists():
        print(f"WARNING: default model not found at:\n  {DEFAULT_MODEL}")
        print("  Use the Browse button to select a .pth checkpoint.")
    else:
        print(f"Default model: {DEFAULT_MODEL.name}")

    initial_path = None
    if len(sys.argv) > 1:
        initial_path = Path(sys.argv[1])
        if not initial_path.exists():
            print(f"ERROR: file not found: {initial_path}")
            sys.exit(1)

    import napari
    from ._widget import BrainPeelWidget

    viewer = napari.Viewer(title="MONAI Brain Peel")
    widget = BrainPeelWidget(viewer)

    if initial_path is not None:
        print(f"\nPre-loading: {initial_path.name}")
        widget.preload(initial_path)

    viewer.window.add_dock_widget(widget, area="right", name="MONAI Brain Peel")

    print("\nNapari viewer open — use the dock panel on the right.")
    print("=" * 70)

    napari.run()


if __name__ == "__main__":
    main()
