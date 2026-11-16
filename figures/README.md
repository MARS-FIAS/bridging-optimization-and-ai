# Figure sources and numerical reproduction

`model_geometries.pdf` is an unchanged schematic illustration from the paper's
supplement, with a PNG preview for the README. Source/render hashes are in
[README_ILLUSTRATION.json](README_ILLUSTRATION.json).

Numerical plotting functions are in `src/cavity_sbi/result_plots.py`, with inputs
in `evidence/`. `python -m cavity_sbi reproduce --execute` writes PDF/SVG/PNG
plots and CSV layers under `outputs/reproduction/`, preserving paper artwork.
See the [panel map](../docs/REPRODUCTION_MAP.md) and [reader guide](../docs/START_HERE.md).

Main Figures 1–4 and 5B–D are illustrative; Figure 5A is computed. Geometry
diagrams are schematic model inputs. Reference plots rebuild numerical content;
the paper's designed layout remains a separate asset.
