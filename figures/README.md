# Figure sources and numerical reproduction

The README begins with `methods_overview.svg`, a conceptual introduction to the
three calibration strategies. `methods_overview_stacked.svg` supplies the
narrow-screen layout at the approved 640-pixel breakpoint. PNG fallbacks are
also supplied. The [overview record](METHODS_OVERVIEW.json) gives checksums
and provenance; both layouts preserve the same scientific wording.

`model_geometries.pdf`, `.svg` and `.png` are the approved revised schematic
from Supplementary Figure S1. They replace the earlier illustration while
keeping the same eight model inputs. [Source record](README_ILLUSTRATION.json).

The [complete enhanced manuscript figure collection](paper/README.md) includes
all twelve PDF/SVG/PNG sets, their computed/illustrative status and checksums.
The matching STIX fonts and original OFL licences are in [fonts/](fonts/README.md).

Numerical plotting functions remain in `src/cavity_sbi/result_plots.py`, with
compact inputs in `evidence/`. `python -m cavity_sbi reproduce --execute`
regenerates reference plots and CSV layers under `outputs/reproduction/`.
See the [panel map](../docs/REPRODUCTION_MAP.md) and [reader guide](../docs/START_HERE.md).
The reference plots reproduce numerical content; editable SVGs preserve the
paper's designed layout. Illustrative panels are not numerical evidence.
