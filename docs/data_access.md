# Data access and statistical reproducibility

The repository distributes the model and analysis implementation, configurations,
eight final neural estimators and compact derived summaries as the reproduction
workflow is packaged. Large training, validation, testing and calibration
simulation banks, raw molecular trajectories and complete posterior-draw banks
are excluded from Git/GitHub. Regeneration commands and configurations provide
the route to producing these outputs in a local data directory.

**The full training and testing datasets are available from the authors on
request and are omitted here because of their substantial size.** Contact
[Michael Alexander Ramirez Sierra](mailto:ramirez-sierra@fias.uni-frankfurt.de).
Sharing is arranged separately from the repository; a clone does not download
those datasets.

## Recorded storage sizes

The following figures come from the read-only data-bank inventory captured on
9 November 2026. These are logical file sizes, not filesystem allocation or
compressed transfer estimates. **GB means 1,000,000,000 bytes.**

| Material used for this paper | Files | Exact bytes | GB |
| --- | ---: | ---: | ---: |
| Catalogued training parameter/trajectory tensors | 7,900 | 290,018,317,172 | 290.02 |
| Eight final estimators supplied in this repository | 8 | 3,702,176 | 0.00370 |

The training figure is a subtotal for the reported workflow. It does not include
later testing and diagnostic outputs, whose combined storage total is not
quoted here. The machine-readable [size record](../evidence/data_sizes.json)
provides the inventory date, checksum and scientific scope.

## What reproduction means here

The target is statistical reproduction of ensemble behavior and scientific
conclusions. Independent stochastic simulations and training runs can differ
in individual trajectories, selected numerical parameters and network weights.
Seeds are shared where available and may be supplied for new runs; recovering
every historical seed or reproducing a particular trajectory is not required.

Compact summaries support arithmetic checks and plot reconstruction. They are
distinguished from the excluded simulation banks. Final neural estimators allow
new forward assessments without retraining; they complement the generation and
training code rather than replacing it. Predictive target attainment and
uncertainty calibration remain separate scientific checks.
