# Compact reproduction evidence

Begin with the [reader guide](../docs/START_HERE.md), then run
`python -m cavity_sbi reproduce --execute`. The eight numerical input files total
**345,111 bytes**, excluding guides/manifests. These are derived results, without
raw molecular trajectories, full parameter-draw banks or bulk simulation banks.
Final neural estimators are separately supplied in [artifacts/](../artifacts/).

[manifest.json](manifest.json) records sizes, SHA-256 hashes, source hashes,
parameter/geometry order and units. NPZ loading uses `allow_pickle=False`
and does not deserialize a neural model.

| File | Content and axes | Recomputed result |
| --- | --- | --- |
| `selection_candidates.npz` | `quartiles`: 400×49×3, q25/q50/q75; geometry, round, task, stored S, selected flag and estimator hashes | S/R25, all 40 choices, ties and progression |
| `goal_predictive.npz` | `scores`: 256×49; `fate_counts`: case × condition × lower/upper group × time × EPI/PRE (256×2×2×49×2); group cell counts 5 and 4 | Predictive scores, full-target count, histograms and intended-fate means; independent score reconstruction from counts |
| `marginal_sbc.npz` | `ranks`, `rank_pit`: 1,000×26; `coverage`: case × interval level × parameter (1,000×4×26); levels 0.5/0.8/0.9/0.95, 10,000 draws per case | All marginal KS/Holm, coverage, DKW and rank plots |
| `prior_predictive.npz` | 1,000×49 joint-score trajectories from the completed original-prior SBC simulations; no true-parameter or molecular-trajectory bank | Prior predictive mean/quantiles and full-target count |
| `selected_MAP_parameters.csv` | Eight geometries × 26 native cached MAP coordinates, bounds and percentages | All 208 selected-MAP dots |
| `reference_density_quantiles_256.csv` | 26 marginal 2.5/25/50/75/97.5% quantiles of saved ideal-target draws, native and prior-range coordinates | Reference boxes/whiskers without the draw bank |
| `selected_final_artifacts.csv` | Geometry, public relative model path and SHA-256 | Link parameterizations to supplied networks |
| `expected_results.json` | Recorded aggregates from completed verified analyses | Check arithmetic; plots derive from input arrays, not this file |

Time is in **hours**, with 49 score times from **36 to 48 h**. Scores and rank
fractions are dimensionless; fate counts are physical ICM cells. Condition order
is knockout then wild type; fate order is EPI then PRE. Parameter coordinates
are native linear values with units in `config/model.json`. CSV percentages
are 0–100. Command/data round indices are 0–4; displayed rounds are 1–5.

The 400 candidates are retained task winners: eight geometries × five rounds ×
ten tasks. They are not all independently initialized fits. Each quartile
trajectory summarizes 50 paired evaluations. Selection preserves four-place S
and first-task ties; R25 is the reporting/sensitivity companion. No alternative
sequential workflow is simulated.

Both distribution-wide diagnostics use one final `(5,3,1)` network. Predictive
attainment and uncertainty coverage remain separate. Parameter boxes use 256
**ideal-target** draws, not SBC conditional draws. The correlation matrix is not
supplied; its [source boundary and recipe](../docs/REPRODUCTION_MAP.md) are explicit.

[data_sizes.json](data_sizes.json) separately records catalogued training and
supplied-model sizes. The [data policy](../docs/data_access.md) excludes bulk
banks and offers full training/testing data from the authors on request.
