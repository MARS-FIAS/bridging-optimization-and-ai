# Start here: choose what to reproduce

Run the commands below from the repository's top directory, the one containing
`README.md`, `config/` and `evidence/`. Use Python 3.8 on Linux for the recorded
software versions. There are two routes:

| Aim | Starting point | Resources |
| --- | --- | --- |
| Check the paper's numbers and rebuild numerical plots | Supplied compact evidence | Local CPU; no neural loading, new simulations or large datasets |
| Generate your own data, fit/select networks and run diagnostics | Model configuration and workflow commands | Substantial CPU time and memory; the original data bank is not required |

For a first visit, follow the compact-evidence route. It checks the calculations
before committing resources to simulation or training.

## 1. Rebuild the supplied results

Create a local environment and install the plotting dependencies:

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install -c environments/constraints-python38.txt '.[plots]'
python -m cavity_sbi reproduce --output-dir outputs/reproduction
```

The last command is a **preview**. It checks input bytes and prints the intended
work. To create the figures and tables:

```bash
python -m cavity_sbi reproduce --output-dir outputs/reproduction --execute
```

This route uses roughly 345 KB of compact numerical evidence, plus small
configuration files. It needs no Torch/sbi installation, GPU, cluster, simulation
bank or LaTeX compiler. Matplotlib writes PDF, editable SVG and PNG directly.
The README's geometry illustration is an unchanged paper asset supplied separately.

Open `outputs/reproduction/results.json` for recomputed findings, PDF/PNG files
for plots, and CSV files for their numerical layers. `reproduction_manifest.json`
records input/configuration/code hashes, software versions and exported files.
Use a new output directory for each execution; existing directories are not
overwritten. To rebuild one component:

```bash
python -m cavity_sbi reproduce --component predictive --output-dir outputs/predictive-review --execute
```

Components are `selection`, `predictive`, `calibration`, `parameters` and `all`.
The [paper reproduction map](REPRODUCTION_MAP.md) identifies individual panels.
These are reference numerical renderings; the paper's designed artwork has its
own layout. Expected checks include 400 retained candidates, 36/40 selection
agreement with R25 maximizers, all eight final-round agreements, 107/256
complete-target predictive pairs, and all marginal SBC results. Predictive
target attainment and uncertainty coverage are different results; see the
[diagnostic guide](diagnostics.md).

## 2. Find the relevant files

| Location | Contents | Use |
| --- | --- | --- |
| `config/model.json` | Ordered parameters, meanings, units, priors, conditions, times and scores | Understand the biological model |
| `config/geometries.json` | Eight fixed cell arrangements and physical/storage dimensions | Choose a geometry, such as `5_3_1` |
| `config/workflow*.json` | Simulation counts, rounds, fitting and selection settings | Plan generation/training |
| `config/diagnostics*.json` | Predictive/SBC counts and sampling settings | Plan a new diagnostic |
| `config/figures.json` | Parameter-panel membership and plot conventions | Understand reference plots |
| `evidence/` | Derived scores, fate counts, ranks, coverage and parameter summaries | Recompute supplied results |
| `artifacts/final/` | Eight native final neural estimators and a manifest | Run checks without retraining |
| `src/cavity_sbi/` | Importable scientific code and commands | Inspect or extend the implementation |
| `environments/` | Pinned CPU requirements | Install the neural/simulation workflow |
| `outputs/` | Your generated data, models, reports and plots | Keep outputs separate from supplied inputs |
| [`Script_Bank/`](../Script_Bank/) | Research reference listings with source hashes | Read for provenance; execute the portable commands instead |

The [model guide](package_foundation.md) explains biological variables, the two
conditions, physical cells versus simulator storage, and the two score levels.
The [evidence guide](../evidence/README.md) explains array axes and units.
The [configuration glossary](CONFIGURATION.md) lists all 26 parameter meanings,
units and prior bounds. Use the [installation checks](VALIDATION.md) to verify
the plotting or full CPU environment before running new scientific work.

## 3. Generate data and train/select estimators

Install the full [pinned CPU environment](../environments/README.md), then check it:

```bash
python -m cavity_sbi doctor --project . --load-models --strict-versions
```

This checks inputs and loads the eight supplied estimators without sampling,
simulation or training. Follow the [workflow guide](workflow.md) for execution.
Round 0 draws directly from the prior and needs no trained network. Later rounds
draw from the preceding selected density. Each parameter vector produces **two
simulations**, knockout and wild type, combined into one 49-value score.
`5_3_1` means layers of five, three and one physical ICM cells, with the first
layer adjacent to the cavity. Scores cover 36–48 h of a 48 h simulation.
The example conditions on an ideal biological target rather than a measured
experimental trajectory.

```text
generate round 0 → train every task → select round 0
generate round 1 → train every task → select round 1 → …
```

Example previews for the paper profile:

```bash
python -m cavity_sbi generate --geometry 5_3_1 --round 0 --run-dir outputs/workflow-5_3_1
python -m cavity_sbi train --geometry 5_3_1 --round 0 --task 0 --run-dir outputs/workflow-5_3_1
python -m cavity_sbi select --geometry 5_3_1 --round 0 --run-dir outputs/workflow-5_3_1
```

Add `--execute` to the stage to run after its inputs are complete. All ten tasks
must finish before selecting a paper-profile round. Command round indices are
**0–4**; plotted labels are **1–5**. The workflow guide explains pooling,
fitting, worker ranges, two fits per task and restartable outputs.
`workflow-small.json` reduces counts for an interface exercise; use the same
profile throughout its run.

The paper profile plans 100,000 parameter–score pairs per round per geometry. Across
five rounds, including candidate evaluation, that is 1,010,000 condition-specific
simulations per geometry. This is a configured workload, not measured runtime.
Small-count profiles retain the full model and its memory needs: main history
buffers alone are about 6.08 GB for `(5,3,1)`, rising to 17.48 GB for the largest
geometry. Use previews and `doctor` to plan workers. Separate CPU processes can
handle disjoint case ranges; the code requires no project-specific cluster account.

## 4. Check a supplied or newly fitted network

The [diagnostic guide](diagnostics.md) distinguishes repeated selected-MAP checks,
density-wide prediction and frozen-estimator marginal SBC. These commands use
the supplied final network by default, so retraining is optional:

```bash
python -m cavity_sbi predictive --geometry 5_3_1 --run-dir outputs/my-predictive
python -m cavity_sbi sbc --geometry 5_3_1 --run-dir outputs/my-sbc
```

The reported predictive profile uses 256 pairs (512 simulations); SBC uses 1,000
cases (2,000 simulations) and 10,000 conditional draws per case. A newly selected
native estimator can be supplied with `--estimator` and its checksum. These
commands also require `--execute` for work. They fix cases first, retain failures
and resume the same cases rather than replacing them.

After completion, regenerate the report and plot your own run:

```bash
python -m cavity_sbi diagnostic-report --run-dir outputs/my-predictive --execute
python -m cavity_sbi plot-diagnostic --run-dir outputs/my-predictive --output-dir outputs/my-predictive-plots --execute
```

Use analogous commands for `outputs/my-sbc`. Plotting refreshes the complete
saved-result summary and writes figures to the separate output directory;
it performs no new simulation, posterior sampling or training. Predictive plots
include intended fates when raw observations were retained. Otherwise they show
score, histogram and target-attainment panels. The supplied compact-evidence
predictive plot includes all four panels.

Conditional correlations are a separate density-grid calculation described in
the [reproduction map](REPRODUCTION_MAP.md). They are not SBC or a proof of
identifiability, and are not run by `reproduce`.

## Data access and repeatability

Bulk training/testing and diagnostic simulation banks and full posterior-draw
banks are excluded. Generation commands provide independent reproduction.
The [data policy](data_access.md) offers full training/testing data on request
and states the verified 290.02 GB training subtotal. Available seeds are recorded;
reproducing ensemble behavior does not require identical trajectories or weights.
