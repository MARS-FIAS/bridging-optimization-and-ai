# Provenance

## Current distribution

The supported executable implementation is `src/cavity_sbi/`, with its explicit
settings in `config/`. Version 0.5.0 passes independent plotting/full CPU
installation checks; see [the validation guide](docs/VALIDATION.md). The
development checkpoints below record which checks accompanied each stage.

Imported on 9 November 2026 from the author's recovered doctoral research files.

| Collection | Role | Imported working files |
| --- | --- | --- |
| `Panel` | Cavity model, simulator, scoring, inference and CPU templates | 6 drivers, 13 resources, 4 HPC templates |
| `Chapter_Draft_C` | Historical posterior/figure analysis | 10 programs |

`provenance/SOURCE_MANIFEST.json` records source collection/member names and
SHA-256 hashes. All 33 code files initially match their source byte-for-byte.
Subsequent changes belong in Git with their purpose documented.

The original imported hashes identify the retained research snapshots. Older
prototypes are not mixed into the selected executable implementation.

The primary analysis uses 400 retained candidate summaries. The eight selected
final estimators are supplied with exact checksums in the artifact manifest.

## Portable foundation, 15–16 November 2026

The model, RDME simulator, geometry and score definitions were extracted into
`src/cavity_sbi/` without editing their numerical bodies. Imports were made
relative, optional plotting became lazy and an unused Blender helper was
excluded. [CORE_EXTRACTION.json](provenance/CORE_EXTRACTION.json) records the
source-file and definition-level hashes. The public model wrapper validates
inputs and the 48 h event horizon. Native float32 parameter input matches the
executed diagnostic driver.

All 26 ordered priors, eight geometries, biological conditions and score settings
are explicit in `config/`. The eight selected final estimators are byte-identical
copies, with their own [manifest](artifacts/manifest.json). Their total size is
3,702,176 bytes. Public size summaries are derived from the catalogued tensor
inventory; they do not infer dataset contents from scheduler time limits.

Package checks cover wheel installation, eleven scientific/interface checks, CPU
loading of all eight estimators, and exact reproduction of S/R25 for all 400
retained candidate summaries. These checks run no new stochastic simulations,
posterior sampling or training. Dependencies were reused from a compatible
existing environment; full dependency installation into an empty environment
remains part of the release check. The larger workflow and diagnostic CLI
integration proceeds in separate batches.

## Generation, fitting and selection, 16 November 2026

Version 0.2.0 adds `generate`, `train` and `select`, with preview mode by default.
The paper/small profiles, fixed parameter banks, paired conditions, adjacent-round
pooling, unweighted conditional fitting, MAP ensembles and stored-S selection
are described in the [workflow guide](docs/workflow.md). Complete outputs are
verified and reused on restart. The selected density is copied unchanged into
the next-round proposal.

Twenty-five checks pass using synthetic observations and test-double estimators
for workflow execution. The installed selector reproduces all 40 recorded task
choices, and all 400 retained S/R25 values match exactly. No forward simulation
campaign or neural training was performed to validate packaging. The README's
geometry illustration is an unchanged copy from the paper's supplement; its
source/render hashes are in [the figure record](figures/README_ILLUSTRATION.json).

## Predictive and calibration diagnostics, 16 November 2026

Version 0.3.0 adds separate `evaluate-map`, `predictive` and `sbc` commands,
plus `diagnostic-report` for saved results and analytic implementation controls.
The [diagnostic guide](docs/diagnostics.md) states the reference laws, reported
settings, distinct interpretations, CPU requirements and restart behavior.
[DIAGNOSTIC_ADAPTATION.json](provenance/DIAGNOSTIC_ADAPTATION.json) identifies
the completed CPU runners and the adapted calculations by source hash.
Original model/simulator numerical bodies and selected artifacts are unchanged.
The public simulation wrapper adds separate per-condition streams for the
recorded diagnostic seed convention.

Forty-seven installed-package checks pass. Synthetic observations and test-double
estimators check execution/conditioning and failure/restart behavior; native
model loading and cached MAP access require no posterior draw. All retained
256-pair predictive statistics and 1,000-case, 26-parameter SBC statistics match
exactly, including KS/Holm, interval coverage and rank randomization controls.
No new biological simulations, neural posterior draws or neural training were
performed for packaging. Package dependencies were reused in a fresh package
environment; a full isolated dependency installation remains a release check.
Compact public evidence and plot reconstruction follow separately.

## Compact evidence and numerical reconstruction, 16 November 2026

Version 0.4.0 provides `reproduce`, saved-run `plot-diagnostic` and an explicit
conditional-correlation recipe. Eight numerical evidence files total 345,111
bytes: retained candidate quartiles/identities, predictive scores and cell-fate
counts, marginal ranks/coverage, original-prior predictive scores, cached MAPs
and fitted reference summaries. Their [manifest](evidence/manifest.json) records
source/file hashes and units; no molecular trajectory or full draw bank is copied.

The [reader guide](docs/START_HERE.md) separates supplied-evidence reconstruction
from new scientific execution, with folder navigation, inputs/outputs, commands
and resource needs. The [paper-panel map](docs/REPRODUCTION_MAP.md) identifies
computed and illustrative material and the original correlation-matrix input
boundary. The source's four parameter groups and matrix display order are retained.

Fifty-nine installed-package tests pass. The full compact-evidence command
regenerates seven reference plots in PDF/SVG/PNG and 14 CSV tables, with no
Torch/sbi import. All numerical layers reconcile with the independently issued
source tables: 400 candidates, 40 selections/progression rows, predictive means
and target counts, 520 rank-bin counts, 104 coverage values, 208 MAP coordinates
and 26 reference summaries. Predictive scores also reconstruct independently
from integer cell-fate counts. Four key PNGs were visually inspected.

The correlation recipe is checked against a known Gaussian density. It was
not executed on the biological network for packaging and does not assert
identity with the paper's original matrix. No new biological simulation,
trained-network posterior draw or neural fitting was performed. Package
dependencies were reused; full isolated dependency installation remains part
of release validation. Article artwork is separate from these numerical plots.

## Independent CPU installation validation, 16 November 2026

Version 0.5.0 is checked from clean tracked source copies in two empty Python
3.8 environments, without shared research site-packages. Public package
distributions supply 17 plotting dependencies and 58 full CPU dependencies;
their tested versions are constrained in `environments/`. The plotting route
leaves Torch/sbi absent. Both routes pass dependency checks and regenerate the
same seven reference plots/14 CSV layers, with numerical results identical to
the previously verified evidence. All eight native models load in the full
CPU environment, and all 60 unit controls pass.

The [portable check script](tools/check_installation.py),
[verification guide](docs/VALIDATION.md) and
[configuration glossary](docs/CONFIGURATION.md) support independent use.
Each fit's training logs now stay in its own run directory; the writer is
closed after fitting or an exception. This changes log routing, not scientific
fitting settings or the deliberate `proposal=None` configuration. A constructor
control verifies the destination without fitting a neural network.

[TESTED_SETUP.json](environments/TESTED_SETUP.json) records the execution scope,
versions and source hashes. Production simulation/training and large diagnostics
are previewed and documented rather than repeated. No new biological simulation,
trained-network posterior draw or neural fitting is performed for validation.
The original correlation matrix, final artwork and publication remain separate.

## Reference-source publication preparation, 16 November 2026

All 33 imported research files have retained original hashes. Twenty-one public
reference copies replace personal filesystem prefixes with `REFERENCE_*`
markers and/or add reference-only execution guards. Sixteen Python listings
stop before scientific execution, and all four shell templates exit before
worker launch. Twelve resource files remain byte-identical. The
[reference guide](Script_Bank/README.md) explains their role, and
[SOURCE_MANIFEST.json](provenance/SOURCE_MANIFEST.json) records both original
and published-copy hashes.

These are publication changes to reference listings. All 27 installed source
modules, the six extracted numerical modules, biological/inference configurations,
eight final estimators and compact numerical inputs retain their verified bytes.
The original model/simulator/geometry/score definitions remain unchanged.
The [source notice](NOTICE.md) preserves the earlier research-software MIT
notice and distinguishes it from separately installed dependencies.
