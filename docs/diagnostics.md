# Predictive performance and marginal calibration

The workflow evaluates two properties separately. Forward predictive checks
measure attainment of the biological objective encoded by the pattern scores.
Frozen-estimator calibration checks measure uncertainty coverage under a stated
parameter-and-simulation reference law. Biological target attainment is the
primary performance criterion for this goal-conditioned example; coverage
provides complementary information about the learned parameter distribution.

| Command | Parameter vectors | Forward computation | Question |
| --- | --- | --- | --- |
| `evaluate-map` | Repeat the saved selected MAP | One knockout and one wild-type simulation per pair | Does this selected parameterization attain the target across stochastic simulations? |
| `predictive` | Draw from the frozen density conditioned on the ideal score | One simulation per condition and draw | How often does the learned parameter-and-simulation distribution attain the target? |
| `sbc` | Independent draws from the original uniform box prior | One simulation per condition and truth; then conditional posterior draws at that simulated score | Do marginal credible intervals cover known truths at their nominal rates under this reference law? |

All commands use the same RDME model, two conditions, fate classifiers and
49-component joint score over 36–48 h. `evaluate-map` reads the saved MAP;
it performs no new optimization. The other checks keep one final network
frozen and perform no training. The completed distribution-wide predictive
and SBC studies concern **geometry (5,3,1) and its final selected network**.
They are not results for all eight geometries.

## Preview and execution

From the repository root in the [pinned CPU environment](../environments/README.md):

```bash
python -m cavity_sbi evaluate-map --geometry 5_3_1 --run-dir outputs/map-5_3_1
python -m cavity_sbi predictive --geometry 5_3_1 --run-dir outputs/predictive-5_3_1
python -m cavity_sbi sbc --geometry 5_3_1 --run-dir outputs/sbc-5_3_1
```

These commands only preview settings and verify input bytes. They create no
run directory, load no neural artifact and perform no sampling or simulation.
Add **`--execute`** to the chosen command to start its work. Use separate run
directories for the three checks. Running any of them is an explicit choice;
installation, imports and report generation never start a diagnostic campaign.

The [reported profile](../config/diagnostics.json) specifies:

- MAP evaluation: **50 pairs**, or 50 simulations per biological condition.
- Density-wide prediction: **256 draws and pairs**, or **512 simulations**.
- SBC: **1,000 prior cases × 10,000 conditional draws**, **2,000 simulations**
  and **10 million posterior draws**, with central interval levels 50%, 80%,
  90% and 95%.

These are planned workloads; retries can add work. One paired simulation runs
its conditions in sequence. Allow memory for a full simulator instance, not
just its 49-value score. The [model guide](package_foundation.md) and `doctor`
report the geometry-dependent event-buffer requirements. Use one process per
worker and one Torch thread, as in the profile; no GPU or scheduler is required.

[diagnostics-small.json](../config/diagnostics-small.json) reduces case/draw
counts for checking interfaces. Supply it with
`--diagnostics config/diagnostics-small.json`. It does not change the biological
model or simulation horizon, and it is not the reported scientific study.

The eight released estimators are available without retraining. To check a
native estimator from a regenerated [training workflow](workflow.md), also
supply `--estimator path/to/selected.pkl --estimator-sha256 <SHA256>`. Specify
its actual geometry. The loader verifies bytes, native format, parameter order
and original prior bounds. A fixed-MAP check additionally requires its saved
MAP coordinates. No replacement network or simplified simulator is used.

## Prepare, divide work and resume

Each run first fixes its parameter bank. SBC preparation draws directly from
the model's prior; it does not deserialize a neural estimator. Density-wide
preparation needs the frozen estimator to draw at the ideal target. MAP
preparation reads its cached coordinates without sampling or optimization.

```bash
python -m cavity_sbi sbc --geometry 5_3_1 --run-dir outputs/sbc-5_3_1 --prepare-only --execute
```

After preparation, workers can process disjoint ranges of the same bank:

```bash
python -m cavity_sbi sbc --geometry 5_3_1 --run-dir outputs/sbc-5_3_1 --start 0 --stop 100 --execute
python -m cavity_sbi sbc --geometry 5_3_1 --run-dir outputs/sbc-5_3_1 --start 100 --stop 200 --execute
```

Continue until all indices in `[0,1000)` are complete. `--stop` is exclusive.
Use the same code, configuration, artifact and profile for every worker.
Processes must have disjoint ranges; case locks prevent concurrent writers.
The manifest binds outputs to their model, geometry, profile, artifact and
package-source hashes. A changed configuration requires a new run directory.

Completed observations and SBC statistics are checksum-verified and reused.
A saved forward observation is reused after a posterior-sampling timeout.
When retained, saved posterior draws can be reused if interval/rank calculation
was interrupted. The same pre-drawn parameter case is retried; no failed case
is replaced or silently omitted. Failure records remain in `errors/`.
An interruption inside a paired simulation retries that pair at the same
parameters and recorded streams; it does not resume inside the event solver.

The implementation rejects nonfinite/out-of-support parameters, invalid
scores and either condition failing to reach 48 h or exhausting its event
history. SBC sampling has a configurable per-case deadline (180 s in the
reported profile). Linux workers use process-local random streams. Available
diagnostic seeds are supplied; optional seeds may be omitted in custom profiles,
in which case the generated entropy is recorded. Statistical reproduction does
not require identical trajectories or identical retrained weights.

If a killed process leaves a lock or a partly written NPZ/JSON pair, inspect
and preserve the affected output, confirm its writer has stopped, then clear
only that stale lock or incomplete output before retrying the same case.
The commands do not overwrite an unexplained partial result automatically.

## Reports from saved results

```bash
python -m cavity_sbi diagnostic-report --run-dir outputs/predictive-5_3_1 --execute
python -m cavity_sbi diagnostic-report --run-dir outputs/sbc-5_3_1 --execute
```

Report generation reads all planned cases and writes `reports/summary.json`
and compact score/rank/coverage arrays. It loads no neural artifact, draws no
new parameters and performs no forward simulation. It refuses a complete
report if a planned case is missing, corrupt or linked to another input.
Every parameter and completed case is retained; earlier failed attempts remain
listed even after a successful retry. These compact arrays support figure
reconstruction without the molecular trajectories or full draw bank. Use
`plot-diagnostic --run-dir <run> --output-dir <new-plot-folder> --execute`
to plot your completed run. The [reader guide](START_HERE.md) explains the
steps; the [reproduction map](REPRODUCTION_MAP.md) links supplied evidence to
the paper's computed panels.

Predictive reports give the mean score, pointwise nearest quartiles, ensemble
meta-score **S**, monotone companion **R25**, and the fraction attaining the
complete joint target at all 49 times (tolerance `1e-12`). S remains the model
selection utility. S = 1 alone does not assert that every trajectory is perfect.
One stochastic pair per parameter draw measures performance under the joint
parameter-and-simulation distribution; it does not estimate every individual
parameter vector's reliability.

SBC reports contain randomized ranks for all **26 marginals**, KS distances
and p-values with **Holm correction**, central-interval counts and coverage,
and the simultaneous DKW reference band. Ties in finite-precision samples are
randomized among valid ranks; independent uniform jitter maps ranks to `[0,1)`.
Intervals use sample quantiles with NumPy's default linear interpolation,
distinct from the nearest quartiles used for S/R25. Each posterior sample is
conditioned on **its own simulated score**, not the ideal target. Sampling uses
the native sbi 0.19.2 `DirectPosterior.sample` with prior-support rejection.
The SBC observations also supply an original-prior predictive baseline without
additional simulations; this appears in `prior_predictive_reference`.

Global marginal SBC follows [Talts et al., *Validating Bayesian Inference
Algorithms with Simulation-Based Calibration*](https://arxiv.org/abs/1804.06788).
Familywise p-value adjustment follows [Holm, *A Simple Sequentially Rejective
Multiple Test Procedure*](https://www.jstor.org/stable/4615733). The implemented
scope is marginal calibration under the original prior, not a joint or local
calibration certificate and not calibration at the ideal target.

## Interpreting the completed checks

The completed 256-pair density-wide predictive ensemble has mean joint score
**0.8865**, pointwise median **1** throughout the evaluation window,
**S = R25 = 0.7564**, and **107/256 (41.8%)** complete-target pairs. The selected
MAP ensemble for this geometry has **S = R25 = 1**. These are different ensembles
and are reported separately. The corresponding prior predictive baseline has
mean **0.05353**, with **0/1,000** complete-target pairs.

The original-prior SBC check departs from its uniform-rank reference in all
26 marginals after Holm correction; nominal 90% interval coverage ranges from
**8.2% to 81.9%**. These findings describe uncertainty coverage of the unchanged
conditional estimator under that reference law. They are reported alongside
the predictive results: target attainment supports the intended biological
objective, while original-prior credible intervals require this qualification.
Neither result substitutes for the other.

## Verification scope

```bash
python -m cavity_sbi diagnostic-controls --execute
python -m unittest discover -s tests -v
```

The analytic controls use an exact conjugate Gaussian posterior and a
deliberately biased alternative. Package tests use synthetic molecular
observations and test-double estimators for execution paths, including restart,
failure accounting and conditioning. They do not fit a neural network or run
the biological event solver. Packaging additionally reconciles the statistics
against the completed scientific evidence, without repeating those campaigns.
Passing implementation controls is separate from a network's calibration result.

Generated simulation and posterior-draw banks remain in ignored run directories.
Their retention is configurable; they are excluded from the public repository
under the [data-access policy](data_access.md).
