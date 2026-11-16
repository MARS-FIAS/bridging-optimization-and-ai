# Regenerating simulations, fitting and selecting estimators

The CPU workflow has three commands: `generate`, `train` and `select`.
They **preview the planned work by default**. Add `--execute` to perform it.
Previews do not create outputs, sample parameters, simulate or fit.
Use the [pinned environment](../environments/README.md).

## Profiles and outputs

The [paper profile](../config/workflow.json) specifies five rounds, 100,000
parameter–score pairs per round, ten tasks and two independently initialized
fits per task. Each candidate MAP receives 50 paired evaluations. The original
250,000-draw posterior-summary and MAP-initialization settings are explicit.
Empty fitting/MAP option dictionaries use the pinned sbi defaults.

The [small profile](../config/workflow-small.json) uses 20 pairs, one task,
two fits, shorter optimizer limits and two evaluation pairs. It is an interface
example, not the configuration behind the paper. Select it with
`--workflow config/workflow-small.json` on every command in that run.

Each run covers one geometry. Use an ignored `outputs/` or `data/` directory,
or an external data directory. `run.json` snapshots the model, geometry,
workflow and code fingerprints; `workflow.json` copies the selected profile.
Changed code/configuration requires a new run directory. Seeds are optional;
new runs record their stream entropy so generated outputs can be resumed.
Each fit's TensorBoard logs are stored in its own `tensorboard/` directory
under the chosen run folder, alongside that fit's checkpoints and evaluation
records. Changing the log destination does not change fitting or MAP settings.

## 1. Generate paired conditions

Preview round zero:

```bash
python -m cavity_sbi generate --project . --geometry 5_3_1 --round 0 --run-dir outputs/5_3_1
```

Append `--execute` to generate. The first round samples the declared box prior
directly, without a trained network. Later rounds sample the preceding round's
selected estimator conditioned on the ideal 49-component all-ones target.

For multiple CPU workers, first fix the parameter bank in one process:

```bash
python -m cavity_sbi generate --project . --geometry 5_3_1 --round 0 --run-dir outputs/5_3_1 --prepare-only --execute
```

This samples parameters without forward simulation. Workers then process
disjoint ranges, for example:

```bash
python -m cavity_sbi generate --project . --geometry 5_3_1 --round 0 --run-dir outputs/5_3_1 --start 0 --stop 1000 --execute
```

A paper round has 100,000 pairs, or 200,000 condition-specific model calls.
Each pair uses the knockout and wild-type conditions, reaches the 48 h horizon,
and yields 49 score values over 36–48 h. Cases save parameters and joint scores;
raw molecular observations are retained when `keep_raw_trajectories` is true.
Generated data remain outside Git/GitHub.

Complete cases are checksum-verified and reused on restart. Failed IDs retain
their fixed parameters. Partial/corrupt files or changed proposals are reported
rather than silently replaced. Locks prevent overlapping writers. After an
interruption, remove a stale lock only after confirming its writer has stopped;
inspect partial outputs before retrying, and preserve failure records.

## 2. Fit and evaluate candidate MAPs

After all required generation cases are complete:

```bash
python -m cavity_sbi train --project . --geometry 5_3_1 --round 0 --task 0 --run-dir outputs/5_3_1
```

Append `--execute` to fit/evaluate. Repeat for task IDs 0–9 in the paper profile,
using separate processes or scheduler tasks. Each task builds two fresh
estimators. Round zero uses its own data; later rounds pool only the current
and immediately preceding rounds. Cached joint scores use the same model
objective and are converted to float32 for fitting, as in the original code.

The deliberate unweighted conditional fit is retained:
`append_simulations(..., proposal=None)`. MAPs are found at the ideal target.
Each is evaluated with 50 stochastic simulations per condition, forming 50
paired score trajectories. S selects candidates; R25 is a reporting companion.
There are 200 condition-specific evaluation calls per paper inference task,
in addition to data generation.

Completed fits and evaluation cases are reused. A trained checkpoint is saved
before evaluation, allowing interrupted evaluation to continue without fitting
that estimator again. Interruption during fitting restarts that unfinished fit;
optimizer-state continuation is not implemented. Complete checkpoints/cases
are never retrained or resimulated automatically.

## 3. Select the next-round density

After every task is complete:

```bash
python -m cavity_sbi select --project . --geometry 5_3_1 --round 0 --run-dir outputs/5_3_1 --execute
```

Selection retains the higher-S fit per task, then the higher-S task winner.
It uses the stored four-decimal S values and first maximum under ties, matching
the original rule. Selected posterior bytes are copied unchanged. The complete
goal-conditioned density supplies the next proposal; no new distribution is
constructed around the MAP.

Repeat generate → train → select for round indices 1–4, with the same profile,
geometry and directory. Use separate directories for other geometries.
Selection requires every task and checks quartiles, scores and estimator
checksums. It performs no simulation or neural training.

## Validation scope

The numerical definitions are preserved and checked separately. Workflow
validation exercises pairing, restarts, complete-case checks, adjacent pooling,
the None-proposal fitting call and stable selection using synthetic observations
and test-double estimators. It repeats no neural training or production
simulation campaign. Predictive/calibration run commands and compact evidence
integration follow in the diagnostic batch.
