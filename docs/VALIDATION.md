# Check your installation

Install the [plotting or full CPU environment](../environments/README.md), then
run the appropriate check from a source checkout. These checks are available
without the original data bank, a scheduler, GPU or LaTeX installation.

```bash
python tools/check_installation.py --mode plots --output-dir outputs/check-plots
python tools/check_installation.py --mode plots --output-dir outputs/check-plots --execute
```

The first command previews the steps. The second performs the dependency and
configuration/artifact-byte checks and regenerates all seven reference plots.
It leaves Torch/sbi unused. For the full CPU installation:

```bash
python tools/check_installation.py --mode cpu --output-dir outputs/check-cpu --execute
```

CPU mode additionally loads all eight supplied networks, requires the recorded
scientific versions, runs analytic controls and the unit suite, and previews
generation/training/selection/diagnostic commands. Previewed production stages
perform no simulation or fitting. Unit execution uses synthetic observations,
test-double estimators, small box-prior draws and native-model loading/cached MAP
access. A real inference-constructor check verifies the fit-local logging path
without fitting a network. It performs no biological event simulation, trained-network posterior
sampling or neural fitting. Passing these controls is separate from a network's
scientific calibration findings.

Use a fresh output directory each time. `validation.json` records commands and
exit codes; each step has separate stdout/stderr files. Generated plots and their
input/output hashes are in `reference-plots/`. The check stops at a failed step.
Local caches stay inside the chosen output directory. The script installs
nothing and sends no jobs or external messages.

## Tested scope

Both routes are checked in independent Python environments created without
`--system-site-packages`, from a clean tracked source copy. Dependencies are
installed from public package distributions rather than copied from a research
environment. The tested platform is **CPython 3.8.20, Linux x86_64**; the original
completed diagnostic environment used Python 3.8.18. Version pins and tested
transitive dependencies are in `environments/`.

| Check | Scope |
| --- | --- |
| Dependency consistency | `pip check` plus the full CPU runtime-version check |
| Portable imports and files | Installed package/configurations, artifact and evidence checksums; no private data paths needed |
| Artifact access | All eight trusted final networks load on CPU, without sampling or MAP optimization |
| Numerical reconstruction | All 400 retained candidates/40 selections, predictive scores/fates, all marginal SBC results, parameter references and figures |
| Workflow controls | Pairing, adjacent-round pooling, deliberate fitting configuration, ties, restart/failure handling and frozen-estimator conditioning |
| Analytic controls | Rank/coverage implementation and a known Gaussian conditional correlation |
| Full scientific campaigns | Generation/training/large diagnostics are documented and previewed; they are not repeated by installation verification |

The original numeric correlation matrix remains a separate artwork input; see
the [panel map](REPRODUCTION_MAP.md). Installation checks do not fabricate it
or compute a replacement on the biological network. The supplied scientific
results retain biological target attainment and the full original-prior coverage
findings, as explained in the [diagnostic guide](diagnostics.md).

For direct checks without the combined script:

```bash
python -m pip check
python -m cavity_sbi doctor --project . --load-models --strict-versions
python -m unittest discover -s tests -v
python -m cavity_sbi diagnostic-controls --execute
python -m cavity_sbi reproduce --output-dir outputs/reference-plots --execute
```

The neural/model-loading and full-unit commands require the CPU environment.
The supplied-evidence reproduction command also works in the smaller plotting
environment. See the [reader guide](START_HERE.md) for either reproduction route
and the [configuration glossary](CONFIGURATION.md) for meanings and units.
