# CPU environment

The retained neural estimators use **sbi 0.19.2 and PyTorch 1.11.0**. These
versions are confirmed by the original project requirements and run environment.
The completed CPU diagnostics used Python 3.8.18; the package targets the Python
3.8 series on Linux. Independent installation checks use CPython 3.8.20 on
Linux x86_64. [constraints-python38.txt](constraints-python38.txt) records all
58 tested CPU dependencies, including the shared plotting subset. Constraints
restrict versions of requested dependencies; they do not install neural packages
when using the plotting-only route.

From the source checkout:

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install -r environments/requirements-toolchain.txt
python -m pip install -r environments/requirements-legacy-cpu.txt
python -m pip install .
python -m cavity_sbi doctor --project . --load-models --strict-versions
```

The requirements select the CPU build of torch 1.11.0 and pin the numerical
and flow libraries. The extra wheel index follows the official
[previous-version PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v1110).
No CUDA device or GPU allocation is required. The recorded
[diagnostic environment freeze](diagnostic-cpu-freeze.txt) is supporting
provenance; the requirements file lists the primary scientific dependencies.

For a configuration/artifact-byte check, omit `--load-models`. The basic
preflight uses the standard library and reports missing or differing packages.
`--strict-versions` makes a discrepancy an explicit preflight failure. Torch
build suffixes such as `+cpu` are reported and ignored only when comparing
base version numbers. The loading preflight checks the imported libraries'
runtime versions and records installation metadata separately. Loading does
not draw parameters or start a simulation.

For numerical plots without neural loading or simulation, install only:

```bash
python -m pip install -c environments/constraints-python38.txt '.[plots]'
```

This route uses 17 scientific/plotting dependencies and leaves Torch/sbi absent.
The [reader guide](../docs/START_HERE.md) shows the full quick start. Both routes
were tested in environments created without `--system-site-packages`, using a
clean source copy and public package downloads rather than the research installation.
The installer versions are in [requirements-toolchain.txt](requirements-toolchain.txt).
Run the [portable installation checks](../docs/VALIDATION.md) after installation.
The compact [tested setup record](TESTED_SETUP.json) states the executed checks
and keeps full scientific campaigns separate from installation verification.

For development in an existing compatible environment, install without resolving dependencies:

```bash
python -m pip install --no-build-isolation --no-deps -e .
```

Use that route after checking the pinned requirements. Explicit forward calls
allocate the buffers described in the [model API](../docs/package_foundation.md);
installation and the loading preflight allocate no forward-simulation buffers.
