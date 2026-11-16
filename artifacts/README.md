# Final neural estimators

The eight files in `final/` are the selected last-round estimators for the eight
ICM geometries. Their original bytes and weights are preserved. The
[manifest](manifest.json) records geometry, round/task, size, SHA-256, parameter
order, prior/configuration source, architecture and ideal conditioning target.
Together the files occupy **3,702,176 bytes**.

Use `cavity_sbi.artifacts.load_final` in the pinned environment. It checks the
artifact bytes and loads on CPU without parameter sampling, MAP optimization,
forward simulation or training. The selected MAP is already cached in each
file. The legacy sbi DirectPosterior pickle loader is intended for these
released, checksum-verified files.

The 400 intermediate estimators and bulk simulation/posterior-sample banks are
not bundled. [Compact candidate summaries](../evidence/README.md) support the
selection comparison independently of those intermediate binaries.
