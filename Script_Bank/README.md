# Research reference listings

These files document the research implementation from which the portable
package was extracted. **Run the supported commands in
[`docs/START_HERE.md`](../docs/START_HERE.md), rather than these listings.**

| Directory | Purpose |
| --- | --- |
| `Prime/` | Model construction and the original generation/inference procedure |
| `Resources/` | Biochemical simulator, geometry, scoring and inference definitions |
| `Analysis/` | Original numerical analysis and figure recipes |
| `HPC/` | Reference Slurm allocations and worker-launch layouts |

The driver/analysis listings stop immediately on import or execution, and the
shell templates exit before launching workers. Their former top-level code can
start substantial work; the guards keep reference browsing separate from the
configured, restartable `cavity_sbi` workflow. Resource definitions are reference
inputs, not an alternative installation of the package.

Personal filesystem prefixes are replaced by descriptive `REFERENCE_*` markers.
These markers are **not runtime settings**. Generation and diagnostic output
paths are configured through the portable commands. The original allocation
limits remain visible for context; they are not measured runtimes or a scheduler
recommendation.

[`SOURCE_MANIFEST.json`](../provenance/SOURCE_MANIFEST.json) distinguishes the
original imported SHA-256 from each published reference-copy SHA-256. The
[`extraction record`](../provenance/CORE_EXTRACTION.json) identifies the original
source snapshots and unchanged numerical definitions in `src/cavity_sbi/`.
No biological or numerical definition in the installed package was changed in
preparing these reference listings.
