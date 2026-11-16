# Paper configuration

- [model.json](model.json): ordered native-coordinate parameters and box priors,
  initialization, biological conditions, 48 h horizon, observation storage,
  fate thresholds, 36–48 h scoring window and the ideal conditioning target.
- [geometries.json](geometries.json): eight geometries, physical ICM counts,
  centered active indices, inactive padding and shared-reservoir index.
- [workflow.json](workflow.json): paper-scale generation, fitting, MAP and
  evaluation settings; [workflow-small.json](workflow-small.json) is a reduced
  interface example. Use the same profile throughout a run.
- [diagnostics.json](diagnostics.json): fixed-MAP, density-wide predictive and
  original-prior marginal SBC settings, including the reported 256-pair and
  1,000-case × 10,000-draw configurations. [diagnostics-small.json](diagnostics-small.json)
  is a reduced interface example. See the [diagnostic guide](../docs/diagnostics.md).
- [figures.json](figures.json): original parameter-panel membership/display order,
  rank-bin references, coverage color scale and conditional-grid settings.
  See the [reproduction map](../docs/REPRODUCTION_MAP.md).

Data and output directories are explicit command arguments; historical
machine paths are not defaults. `random_seed` is optional. The event-history
capacity is a memory setting, not a measured-runtime estimate. The public model
wrapper rejects a trajectory that ends before the 48 h horizon.
