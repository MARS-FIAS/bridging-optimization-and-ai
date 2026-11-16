# From paper content to code and evidence

Stable LaTeX labels identify content across manuscript versions; printed
supplementary numbers depend on the final article. Paths below start at the
repository root. Reference plots are numerical renderings, separate from the
paper's designed layout. Begin with the [reader guide](START_HERE.md).

`python -m cavity_sbi reproduce --execute` rebuilds the supplied computed plots
and tables in `outputs/reproduction/`. Restrict work with `--component selection`,
`predictive`, `calibration` or `parameters`. It verifies input hashes, recomputes
statistics and writes PDF/SVG/PNG plots, CSV tables, `results.json` and a manifest.
It loads no neural model and generates no simulations or posterior draws.

## Computed panels and numerical tables

| Paper content | Supplied input | Component | Generated numerical layer |
| --- | --- | --- | --- |
| Main Figure 5A, selected-MAP progression (`fig:cavity_results`) | `evidence/selection_candidates.npz` | `selection` | `score_progression.csv`, `score_progression.pdf` |
| Score-policy comparison (`fig:score_policy_agreement`) | All 400 candidates' 49×3 quartile trajectories and stored-S/task identities | `selection` | `candidate_scores.csv`, `selection_groups.csv`, `score_policy_agreement.pdf` |
| Predictive scores, histogram and target attainment (`fig:supp_goal_predictive`, A/B/D) | `evidence/goal_predictive.npz`: all 256 joint-score trajectories | `predictive` | `predictive_score_timeseries.csv`, `predictive_score_histogram.csv`, `goal_predictive.pdf` |
| Intended fates (same figure, C) | Same NPZ: per-case classified cell counts for both conditions and groups | `predictive` | `predictive_intended_fates.csv`; independent score reconstruction from counts |
| Rank histograms (`fig:supp_sbc_ranks`) | `evidence/marginal_sbc.npz`: 1,000 per-case randomized ranks and rank fractions | `calibration` | `rank_histograms.csv`, `sbc_ranks.pdf`; also `sbc_rank_cdf.pdf` for every marginal CDF |
| Interval coverage (`fig:supp_sbc_coverage`) | Same NPZ: 1,000×4×26 coverage indicators | `calibration` | `calibration_parameters.csv`, `calibration_coverage.csv`, `sbc_coverage.pdf` |
| MAPs and fitted reference ranges (`fig:supp_parameter_reproducibility_fgf4`) | `evidence/selected_MAP_parameters.csv`, `evidence/reference_density_quantiles_256.csv` | `parameters` | `parameter_reference.pdf`; all 208 MAP coordinates and 26 reference boxes |
| Prior predictive comparison in diagnostic text | `evidence/prior_predictive.npz`: all 1,000 prior-case score trajectories | `predictive` | `results.json` → `prior_predictive`; `prior_predictive_score_timeseries.csv` |
| Parameter/prior table (`tab:supp_parameters`) | `config/model.json` | `parameters` | `parameter_priors.csv`: 26 ordered keys, meanings, units and bounds |
| Geometry dimensions in the technical methods | `config/geometries.json` | `parameters` | `geometry_dimensions.csv`: eight physical cell counts and storage layouts |

Selection compares retained task winners along the original S-selected path.
It recomputes four-place S, unrounded R25, first-maximum ties and all 40 choices.
It does not simulate an alternative sequential policy. Seven final S values are
1; the largest geometry has 0.9771.

The predictive check concerns one final `(5,3,1)` density at the ideal target.
One pair per parameter draw evaluates the joint parameter-and-simulation law.
The original-prior baseline reuses completed SBC forward simulations. SBC
retains all 26 marginals, Holm correction and its original-prior reference.
Coverage and target attainment remain separate; see the [diagnostic guide](diagnostics.md).

Parameter boxes use the declared 256 ideal-target draws from the final `(5,3,1)`
density. Only marginal summaries are distributed. Dots are saved geometry-specific
MAP coordinates. Native linear coordinates are normalized as
`100 × (value − prior lower) / (prior upper − prior lower)`.
The four groups are in `config/figures.json`. Fitted quantiles are not calibrated
coverage or a minimum-required-dose claim.

## Conditional-correlation recipe

The panel `fig:supp_posterior_correlations` uses pairwise Pearson coefficients
while other coordinates stay fixed at the MAP. The retained
[analysis service](../Script_Bank/Analysis/Fig0_Services.py) calls sbi
0.19.2 `analysis.conditional_corrcoeff` on the prior bounds at resolution 250.
It rounds MAP coordinates above 1 to integers, and others to two decimals.
That is the recipe default; `--condition saved-map` selects unrounded coordinates.

Preview a two-parameter calculation:

```bash
python -m cavity_sbi conditional-correlations --geometry 5_3_1 --parameters FL,tau_L --output-dir outputs/correlation-pair
```

Add `--execute` in the [pinned neural environment](../environments/README.md)
to evaluate the frozen density. Omit `--parameters` for all 26 in source display
order. A full grid evaluates 325 pairs × 62,500 points: **20,312,500 density
evaluations**. This is explicit CPU work, separate from `reproduce`, and generates
no forward simulations or posterior draws.

The recipe uses raw conditional log density: its normalization constant cancels
in the correlation, avoiding incidental sampling for normalization. CSV and
manifest record coordinates, resolution, parameter order, artifact/weight hashes
and software versions.

**Numerical source boundary:** the original matrix behind the paper's heatmap
is a separate artwork input and is not distributed here. This recipe supports
a new reproducible evaluation; identity with the original coefficients requires
that matrix for comparison. No matrix was invented or recovered from an image.
Conditional coefficients do not establish global identifiability or exclude
compensatory parameter changes.

## Illustrative and conceptual material

| Paper content | Scientific source | Reproduction scope |
| --- | --- | --- |
| Strategies (`fig:calibration_strategies`) | Review definitions and cited literature | Conceptual diagram; no numerical dataset |
| Cavity/regulatory model (`fig:cavity_model`) | Mechanistic model and `config/model.json` | Schematic biology/transport, not measured trajectories or concentrations |
| Score levels (`fig:pattern_score`) | Score definitions and targets | Illustrative trajectories and symbolic stages, without physical times |
| Workflow (`fig:sbi_workflow`) | `config/workflow.json`, `generate` → `train` → `select` | Implemented procedure; example curves are illustrative |
| Main Figure 5B–D (`fig:cavity_results`) | Biological target and model hypothesis | Illustrative interpretation, without an empirical dose–response threshold |
| Geometries (`fig:supp_geometry`) | `config/geometries.json` | Schematic model inputs; vector illustration at `figures/model_geometries.pdf` |
| Strategy table (`tab:optimization_ai_sbi`) | Cited methods and review discussion | Qualitative comparison, not a computed benchmark dataset |

For new numerical results, follow the [training workflow](workflow.md) and
[diagnostic commands](diagnostics.md), then `plot-diagnostic` for a completed
predictive/SBC run. Full execution does not require the private data bank.
Generated banks stay in ignored output folders under the [data policy](data_access.md).
