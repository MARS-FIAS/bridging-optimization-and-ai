# Enhanced manuscript figures

These are the approved figures from the revised manuscript of 18 November 2026.
PDFs and PNG previews show the intended typography; SVGs are editable sources.
All files are copied unchanged from the manuscript artwork. The font files
and their licences are supplied [alongside them](../fonts/README.md).
[Checksums and provenance](manifest.json) identify every file.

| Figure | Title | Content | Files |
| --- | --- | --- | --- |
| 1 | Calibration strategies for spatial-stochastic biochemical models | Illustrative method comparison | [PDF](fig_calibration_strategies.pdf) / [SVG](fig_calibration_strategies.svg) / [PNG](fig_calibration_strategies.png) |
| 2 | Spatial-stochastic blastocyst cavity model | Schematic model and regulatory network | [PDF](fig_cavity_model.pdf) / [SVG](fig_cavity_model.svg) / [PNG](fig_cavity_model.png) |
| 3 | Pattern score and ensemble meta-score | Illustrative pattern scores and ensemble summary | [PDF](fig_pattern_score.pdf) / [SVG](fig_pattern_score.svg) / [PNG](fig_pattern_score.png) |
| 4 | Two-condition training and ensemble-based candidate selection | Illustrative inference workflow | [PDF](fig_sbi_workflow.pdf) / [SVG](fig_sbi_workflow.svg) / [PNG](fig_sbi_workflow.png) |
| 5 | Selected MAP performance and conceptual interpretation | A computed; B–D illustrative | [PDF](fig_cavity_results.pdf) / [SVG](fig_cavity_results.svg) / [PNG](fig_cavity_results.png) |
| S1 | Eight fixed ICM geometries (schematic model inputs) | Schematic model inputs | [PDF](fig_supp_geometry.pdf) / [SVG](fig_supp_geometry.svg) / [PNG](fig_supp_geometry.png) |
| S2 | Agreement of meta-score selection policies | Computed score-policy comparison | [PDF](fig_score_policy_agreement.pdf) / [SVG](fig_score_policy_agreement.svg) / [PNG](fig_score_policy_agreement.png) |
| S3 | Biological target attainment across the fitted density (Figure S3) | Computed predictive assessment | [PDF](fig_supp_goal_predictive.pdf) / [SVG](fig_supp_goal_predictive.svg) / [PNG](fig_supp_goal_predictive.png) |
| S4 | Frozen-estimator marginal SBC under the original prior | Computed marginal rank assessment | [PDF](fig_supp_sbc_ranks.pdf) / [SVG](fig_supp_sbc_ranks.svg) / [PNG](fig_supp_sbc_ranks.png) |
| S5 | Empirical marginal interval coverage | Computed marginal coverage | [PDF](fig_supp_sbc_coverage.pdf) / [SVG](fig_supp_sbc_coverage.svg) / [PNG](fig_supp_sbc_coverage.png) |
| S6 | Conditional parameter correlations at the MAP | Preserved computed conditional-correlation artwork | [PDF](fig_supp_posterior_correlations.pdf) / [SVG](fig_supp_posterior_correlations.svg) / [PNG](fig_supp_posterior_correlations.png) |
| S7 | Selected MAP parameterizations and fitted reference ranges | Computed selected parameters and fitted reference ranges | [PDF](fig_supp_parameter_reproducibility_fgf4.pdf) / [SVG](fig_supp_parameter_reproducibility_fgf4.svg) / [PNG](fig_supp_parameter_reproducibility_fgf4.png) |

## Recreate the numerical content

The [paper reproduction map](../../docs/REPRODUCTION_MAP.md) connects computed
panels to the supplied evidence and commands. `python -m cavity_sbi reproduce --execute` regenerates numerical plots and summaries without simulations or
neural sampling. These reference renderings use a different layout from the
supplied designed artwork; the editable SVGs preserve the publication design.

Main Figures 1–4 and 5B–D are illustrative, and S1 depicts fixed model inputs.
S6 preserves the conditional-correlation artwork; the full-precision original
matrix is not supplied. Its [recomputation recipe](../../docs/REPRODUCTION_MAP.md#conditional-correlation-recipe)
remains available. Conditional correlations do not establish global identifiability.
No full simulation datasets, posterior-draw banks or manuscript/reviewer files
are included in this figure collection.
