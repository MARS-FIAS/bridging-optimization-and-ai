# Configuration and biological glossary

The canonical numerical settings are in [model.json](../config/model.json),
[geometries.json](../config/geometries.json), [workflow.json](../config/workflow.json)
and [diagnostics.json](../config/diagnostics.json). The tables below describe
those settings; they do not change the model or supply a new fitted solution.

## Biological symbols

FGF4 is the secreted signaling protein, also called the ligand. NANOG and GATA6
regulate the EPI/PRE fate choice. Ligand–receptor signaling activates ERK, which
feeds back to transcription and NANOG phosphorylation.

| Code symbol | Meaning and role |
| --- | --- |
| `N` | Unphosphorylated NANOG; promotes the EPI program, activates Nanog/Fgf4 and represses Gata6 |
| `G` | GATA6; promotes the PRE program, activates Gata6 and represses Nanog/Fgf4 |
| `NP` | Phosphorylated NANOG; can occupy NANOG-binding sites and has faster degradation |
| `N_MRNA`, `G_MRNA`, `FC_MRNA` | Nanog, Gata6 and Fgf4 mRNAs; templates for their proteins |
| `FC` | Cytoplasmic FGF4, the intracellular source of exported ligand |
| `FM`, `FD` | Monomeric and dimeric FGFR–FGF4 complexes; the active dimer promotes ERK activation |
| `EI`, `EA` | Inactive and activated ERK; activated ERK regulates transcription and NANOG phosphorylation |
| `FL` | Cavity-associated/luminal FGF4 pool; also the name of its initial loading parameter |
| `FS_UP`, `FS_DO` | FGF4 in effective TE-associated reservoirs, with exchange and degradation |

These symbols distinguish biochemical species from promoter-occupancy indicators
and activation/inhibition counters. The latter encode regulation rather than
additional independent biochemical species. The fate classifier uses total
NANOG, `N + NP`, together with GATA6. Cells are fixed; the TE is represented by
reservoirs rather than individually simulated TE cells.

## Parameter coordinates, units and priors

Keep the following **26-key order** when passing parameter arrays to the model
or estimator. Coordinates are native and linear; there is no implicit log10
mapping. The first eight keys name the regulated promoter first and regulator
second, and tune the corresponding binding-threshold scales. Lifetimes and
transport times use **seconds**, while recorded/evaluation times use **hours**.
The three inverse effective bimolecular coefficients follow the model's
dimensionless-count convention; they are not individual event waiting times.
`MRNA` and `PRO` are initialization randomization scales, not hard count caps.
`reference_value` in the JSON is a source default, not a fitted MAP.

| Key | Biological role | Native unit | Uniform prior bounds |
| --- | --- | --- | --- |
| `N_N` | Activation of N transcription by N | molecule count | [0, 1000] |
| `G_G` | Activation of G transcription by G | molecule count | [0, 1000] |
| `FC_N` | Activation of FC transcription by N | molecule count | [0, 1000] |
| `G_EA` | Activation of G transcription by EA | molecule count | [0, 1000] |
| `G_N` | Repression of G transcription by N | molecule count | [0, 1000] |
| `N_G` | Repression of N transcription by G | molecule count | [0, 1000] |
| `FC_G` | Repression of FC transcription by G | molecule count | [0, 1000] |
| `N_EA` | Repression of N transcription by EA | molecule count | [0, 1000] |
| `MRNA` | Initial N_MRNA/G_MRNA randomization scale | molecule count | [0, 250] |
| `PRO` | Initial N/G protein randomization scale | molecule count | [0, 1000] |
| `FL` | Initial luminal FGF4 loading coordinate | molecule count | [0, 25000] |
| `td_FC` | Cytoplasmic FGF4 degradation lifetime | s | [300, 28800] |
| `td_FM` | Membrane-associated FGF4 degradation lifetime | s | [300, 28800] |
| `td_FL` | Luminal FGF4 degradation lifetime | s | [300, 28800] |
| `td_FS` | Shared degradation lifetime for both external sink pools | s | [300, 28800] |
| `tau_C` | Total cytoplasmic export time scale | s | [300, 45000] |
| `tau_M` | Membrane export time scale | s | [300, 42000] |
| `tau_L` | Luminal export time scale | s | [30, 450000] |
| `tau_S` | Shared external-pool export time scale | s | [30, 420000] |
| `tau_df_FD` | Inverse effective dimerization coefficient | s in dimensionless-count convention | [300, 345600] |
| `tau_db_FD` | Dimer dissociation time scale | s | [30, 345600] |
| `tau_ef_EA` | Inverse effective ERK-activation coefficient | s in dimensionless-count convention | [300, 43200] |
| `tau_eb_EA` | ERK inactivation time scale | s | [30, 43200] |
| `tau_pf_NP` | Inverse effective NANOG-phosphorylation coefficient | s in dimensionless-count convention | [300, 43200] |
| `tau_pb_NP` | NANOG dephosphorylation time scale | s | [30, 43200] |
| `chi_auto` | Autocrine fraction of cytoplasmic export | dimensionless | [0, 1] |

## Geometry and state storage

The first layer is cavity-adjacent. Wild type targets PRE there and EPI in all
upper layers; the Fgf4-knockout condition targets EPI throughout. The export
switch reproduces loss of downstream FGF4 feedback in that control. Inactive
padding and the shared reservoir are storage positions, not extra physical ICM
cells. The geometry file records the centered active indices explicitly.

| Geometry ID | Physical ICM cells | Storage positions | Wild-type EPI:PRE target counts |
| --- | --- | --- | --- |
| `1_1` | 2 | 3 | 1:1 |
| `2_2` | 4 | 5 | 2:2 |
| `3_3` | 6 | 7 | 3:3 |
| `4_4` | 8 | 9 | 4:4 |
| `5_5` | 10 | 11 | 5:5 |
| `5_3_1` | 9 | 16 | 4:5 |
| `7_5_3_1` | 16 | 29 | 9:7 |
| `9_7_5_3_1` | 25 | 46 | 16:9 |

The 67 state components per simulator position include constrained promoter
states and auxiliary variables; they are not 67 independent inferred parameters.
The inference vector has the 26 coordinates listed above. `doctor` reports
geometry-specific storage and event-buffer requirements.

## Observations and the two score levels

- Each parameter vector produces knockout and wild-type simulations to 48 h.
  The retained molecular format has 196 samples at 0–48.75 h; its last three
  format positions do not extend the event horizon or biological evaluation.
- The joint pattern-score observation has 49 values at 36–48 h, spaced by
  0.25 h. Layer/condition scores are combined by their minimum. This time series
  is the neural estimator's conditioning variable.
- The ensemble meta-score `S` averages `q75 + q25 − q50` across those times,
  using nearest-order ensemble quartiles. Stored S is rounded to four decimal
  places and selects task/round winners, keeping the first maximum under ties.
- `R25` averages `q25` and is the monotone reporting/sensitivity companion.
  It does not replace S in the preserved training/selection workflow. S = 1
  does not certify that every individual trajectory is perfect.

The ideal neural conditioning target is 49 ones. This is a goal-conditioned
parameter distribution, not inference conditioned on an experimental trajectory.
The [diagnostic guide](diagnostics.md) distinguishes predictive target attainment
from marginal uncertainty calibration under the original prior.

## Workflow profiles and generated files

`workflow.json` specifies five rounds, 100,000 parameter–score pairs per round,
ten tasks and two fits per task. Round 0 uses its own data; later fits pool the
current and preceding rounds. Candidates use their cached MAP and 50 paired
evaluations. `workflow-small.json` reduces counts for an interface exercise,
while retaining the biological model, horizon and memory needs.

`diagnostics.json` specifies 50 selected-MAP pairs, 256 density-wide pairs and
1,000 SBC cases with 10,000 conditional draws each. `diagnostics-small.json`
is an interface profile. Available seeds are recorded; seeds are optional for
new stochastic work. `figures.json` fixes parameter-panel membership, rank-bin
references and plot scales for the supplied numerical reproduction.

Use one unchanged profile/code version per run. Run manifests bind generated
files to settings, artifact and code hashes. The [workflow guide](workflow.md)
explains restartable case/fit records; the [evidence guide](../evidence/README.md)
describes supplied compact array schemas. Outputs belong in a new `outputs/`
folder or another explicit data directory, outside source/config/artifact folders.
