# Package foundation and public interfaces

The portable package is `cavity_sbi`, installed from this checkout. Its initial
interfaces provide the mechanistic model, fixed spatial geometry, joint pattern
score, ensemble meta-score, project validation and final-estimator access.
The command-line interface provides `doctor`, `geometries`, `generate`, `train`
and `select`. The [workflow guide](workflow.md) covers generation, fitting,
pooling and selection. The [diagnostic guide](diagnostics.md) adds `evaluate-map`,
`predictive`, `sbc` and saved-result reports. The [reader guide](START_HERE.md)
starts with compact-evidence `reproduce`, then explains full execution and
`plot-diagnostic`. The [panel map](REPRODUCTION_MAP.md) locates figure inputs.

## Configuration and model

[`config/model.json`](../config/model.json) defines the ordered 26 parameters,
uniform prior bounds, initial molecular randomization, two biological conditions,
48 h simulation horizon and joint score. [`config/geometries.json`](../config/geometries.json)
distinguishes physical ICM cells from inactive padding and the shared reservoir.
FGF4 is the secreted signaling protein (ligand). In the knockout control, the
model's ligand-export switch reproduces the loss of downstream signaling
feedback; the wild-type condition retains that feedback.

The original observation format contains 196 regular time points, 0–48.75 h,
at 0.25 h spacing. The event simulation reaches 48 h; only the 49 values at
36–48 h enter the objective. The final three format positions do not extend the
biological evaluation window or the event horizon. This storage convention is
preserved for compatibility with the retained data and estimators.

Importing the package does not start work. Explicit forward simulation is
available through the Python API:

```python
from cavity_sbi.configuration import model_config
from cavity_sbi.model import simulate_pair
from cavity_sbi.scoring import joint_pattern_scores, ensemble_scores

config = model_config(".")
theta = [p["reference_value"] for p in config["parameters"]]
# This explicit call performs forward computation and allocates event buffers.
pair = simulate_pair(theta, (5, 3, 1), config=config)
score = joint_pattern_scores(pair.flat, pair.geometry, config=config)
```

The reference parameter values are source defaults, not a fitted solution.
Use the saved selected MAP from a final estimator when evaluating that solution.
The original model-construction and simulation function bodies are preserved;
the public wrapper validates inputs and rejects an event history that ends
before 48 h. The event-history capacity prevents buffer overflow. At the paper
capacity, the three main history buffers alone use about 6.08 GB for (5,3,1)
and 17.48 GB for the largest geometry, per simulator instance. `doctor` reports
these lower bounds for all geometries. Use separate processes for parallel
simulations; the retained numerical kernels use process-local state.

## Scores and estimator access

`joint_pattern_scores` returns one 49-value joint trajectory per paired input.
It retains the minimum aggregation across lower/upper layers and both conditions.
The knockout target is EPI throughout; the wild-type target is cavity-adjacent
PRE with upper EPI. The NANOG fate marker is total NANOG, N + NP, where NP
denotes phosphorylated NANOG.

`ensemble_scores` computes nearest-order quartile trajectories and returns
`meta_score` S, its four-place stored value, and the companion `r25`. Here
S is the time-average of q75 + q25 − q50; R25 is the time-average of q25.
`scores_from_quantiles` accepts existing ordered quartile trajectories without
requiring raw simulations. S is the candidate-selection rule. Its maximum does
not certify that every individual trajectory attains the target.

```python
from cavity_sbi.artifacts import load_final

posterior = load_final("5_3_1", project=".")
selected_map = posterior._map.detach().cpu().numpy().copy()
```

`load_final` verifies the manifest and parameter support, loads on CPU and
freezes gradient updates. It does not draw parameters, perform MAP optimization,
simulate or train. The explicit `_map` access above reads the saved selected
coordinates, preserving the original estimator's cache.

## Source preservation and validation

[`provenance/CORE_EXTRACTION.json`](../provenance/CORE_EXTRACTION.json) records
definition-level hashes and source identities for the extracted implementation.
No numerical function body was edited. Imports were made relative, optional
plotting was made lazy and an unused Blender helper was excluded. Scientific
wrappers and package configuration are separate from those retained bodies.

The `Script_Bank/` reference listings remain traceability inputs;
use the portable package for execution. Historical script paths and scheduler
settings are not the public interface. Final artwork and release checks remain
separate from the numerical reproduction commands.
