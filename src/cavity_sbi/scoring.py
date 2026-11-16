"""Joint biological pattern scores and ensemble meta-score summaries."""
from dataclasses import dataclass
import numpy as np
from .configuration import geometry_layout, model_config, validate_model


@dataclass(frozen=True)
class EnsembleScores:
    meta_score: float
    stored_meta_score: float
    r25: float


def scores_from_quantiles(quantiles):
    """Summarize ordered time-by-(q25,q50,q75) values using the original arithmetic."""
    q = np.asarray(quantiles, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3 or q.shape[0] == 0 or not np.isfinite(q).all():
        raise ValueError('Expected finite time-by-three ensemble quantiles')
    if np.any(q < 0) or np.any(q > 1) or np.any(np.diff(q, axis=1) < 0):
        raise ValueError('Quantiles must be ordered and contained in [0,1]')
    value = float(np.mean(q[:, 0] + q[:, 2] - q[:, 1]))
    return EnsembleScores(value, float(np.round(value, 4)), float(np.mean(q[:, 0])))


def ensemble_scores(trajectories):
    """Compute nearest-order quartiles, meta-score S and companion R25."""
    values = np.asarray(trajectories, dtype=np.float64)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError('Expected finite simulation-by-time pattern scores')
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError('Pattern scores must be in [0,1]')
    q = np.percentile(values, [25, 50, 75], axis=0, method='nearest').T
    return scores_from_quantiles(q)


def joint_pattern_scores(flat_trajectories, layers, config=None, project='.'):
    """Score flattened knockout/wild-type molecular simulations; return (N,49).

    Each row uses condition, species, recorded time, storage-position order.
    N + NP is the NANOG fate marker. The biological targets are specified in
    config/model.json. Padded positions and the reservoir retain their original
    storage representation; only the model's physical cell counts normalize fate.
    """
    from ._numerics.objective import ObjectiveFunPanelRule
    cfg = model_config(project) if config is None else validate_model(config)
    g = geometry_layout(layers)
    raw = np.asarray(flat_trajectories)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    frames = len(np.arange(0, 49, 0.25))
    expected = 2 * len(cfg['species']) * frames * g['total_storage_positions']
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] != expected or not np.isfinite(raw).all():
        raise ValueError('Invalid paired molecular-trajectory shape or values')
    score_cfg = cfg['score']
    keywords = {k: score_cfg[k] for k in ['beta', 'L', 'aim_NT', 'aim_G',
                'positive_NT', 'negative_NT', 'positive_G', 'negative_G', 'cusp_NT', 'cusp_G']}
    objective = ObjectiveFunPanelRule(raw, cfg['species'], 0, 48, 'Hours', 0.25,
                                     simulations_maxi=1, verbose=False, show=False,
                                     rules=[0, 1], cellulate=tuple(layers), **keywords)
    objective.apply(tau_mini=36, tau_maxi=48)
    result = objective.data_objective.reshape(raw.shape[0], 49)
    if not np.array_equal(objective.tau, score_cfg['time_hours']) or not np.isfinite(result).all():
        raise ValueError('Invalid joint-score output')
    return result
