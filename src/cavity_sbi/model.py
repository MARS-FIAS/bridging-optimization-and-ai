"""Public model API; forward computation occurs only when explicitly called."""
import contextlib
from dataclasses import dataclass
import numpy as np
from . import _model
from .configuration import geometry_layout, model_config, parameter_declaration, validate_model

# The source-level functions remain available with their original signatures.
construct_stem = _model.construct_stem
simulator = _model.simulator


@dataclass
class SimulationPair:
    trajectories: np.ndarray
    geometry: tuple
    event_audit: tuple

    @property
    def flat(self):
        return self.trajectories.reshape(-1)


@contextlib.contextmanager
def independent_streams(seed):
    """Optional reproducible independent streams for sequential process-local calls."""
    if seed is None:
        yield
        return
    original = np.random.default_rng
    sequence = np.random.SeedSequence(seed)

    def factory(seed=None):
        return original(sequence.spawn(1)[0]) if seed is None else original(seed)

    np.random.default_rng = factory
    try:
        yield
    finally:
        np.random.default_rng = original


@contextlib.contextmanager
def _audit_horizon(audits):
    original = _model.BiochemSimulSwift

    class CheckedSimulator(original):
        def meth_direct(self, jump_diffuse_tor=None, epoch_halt_tup=(None, None)):
            result = super().meth_direct(jump_diffuse_tor, epoch_halt_tup)
            last = self.epoch_mat[self.step_cells - 1, np.arange(self.cells)]
            latest = float(np.max(last))
            reached = bool(np.isfinite(latest) and latest >= 48 * 3600)
            audits.append({'horizon_reached': reached, 'last_event_seconds': latest,
                           'event_history_capacity_reached': bool(np.any(self.step_cells >= self.steps))})
            if not reached:
                raise RuntimeError('Event history ended before 48 h; increase its capacity')
            return result

    _model.BiochemSimulSwift = CheckedSimulator
    try:
        yield
    finally:
        _model.BiochemSimulSwift = original


def simulate_pair(theta, layers, config=None, project='.', seed=None, condition_seeds=None):
    """Run one simulation per biological condition with the paper settings.

    This allocates the configured event buffers and performs forward computation.
    condition_seeds optionally supplies one independent stream specification per
    condition, as in the completed predictive/calibration diagnostics.
    Use one process per worker; the original compiled kernels and optional RNG
    adapter use process-local state. Seed omission preserves stochastic streams.
    """
    import torch
    cfg = model_config(project) if config is None else validate_model(config)
    geometry = geometry_layout(layers)
    values = np.asarray(theta, dtype=float)
    lower = np.asarray([p['prior_low'] for p in cfg['parameters']])
    upper = np.asarray([p['prior_high'] for p in cfg['parameters']])
    if values.shape != (26,) or not np.isfinite(values).all() or np.any(values < lower) or np.any(values > upper):
        raise ValueError('Expected 26 finite parameter coordinates within the prior')
    # Match the executed training/diagnostic driver's native float32 input.
    parameters = torch.as_tensor(theta, dtype=torch.float32, device='cpu')
    if seed is None:
        seed = cfg.get('random_seed')
    audits = []
    if condition_seeds is not None and len(condition_seeds) != 2:
        raise ValueError('Expected one stream specification per biological condition')
    groups = [([0, 1], seed)] if condition_seeds is None else [([c], condition_seeds[c]) for c in [0, 1]]
    parts = []
    for rules, streams in groups:
        with independent_streams(streams), _audit_horizon(audits):
            parts.append(_model.simulator(parameters, parameter_declaration(cfg), 0,
                parameter_set_rules=rules, species=cfg['species'],
                steps=cfg['event_history_capacity'], cells=geometry['total_storage_positions'],
                cellulate=tuple(layers), time_mini=0, time_maxi=48,
                time_unit='Hours', time_delta=0.25, cell_layers=len(layers),
                layer_cells=max(layers), faces=cfg['faces'], seed=None,
                sup_comp=False, verbose=False))
    flat = np.concatenate(parts)
    raw = flat.reshape(2, 3, len(np.arange(0, 49, 0.25)), geometry['total_storage_positions'])
    return SimulationPair(raw, tuple(layers), tuple(audits))
