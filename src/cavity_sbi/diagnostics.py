"""Restartable CPU checks of one frozen estimator; no training or MAP optimization.

Preview/configuration uses the standard library only. Scientific imports occur
only in execution functions. Generated observations/draws stay in the run folder.
"""
import contextlib
from datetime import datetime, timezone
import os
from pathlib import Path
import secrets
import signal
import time

from . import __version__
from .artifacts import checked_inventory, file_sha256
from .configuration import get_geometry, model_config
from .workflow import (atomic_json, atomic_npz, fingerprint, load_run_posterior,
                       lock, read_json, validate_run_location)

OPERATIONS = ('evaluate-map', 'predictive', 'sbc')


def diagnostic_config(project='.', profile='config/diagnostics.json'):
    path = Path(profile)
    cfg = read_json(path if path.is_absolute() else Path(project) / path)
    for key in ['map_pairs', 'predictive_pairs', 'sbc_cases', 'posterior_draws_per_case',
                'torch_threads', 'max_sampling_batch_size', 'posterior_sampling_timeout_seconds']:
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError('Expected a positive integer for ' + key)
    if cfg['posterior_draws_per_case'] < 2:
        raise ValueError('At least two posterior draws are required for intervals')
    levels = cfg['coverage_levels']
    if (not isinstance(levels, list) or not levels or any(type(c) not in [float, int] or not 0 < c < 1 for c in levels)
            or any(a >= b for a, b in zip(levels, levels[1:])) or not 0 < cfg['alpha'] < 1):
        raise ValueError('Invalid coverage levels or familywise significance level')
    for operation in OPERATIONS:
        seed = cfg['seeds'][operation]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2 ** 63):
            raise ValueError('Optional seeds must be nonnegative integers below 2**63')
        if type(cfg['forward_seed_roles'][operation]) is not int or cfg['forward_seed_roles'][operation] < 0:
            raise ValueError('Invalid forward seed role')
    for key in ['posterior_seed_offset', 'rank_seed_role']:
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError('Invalid seed role or offset')
    for key in ['keep_raw_trajectories', 'keep_posterior_samples']:
        if type(cfg[key]) is not bool:
            raise ValueError(key + ' must be boolean')
    return cfg


def diagnostic_specification(operation, project, geometry, profile='config/diagnostics.json',
                             estimator=None, estimator_sha256=None):
    if operation not in OPERATIONS:
        raise ValueError('Unknown diagnostic operation')
    if (estimator is None) != (estimator_sha256 is None):
        raise ValueError('A custom estimator requires its path and SHA-256 together')
    if estimator is None:
        inventory = checked_inventory(project)
        row = next((r for r in inventory['artifacts'] if r['geometry'] == geometry), None)
        if row is None:
            raise ValueError('No released estimator for this geometry')
        path, expected = Path(project) / row['file'], row['sha256']
    else:
        path, expected = Path(estimator), estimator_sha256
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('Expected a lowercase SHA-256 digest')
    if file_sha256(path) != expected:
        raise ValueError('Diagnostic estimator checksum differs')
    code = {str(p.relative_to(Path(__file__).parent)): file_sha256(p)
            for p in sorted(Path(__file__).parent.rglob('*.py'))}
    return {'schema_version': 1, 'package_version': __version__, 'operation': operation,
            'model': model_config(project), 'geometry': get_geometry(geometry, project),
            'settings': diagnostic_config(project, profile),
            'estimator': {'source_file': str(path.resolve()), 'sha256': expected,
                          'origin': 'released final' if estimator is None else 'explicit user-supplied frozen estimator'},
            'code_sha256': code}


def case_count(spec):
    key = {'evaluate-map': 'map_pairs', 'predictive': 'predictive_pairs', 'sbc': 'sbc_cases'}[spec['operation']]
    return spec['settings'][key]


def diagnostic_plan(spec, project, run, start=0, stop=None, prepare_only=False):
    count = case_count(spec)
    stop = count if stop is None else stop
    if not 0 <= start < stop <= count:
        raise ValueError('Invalid diagnostic case range')
    operation = spec['operation']
    return {'operation': operation, 'mode': 'preview', 'geometry': spec['geometry']['id'],
            'profile': spec['settings']['label'], 'planned_cases': count, 'case_range': [start, stop],
            'condition_specific_simulations': 0 if prepare_only else 2 * (stop - start),
            'parameter_preparation': {'evaluate-map': 'repeat the saved selected MAP; no optimization',
                                      'predictive': 'sample the frozen estimator at the ideal target',
                                      'sbc': 'independent draws from the original uniform box prior'}[operation],
            'conditioning': 'each case\'s simulated 49-component score' if operation == 'sbc' else 'ideal 49-component target',
            'posterior_draws_per_sbc_case': spec['settings']['posterior_draws_per_case'] if operation == 'sbc' else 0,
            'planned_conditional_posterior_draws': ((0 if prepare_only else (stop-start) * spec['settings']['posterior_draws_per_case'])
                                                  if operation == 'sbc' else count if operation == 'predictive' else 0),
            'coverage_levels': spec['settings']['coverage_levels'] if operation == 'sbc' else [],
            'prepare_only': prepare_only, 'fingerprint': fingerprint(spec),
            'run_directory': str(validate_run_location(project, run)),
            'execute_required': True, 'files_written': False, 'training_runs': 0}


def ensure_diagnostic(project, run, spec):
    run = validate_run_location(project, run)
    path = run / 'diagnostic.json'
    if path.exists():
        return checked_diagnostic(run, spec)
    run.mkdir(parents=True, exist_ok=True)
    with lock(run / '.diagnostic.lock'):
        if path.exists():
            record = checked_diagnostic(run, spec)
        else:
            if (run / 'run.json').exists():
                raise ValueError('Use a separate directory from generation/training outputs')
            from .cli import environment_report
            environment = environment_report(check_runtime=True)
            if environment['reference_version_differences']:
                raise RuntimeError('Use the pinned execution environment: ' + ', '.join(environment['reference_version_differences']))
            seed = spec['settings']['seeds'][spec['operation']]
            record = {'fingerprint': fingerprint(spec), 'specification': spec,
                      'random_entropy': secrets.randbits(63) if seed is None else seed,
                      'execution_environment': environment,
                      'created_utc': datetime.now(timezone.utc).isoformat()}
            atomic_json(path, record)
    return record


def checked_diagnostic(run, spec=None):
    record = read_json(Path(run) / 'diagnostic.json')
    if record['fingerprint'] != fingerprint(record['specification']):
        raise ValueError('Diagnostic manifest fingerprint differs')
    if spec is not None and record['specification'] != spec:
        raise ValueError('Diagnostic configuration, estimator or code differs')
    return record


def case_identity(record, stage):
    return fingerprint({'run': record['fingerprint'], 'stage': stage})


def write_array_record(path, metadata, **arrays):
    atomic_npz(path, **arrays)
    atomic_json(Path(path).with_suffix('.json'), dict(metadata, sha256=file_sha256(Path(path))))


def read_array_record(path, identity, case_id=None):
    import numpy as np
    path = Path(path)
    record = read_json(path.with_suffix('.json'))
    if record['fingerprint'] != identity or (case_id is not None and record['case_id'] != case_id):
        raise ValueError('Diagnostic array identity differs')
    if record['sha256'] != file_sha256(path):
        raise ValueError('Diagnostic array checksum differs')
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.array(data[key]) for key in data.files}
    return arrays, record


def _partial_output(path):
    path = Path(path)
    if path.exists() != path.with_suffix('.json').exists():
        raise ValueError('Incomplete output retained for inspection: ' + str(path))


def _load_estimator(spec):
    from .artifacts import weights_sha256
    posterior = load_run_posterior(spec['estimator']['source_file'], spec['estimator']['sha256'], spec['model'])
    for parameter in posterior.posterior_estimator.parameters():
        parameter.requires_grad_(False)
    return posterior, weights_sha256(posterior)


def _validate_theta(theta, spec, count):
    import numpy as np
    from .datasets import _validate_theta as validate
    if theta.shape != (count, 26):
        raise ValueError('Unexpected diagnostic parameter-bank dimensions')
    validate(theta, spec['model'])


def _parameter_bank(spec, entropy):
    import numpy as np
    import torch
    from sbi.utils import BoxUniform
    from .artifacts import weights_sha256
    cfg, model, operation = spec['settings'], spec['model'], spec['operation']
    count = case_count(spec)
    torch.set_num_threads(cfg['torch_threads'])
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(entropy)
        if operation == 'sbc':
            # No conditional network is needed to draw independent prior truths.
            prior = BoxUniform(low=torch.tensor([p['prior_low'] for p in model['parameters']], dtype=torch.float32),
                               high=torch.tensor([p['prior_high'] for p in model['parameters']], dtype=torch.float32))
            theta = prior.sample((count,)).cpu().numpy()
        else:
            posterior, before = _load_estimator(spec)
            if operation == 'evaluate-map':
                cached = getattr(posterior, '_map', None)
                if cached is None:
                    raise ValueError('The estimator must contain the saved selected MAP')
                theta = np.repeat(cached.detach().cpu().numpy().reshape(1, -1), count, axis=0)
            else:
                with sampling_deadline(cfg['posterior_sampling_timeout_seconds']):
                    theta = posterior.sample((count,), x=torch.ones(49), show_progress_bars=False,
                                             max_sampling_batch_size=cfg['max_sampling_batch_size']).detach().cpu().numpy()
            if weights_sha256(posterior) != before:
                raise ValueError('Frozen estimator weights changed during parameter preparation')
    return np.asarray(theta, dtype=np.float32)


def prepare_diagnostic(project, run, spec):
    import numpy as np
    record = ensure_diagnostic(project, run, spec)
    path = Path(run) / 'parameters.npz'
    identity = case_identity(record, 'parameters')
    _partial_output(path)
    if not path.exists():
        with lock(path.with_suffix('.lock')):
            _partial_output(path)
            if not path.exists():
                theta = _parameter_bank(spec, record['random_entropy'])
                _validate_theta(theta, spec, case_count(spec))
                metadata = {'fingerprint': identity, 'cases': len(theta), 'operation': spec['operation']}
                write_array_record(path, metadata, theta=theta,
                                   parameter_keys=np.asarray([p['key'] for p in spec['model']['parameters']]))
    arrays, metadata = read_array_record(path, identity)
    _validate_theta(arrays['theta'], spec, case_count(spec))
    if arrays['parameter_keys'].tolist() != [p['key'] for p in spec['model']['parameters']]:
        raise ValueError('Parameter-bank column order differs')
    return record, arrays['theta'], metadata


@contextlib.contextmanager
def sampling_deadline(seconds):
    """Bound a per-observation sampler on Linux; retain failed cases for retry."""
    def timeout_handler(signum, frame):
        raise TimeoutError('Frozen posterior sampling exceeded %s seconds' % seconds)
    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _observation(run, spec, record, case_id, theta, parameter_sha):
    import numpy as np
    from .model import simulate_pair
    from .scoring import joint_pattern_scores
    path = Path(run) / 'observations' / ('case_%06d.npz' % case_id)
    identity = case_identity(record, 'observation')
    _partial_output(path)
    if path.exists():
        arrays, meta = read_array_record(path, identity, case_id)
        if meta['parameter_sha256'] != parameter_sha or not np.array_equal(arrays['theta'], theta):
            raise ValueError('Observation differs from the fixed parameter draw')
    else:
        cfg = spec['settings']
        entropy = [[record['random_entropy'], cfg['forward_seed_roles'][spec['operation']], case_id, c] for c in [0, 1]]
        started = time.monotonic()
        pair = simulate_pair(theta, tuple(spec['geometry']['layers']), config=spec['model'], condition_seeds=entropy)
        if not np.isfinite(pair.flat).all() or np.any(pair.flat < 0):
            raise ValueError('Invalid molecular observations')
        score = joint_pattern_scores(pair.flat, pair.geometry, config=spec['model']).reshape(49)
        audits = list(pair.event_audit)
        meta = {'fingerprint': identity, 'case_id': case_id, 'parameter_sha256': parameter_sha,
                'condition_audits': audits, 'paired_wall_seconds': time.monotonic() - started}
        arrays = {'theta': theta, 'joint_score': score, 'condition_stream_entropy': np.asarray(entropy, dtype=np.uint64)}
        if cfg['keep_raw_trajectories']:
            arrays['raw_trajectories'] = pair.flat
        _validate_observation(arrays, meta)
        write_array_record(path, meta, **arrays)
        arrays, meta = read_array_record(path, identity, case_id)
    _validate_observation(arrays, meta)
    return arrays, meta


def _validate_observation(arrays, meta):
    import numpy as np
    score = arrays['joint_score']
    if score.shape != (49,) or not np.isfinite(score).all() or np.any(score < 0) or np.any(score > 1):
        raise ValueError('Invalid joint-score observation')
    audits = meta['condition_audits']
    if len(audits) != 2 or any(not a['horizon_reached'] or a['last_event_seconds'] < 48 * 3600
                              or a['event_history_capacity_reached'] for a in audits):
        raise ValueError('Every condition must reach 48 h without exhausting event-history capacity')


def _conditional_samples(posterior, spec, score, seed):
    import torch
    cfg = spec['settings']
    with torch.random.fork_rng(devices=[]), torch.no_grad(), sampling_deadline(cfg['posterior_sampling_timeout_seconds']):
        torch.manual_seed(seed % (2 ** 63))
        values = posterior.sample((cfg['posterior_draws_per_case'],), x=torch.as_tensor(score, dtype=torch.float32),
                                  max_sampling_batch_size=cfg['max_sampling_batch_size'], show_progress_bars=False)
    samples = values.detach().cpu().numpy()
    _validate_theta(samples, spec, cfg['posterior_draws_per_case'])
    return samples


def _sbc_case(run, spec, record, case_id, theta, observation, observation_meta, posterior, weight_hash):
    import numpy as np
    from .artifacts import weights_sha256
    from .diagnostic_stats import ranks_and_intervals
    path = Path(run) / 'diagnostics' / ('case_%06d.npz' % case_id)
    identity = case_identity(record, 'sbc')
    _partial_output(path)
    if path.exists():
        arrays, meta = read_array_record(path, identity, case_id)
        if meta['observation_sha256'] != observation_meta['sha256'] or not np.array_equal(arrays['truth'], theta):
            raise ValueError('SBC case differs from the fixed observation/truth')
        if meta['weights_sha256'] != weight_hash:
            raise ValueError('SBC estimator weights differ')
        return False
    cfg = spec['settings']
    samples_path = Path(run) / 'posterior' / ('case_%06d.npz' % case_id)
    _partial_output(samples_path)
    sample_identity = case_identity(record, 'conditional_samples')
    if samples_path.exists():
        saved, meta = read_array_record(samples_path, sample_identity, case_id)
        if meta['observation_sha256'] != observation_meta['sha256'] or meta['weights_sha256'] != weight_hash:
            raise ValueError('Saved conditional samples differ from the observation or frozen weights')
        samples = saved['samples']
        _validate_theta(samples, spec, cfg['posterior_draws_per_case'])
    else:
        samples = _conditional_samples(posterior, spec, observation['joint_score'],
                                       record['random_entropy'] + cfg['posterior_seed_offset'] + case_id)
        if weights_sha256(posterior) != weight_hash:
            raise ValueError('Frozen estimator weights changed during conditional sampling')
        if cfg['keep_posterior_samples']:
            write_array_record(samples_path, {'fingerprint': sample_identity, 'case_id': case_id,
                               'observation_sha256': observation_meta['sha256'], 'weights_sha256': weight_hash}, samples=samples)
    if weights_sha256(posterior) != weight_hash:
        raise ValueError('Frozen estimator weights changed during conditional sampling')
    rng = np.random.default_rng(np.random.SeedSequence([record['random_entropy'], cfg['rank_seed_role'], case_id]))
    statistics = ranks_and_intervals(samples, theta, cfg['coverage_levels'], rng)
    write_array_record(path, {'fingerprint': identity, 'case_id': case_id, 'weights_sha256': weight_hash,
                       'observation_sha256': observation_meta['sha256'], 'posterior_draws': len(samples)},
                       truth=theta, observation=observation['joint_score'], **statistics)
    return True


def run_diagnostic(project, run, spec, start=0, stop=None, prepare_only=False):
    # The same range validation applies to direct API callers and the CLI.
    diagnostic_plan(spec, project, run, start, stop, prepare_only)
    record, theta, parameters_meta = prepare_diagnostic(project, run, spec)
    if prepare_only:
        return {'operation': spec['operation'], 'mode': 'prepared parameters', 'planned_cases': len(theta),
                'condition_specific_simulations': 0, 'training_runs': 0}
    import torch
    torch.set_num_threads(spec['settings']['torch_threads'])
    posterior, weight_hash = _load_estimator(spec) if spec['operation'] == 'sbc' else (None, None)
    stop = len(theta) if stop is None else stop
    new_observations = new_sbc = reused = 0
    for case_id in range(start, stop):
        with lock(Path(run) / 'locks' / ('case_%06d.lock' % case_id)):
            phase = 'forward simulations'
            try:
                existed = (Path(run) / 'observations' / ('case_%06d.npz' % case_id)).exists()
                observation, meta = _observation(run, spec, record, case_id, theta[case_id], parameters_meta['sha256'])
                new_observations += int(not existed)
                reused += int(existed)
                if spec['operation'] == 'sbc':
                    phase = 'conditional sampling and ranks'
                    new_sbc += int(_sbc_case(run, spec, record, case_id, theta[case_id], observation, meta, posterior, weight_hash))
            except Exception as error:
                directory = Path(run) / 'errors'
                suffix = '%s-%s' % (time.time_ns(), os.getpid())
                atomic_json(directory / ('case_%06d_%s.json' % (case_id, suffix)),
                            {'case_id': case_id, 'phase': phase, 'fingerprint': record['fingerprint'],
                             'error_type': type(error).__name__, 'message': str(error),
                             'policy': 'Retain and retry the same pre-drawn case; no replacement truth.'})
                raise
    return {'operation': spec['operation'], 'mode': 'executed', 'case_range': [start, stop],
            'new_paired_observations': new_observations, 'verified_existing_pairs': reused,
            'condition_specific_simulations': 2 * new_observations, 'new_sbc_cases': new_sbc,
            'training_runs': 0}
