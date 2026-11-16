"""Portable run configuration, manifests and atomic output helpers."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import secrets

from . import __version__
from .artifacts import file_sha256
from .configuration import get_geometry, model_config, project_file


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def run_file(run, relative):
    root = Path(run).resolve()
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = path.open('x')
    except FileExistsError:
        raise RuntimeError('Another writer or an interrupted operation holds ' + str(path))
    try:
        stream.write(str(os.getpid()))
        stream.close()
        yield
    finally:
        path.unlink()


def workflow_config(project='.', relative='config/workflow.json'):
    # Profiles may be supplied outside the source checkout; they are read only.
    path = Path(relative)
    path = path if path.is_absolute() else Path(project) / path
    cfg = read_json(path)
    for key in ['rounds', 'simulation_pairs_per_round', 'tasks', 'fits_per_task',
                'posterior_summary_draws', 'map_num_init_samples', 'evaluation_pairs', 'torch_threads']:
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError('Expected a positive integer for ' + key)
    if cfg['rounds'] > 5 or cfg['fits_per_task'] != 2:
        raise ValueError('The supplied workflow uses at most five rounds and two fits per task')
    if cfg['seed'] is not None and (type(cfg['seed']) is not int or cfg['seed'] < 0):
        raise ValueError('An optional seed must be a nonnegative integer')
    if type(cfg['keep_raw_trajectories']) is not bool:
        raise ValueError('keep_raw_trajectories must be boolean')
    train_keys = {'training_batch_size', 'learning_rate', 'validation_fraction', 'stop_after_epochs',
                  'max_num_epochs', 'clip_max_norm', 'show_train_summary'}
    map_keys = {'num_iter', 'num_to_optimize', 'learning_rate', 'show_progress_bars', 'save_best_every'}
    if set(cfg['train_kwargs']) - train_keys or set(cfg['map_kwargs']) - map_keys:
        raise ValueError('Unsupported fitting or MAP options')
    return cfg


def specification(project, geometry, profile):
    code = {str(p.relative_to(Path(__file__).parent)): file_sha256(p)
            for p in sorted(Path(__file__).parent.rglob('*.py'))}
    return {'schema_version': 1, 'package_version': __version__,
            'model': model_config(project), 'geometry': get_geometry(geometry, project),
            'workflow': workflow_config(project, profile), 'code_sha256': code}


def validate_run_location(project, run):
    project, run = Path(project).resolve(), Path(run).resolve()
    if run == project:
        raise ValueError('Use a separate output directory for a run')
    for name in ['src', 'config', 'artifacts', 'Script_Bank', 'environments', 'docs',
                 'provenance', 'evidence', 'figures', 'paper', 'tests', 'tools', '.git']:
        protected = project / name
        if run == protected or protected in run.parents:
            raise ValueError('Run outputs must not overwrite source/configuration/artifacts')
    return run


def ensure_run(project, run, spec):
    run = validate_run_location(project, run)
    if (run / 'run.json').exists():
        return checked_run(run, spec)
    run.mkdir(parents=True, exist_ok=True)
    with lock(run / '.run.lock'):
        path = run / 'run.json'
        identity = fingerprint(spec)
        if path.exists():
            existing = read_json(path)
            if existing['fingerprint'] != identity or existing['specification'] != spec:
                raise ValueError('Run code or configuration differs; use a new run directory')
        else:
            seed = spec['workflow']['seed']
            existing = {'fingerprint': identity, 'specification': spec,
                        'random_entropy': secrets.randbits(63) if seed is None else seed}
            profile_path = run / 'workflow.json'
            if profile_path.exists() and read_json(profile_path) != spec['workflow']:
                raise ValueError('Existing workflow profile differs; use a new run directory')
            atomic_json(profile_path, spec['workflow'])
            atomic_json(path, existing)
    return existing


def checked_run(run, spec):
    record = read_json(Path(run) / 'run.json')
    if record['fingerprint'] != fingerprint(spec) or record['specification'] != spec:
        raise ValueError('Run code or configuration differs')
    return record


def round_path(run, round_index):
    return Path(run) / ('round_%03d' % round_index)


def selected_record(run, round_index, identity):
    path = round_path(run, round_index) / 'selection.json'
    value = read_json(path)
    if value['fingerprint'] != identity or value['round_index'] != round_index:
        raise ValueError('Selected proposal belongs to another run/round')
    posterior = run_file(run, value['posterior_file'])
    if file_sha256(posterior) != value['posterior_sha256']:
        raise ValueError('Selected posterior checksum differs')
    return value


def load_run_posterior(path, expected_sha, model):
    if file_sha256(Path(path)) != expected_sha:
        raise ValueError('Run posterior checksum differs')
    from importlib import metadata
    if metadata.version('sbi') != '0.19.2' or metadata.version('torch').split('+')[0] != '1.11.0':
        raise RuntimeError('Use the pinned sbi 0.19.2 / torch 1.11.0 environment')
    import pickle
    import numpy as np
    with Path(path).open('rb') as stream:
        posterior = pickle.load(stream)
    if type(posterior).__name__ != 'DirectPosterior':
        raise ValueError('Expected a run-generated sbi DirectPosterior')
    if list(posterior.parameter_set_true) != [p['key'] for p in model['parameters']]:
        raise ValueError('Run posterior parameter order differs')
    prior = getattr(posterior.prior, 'base_dist', posterior.prior)
    for attr, key in [('low', 'prior_low'), ('high', 'prior_high')]:
        if not np.array_equal(getattr(prior, attr).detach().cpu().numpy(), [p[key] for p in model['parameters']]):
            raise ValueError('Run posterior prior support differs')
    posterior.posterior_estimator.to('cpu').eval()
    return posterior


def atomic_posterior(path, posterior):
    import pickle
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    try:
        with tmp.open('wb') as stream:
            pickle.dump(posterior, stream)
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()
    return file_sha256(path)


def plan(operation, project, geometry, round_index, run, profile='config/workflow.json', task=0, start=0, stop=None):
    spec = specification(project, geometry, profile)
    cfg = spec['workflow']
    run = validate_run_location(project, run)
    if not 0 <= round_index < cfg['rounds']:
        raise ValueError('Round index is outside the configured workflow')
    if not 0 <= task < cfg['tasks']:
        raise ValueError('Task index is outside the configured workflow')
    stop = cfg['simulation_pairs_per_round'] if stop is None else stop
    if not 0 <= start < stop <= cfg['simulation_pairs_per_round']:
        raise ValueError('Invalid generation case range')
    return {'operation': operation, 'mode': 'preview', 'geometry': geometry,
            'round_index': round_index, 'task_id': task, 'fingerprint': fingerprint(spec),
            'run_directory': str(run), 'profile': cfg['label'],
            'generation_case_range': [start, stop],
            'generation_condition_specific_calls': 2 * (stop - start),
            'fitting_data_rounds': [0] if round_index == 0 else [round_index - 1, round_index],
            'training_pairs_when_complete': cfg['simulation_pairs_per_round'] * (1 if round_index == 0 else 2),
            'fits_per_task': cfg['fits_per_task'], 'tasks_per_round': cfg['tasks'],
            'candidate_evaluation_pairs': cfg['evaluation_pairs'],
            'evaluation_condition_specific_calls_per_task': 2 * cfg['fits_per_task'] * cfg['evaluation_pairs'],
            'next_proposal': 'selected estimator density conditioned on the ideal target',
            'execute_required': True, 'source_or_run_files_written': False}
