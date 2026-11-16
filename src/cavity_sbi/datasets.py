"""Paired simulation generation and loading of adjacent-round score datasets."""
from pathlib import Path
import numpy as np

from .artifacts import file_sha256
from .model import simulate_pair
from .scoring import joint_pattern_scores
from .workflow import (atomic_json, atomic_npz, checked_run, ensure_run, fingerprint,
                       load_run_posterior, lock, read_json, round_path, run_file,
                       selected_record)


def _sample_parameters(spec, run, round_index, entropy):
    import torch
    from sbi.utils import BoxUniform
    model = spec['model']
    count = spec['workflow']['simulation_pairs_per_round']
    torch.set_num_threads(spec['workflow']['torch_threads'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed((entropy + round_index) % (2 ** 63))
        if round_index == 0:
            proposal = BoxUniform(low=torch.tensor([p['prior_low'] for p in model['parameters']], dtype=torch.float32),
                                  high=torch.tensor([p['prior_high'] for p in model['parameters']], dtype=torch.float32))
            values = proposal.sample((count,))
        else:
            selected = selected_record(run, round_index - 1, fingerprint(spec))
            proposal = load_run_posterior(run_file(run, selected['posterior_file']), selected['posterior_sha256'], model)
            values = proposal.sample((count,), x=torch.tensor(model['score']['target'], dtype=torch.float32))
    return values.detach().cpu().numpy().astype(np.float32)


def _validate_theta(theta, model):
    if theta.shape[-1] != 26 or not np.isfinite(theta).all():
        raise ValueError('Invalid parameter shape or values')
    low = np.asarray([p['prior_low'] for p in model['parameters']])
    high = np.asarray([p['prior_high'] for p in model['parameters']])
    if np.any(theta < low) or np.any(theta > high):
        raise ValueError('Parameters lie outside the model prior')


def prepare_round(project, run, spec, round_index):
    record = ensure_run(project, run, spec)
    directory = round_path(run, round_index)
    manifest_path = directory / 'generation.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest['fingerprint'] != record['fingerprint'] or manifest['round_index'] != round_index:
            raise ValueError('Generation belongs to another run/round')
        if file_sha256(directory / 'parameters.npz') != manifest['parameter_sha256']:
            raise ValueError('Fixed round parameters changed')
        if round_index > 0 and manifest['proposal_sha256'] != selected_record(run, round_index - 1, record['fingerprint'])['posterior_sha256']:
            raise ValueError('The preceding selected proposal has changed')
        return manifest
    with lock(directory / '.prepare.lock'):
        if manifest_path.exists():
            return prepare_round(project, run, spec, round_index)
        proposal = None if round_index == 0 else selected_record(run, round_index - 1, record['fingerprint'])
        theta = _sample_parameters(spec, run, round_index, record['random_entropy'])
        expected = (spec['workflow']['simulation_pairs_per_round'], 26)
        if theta.shape != expected:
            raise ValueError('Unexpected sampled parameter shape')
        _validate_theta(theta, spec['model'])
        atomic_npz(directory / 'parameters.npz', theta=theta)
        manifest = {'schema_version': 1, 'fingerprint': record['fingerprint'], 'round_index': round_index,
                    'cases': len(theta), 'parameter_sha256': file_sha256(directory / 'parameters.npz'),
                    'proposal': 'box prior' if proposal is None else 'previous selected goal-conditioned density',
                    'proposal_sha256': None if proposal is None else proposal['posterior_sha256']}
        atomic_json(manifest_path, manifest)
    return manifest


def save_case(path, theta, score, raw, identity, case_id, seed, keep_raw):
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if score.shape != (49,) or not np.isfinite(score).all() or np.any(score < 0) or np.any(score > 1):
        raise ValueError('Invalid 49-component joint pattern score')
    arrays = {'theta': np.asarray(theta, dtype=np.float32), 'joint_score': score,
              'stream_entropy': np.asarray(seed, dtype=np.uint64)}
    if keep_raw:
        raw = np.asarray(raw)
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite molecular trajectories')
        arrays['raw_trajectories'] = raw
    atomic_npz(path, **arrays)
    metadata = {'fingerprint': identity, 'case_id': case_id,
                'sha256': file_sha256(Path(path)), 'raw_trajectories_recorded': keep_raw}
    atomic_json(Path(path).with_suffix('.json'), metadata)


def read_case(path, identity, case_id, model, expected_theta=None):
    path = Path(path)
    metadata = read_json(path.with_suffix('.json'))
    if metadata['fingerprint'] != identity or metadata['case_id'] != case_id:
        raise ValueError('Case identity differs')
    if metadata['sha256'] != file_sha256(path):
        raise ValueError('Case checksum differs')
    with np.load(path, allow_pickle=False) as saved:
        theta = np.array(saved['theta'], dtype=np.float32)
        score = np.array(saved['joint_score'], dtype=np.float64)
    if theta.shape != (26,):
        raise ValueError('Invalid case parameter dimensions')
    _validate_theta(theta, model)
    if expected_theta is not None and not np.array_equal(theta, expected_theta):
        raise ValueError('Case parameters differ from the fixed round draw')
    if score.shape != (49,) or not np.isfinite(score).all() or np.any(score < 0) or np.any(score > 1):
        raise ValueError('Invalid case score')
    return theta, score


def generate(project, run, spec, round_index, start=0, stop=None, prepare_only=False):
    manifest = prepare_round(project, run, spec, round_index)
    if prepare_only:
        return {'operation': 'generate', 'mode': 'prepared parameters', 'round_index': round_index,
                'parameter_pairs': manifest['cases'], 'condition_specific_simulations': 0}
    count = spec['workflow']['simulation_pairs_per_round']
    stop = count if stop is None else stop
    if not 0 <= start < stop <= count:
        raise ValueError('Invalid generation range')
    record = checked_run(run, spec)
    case_identity = fingerprint({'run': record['fingerprint'], 'stage': 'generation', 'round_index': round_index})
    directory = round_path(run, round_index)
    with np.load(directory / 'parameters.npz', allow_pickle=False) as saved:
        theta = np.array(saved['theta'], dtype=np.float32)
    complete = reused = 0
    for case_id in range(start, stop):
        path = directory / 'cases' / ('case_%06d.npz' % case_id)
        with lock(path.with_suffix('.lock')):
            if path.exists() and path.with_suffix('.json').exists():
                read_case(path, case_identity, case_id, spec['model'], theta[case_id])
                reused += 1
                continue
            if path.exists() or path.with_suffix('.json').exists():
                raise ValueError('Incomplete case output; preserve it for inspection before retrying: ' + str(path))
            seed = [record['random_entropy'], round_index, case_id]
            try:
                result = simulate_pair(theta[case_id], tuple(spec['geometry']['layers']), config=spec['model'], seed=seed)
                score = joint_pattern_scores(result.flat, result.geometry, config=spec['model']).reshape(49)
                save_case(path, theta[case_id], score, result.flat, case_identity, case_id,
                          seed, spec['workflow']['keep_raw_trajectories'])
                complete += 1
            except Exception as error:
                atomic_json(directory / 'errors' / ('case_%06d.json' % case_id),
                            {'case_id': case_id, 'fingerprint': record['fingerprint'],
                             'error_type': type(error).__name__, 'message': str(error)})
                raise
    return {'operation': 'generate', 'mode': 'executed', 'round_index': round_index,
            'new_pairs': complete, 'verified_existing_pairs': reused,
            'condition_specific_simulations': 2 * complete}


def load_training_data(run, spec, round_index):
    record = checked_run(run, spec)
    rounds = [0] if round_index == 0 else [round_index - 1, round_index]
    theta, scores = [], []
    for index in rounds:
        directory = round_path(run, index)
        manifest = read_json(directory / 'generation.json')
        if (manifest['fingerprint'] != record['fingerprint'] or manifest['round_index'] != index
                or manifest['cases'] != spec['workflow']['simulation_pairs_per_round']):
            raise ValueError('Training dataset configuration differs')
        if file_sha256(directory / 'parameters.npz') != manifest['parameter_sha256']:
            raise ValueError('Fixed dataset parameters changed')
        with np.load(directory / 'parameters.npz', allow_pickle=False) as saved:
            expected = np.array(saved['theta'], dtype=np.float32)
        if expected.shape != (manifest['cases'], 26):
            raise ValueError('Fixed parameter-bank dimensions differ')
        _validate_theta(expected, spec['model'])
        case_identity = fingerprint({'run': record['fingerprint'], 'stage': 'generation', 'round_index': index})
        for case_id in range(manifest['cases']):
            t, x = read_case(directory / 'cases' / ('case_%06d.npz' % case_id),
                             case_identity, case_id, spec['model'], expected[case_id])
            theta.append(t)
            scores.append(x)
    return np.asarray(theta, dtype=np.float32), np.asarray(scores, dtype=np.float32), rounds
