"""Two independent conditional fits per task and fixed-MAP ensemble evaluation."""
from pathlib import Path
from contextlib import nullcontext
import numpy as np

from .configuration import parameter_declaration
from .datasets import load_training_data, read_case, save_case
from .model import simulate_pair
from .scoring import ensemble_scores, joint_pattern_scores
from .selection import best_candidate, checked_candidate, checked_task
from .workflow import (atomic_json, atomic_posterior, checked_run, fingerprint,
                       load_run_posterior, lock, read_json, round_path)


def _make_inference(prior, summary_writer=None):
    from sbi.inference import SNPE
    return SNPE(prior=prior, summary_writer=summary_writer)


def fit_estimator(theta, scores, model, workflow, seed, log_directory=None):
    import torch
    from importlib import metadata
    from sbi.utils import BoxUniform
    if metadata.version('sbi') != '0.19.2':
        raise RuntimeError('Fitting uses the pinned sbi 0.19.2 configuration')
    torch.set_num_threads(workflow['torch_threads'])
    prior = BoxUniform(low=torch.tensor([p['prior_low'] for p in model['parameters']], dtype=torch.float32),
                       high=torch.tensor([p['prior_high'] for p in model['parameters']], dtype=torch.float32))
    if log_directory is None:
        writer_context = nullcontext(None)
    else:
        from torch.utils.tensorboard import SummaryWriter
        writer_context = SummaryWriter(log_dir=str(log_directory))
    with torch.random.fork_rng(devices=[]), writer_context as writer:
        torch.manual_seed(seed % (2 ** 63))
        inference = _make_inference(prior, summary_writer=writer)
        # Deliberate unweighted conditional fit, as used in the reported workflow.
        inference = inference.append_simulations(torch.as_tensor(theta, dtype=torch.float32),
                                                 torch.as_tensor(scores, dtype=torch.float32), proposal=None)
        density_estimator = inference.train(**workflow['train_kwargs'])
        posterior = inference.build_posterior(density_estimator)
        target = torch.tensor(model['score']['target'], dtype=torch.float32)
        posterior.set_default_x(target)
        samples = posterior.sample((workflow['posterior_summary_draws'],), x=target, show_progress_bars=False)
        median = torch.quantile(samples, 0.5, dim=0).detach().cpu().numpy()
        mape = posterior.map(num_init_samples=workflow['map_num_init_samples'], **workflow['map_kwargs'])
    mape = mape.detach().cpu().reshape(26)
    if not torch.isfinite(mape).all() or not torch.isfinite(prior.log_prob(mape)):
        raise ValueError('MAP lies outside the finite model support')
    posterior._map = mape
    posterior.parameter_set_true = parameter_declaration(model)
    return posterior, mape.numpy(), median


def _fit_candidate(run, spec, round_index, task_id, fit_id, theta, scores, pooled_rounds):
    record = checked_run(run, spec)
    identity = record['fingerprint']
    directory = round_path(run, round_index) / ('task_%02d' % task_id) / ('fit_%02d' % fit_id)
    complete = directory / 'fit.json'
    if complete.exists():
        result = checked_candidate(read_json(complete), run, identity)
        if result['round_index'] != round_index or result['task_id'] != task_id or result['fit_id'] != fit_id:
            raise ValueError('Completed fit identity differs')
        return result
    checkpoint = directory / 'trained.json'
    if checkpoint.exists():
        saved = read_json(checkpoint)
        if (saved['fingerprint'] != identity or saved['training_rounds'] != pooled_rounds
                or saved['round_index'] != round_index or saved['task_id'] != task_id or saved['fit_id'] != fit_id):
            raise ValueError('Trained checkpoint belongs to another run or pool')
        posterior = load_run_posterior(directory / 'trained.pkl', saved['posterior_sha256'], spec['model'])
        mape = np.asarray(saved['map'], dtype=np.float32)
        if not np.array_equal(mape, posterior._map.detach().cpu().numpy()):
            raise ValueError('Trained checkpoint MAP coordinates differ')
    else:
        if (directory / 'trained.pkl').exists():
            raise ValueError('Partial training checkpoint; preserve it for inspection')
        seed = record['random_entropy'] + 1000000 + round_index * 10000 + task_id * 10 + fit_id
        posterior, mape, median = fit_estimator(theta, scores, spec['model'], spec['workflow'], seed,
                                              log_directory=directory / 'tensorboard')
        saved = {'fingerprint': identity, 'round_index': round_index, 'task_id': task_id,
                 'fit_id': fit_id, 'training_rounds': pooled_rounds, 'training_pairs': len(theta),
                 'tensorboard_directory': str((directory / 'tensorboard').relative_to(Path(run))),
                 'map': mape.tolist(), 'median': median.tolist(),
                 'posterior_sha256': atomic_posterior(directory / 'trained.pkl', posterior)}
        atomic_json(checkpoint, saved)
    values = []
    count = spec['workflow']['evaluation_pairs']
    evaluation_identity = fingerprint({'run': identity, 'stage': 'evaluation',
                                      'round_index': round_index, 'task_id': task_id, 'fit_id': fit_id})
    for case_id in range(count):
        path = directory / 'evaluation' / ('case_%06d.npz' % case_id)
        if path.exists() and path.with_suffix('.json').exists():
            _, score = read_case(path, evaluation_identity, case_id, spec['model'], mape)
        else:
            if path.exists() or path.with_suffix('.json').exists():
                raise ValueError('Partial evaluation output; preserve it for inspection')
            seed = [record['random_entropy'], round_index, 100000 + task_id * 10 + fit_id, case_id]
            result = simulate_pair(mape, tuple(spec['geometry']['layers']), config=spec['model'], seed=seed)
            score = joint_pattern_scores(result.flat, result.geometry, config=spec['model']).reshape(49)
            save_case(path, mape, score, result.flat, evaluation_identity, case_id, seed, spec['workflow']['keep_raw_trajectories'])
        values.append(score)
    values = np.asarray(values, dtype=np.float64)
    summary = ensemble_scores(values)
    quartiles = np.percentile(values, [25, 50, 75], axis=0, method='nearest').T
    posterior.appraisal = summary.stored_meta_score
    posterior.percentiles = quartiles
    posterior.parameter_set_true = parameter_declaration(spec['model'])
    path = directory / 'posterior.pkl'
    result = {'fingerprint': identity, 'round_index': round_index, 'task_id': task_id, 'fit_id': fit_id,
              'training_rounds': pooled_rounds, 'training_pairs': len(theta), 'evaluation_pairs': count,
              'raw_meta_score': summary.meta_score, 'stored_meta_score': summary.stored_meta_score,
              'r25': summary.r25, 'quartiles': quartiles.tolist(),
              'posterior_file': str(path.relative_to(Path(run))),
              'posterior_sha256': atomic_posterior(path, posterior)}
    atomic_json(complete, result)
    return result


def train_task(run, spec, round_index, task_id):
    record = checked_run(run, spec)
    directory = round_path(run, round_index) / ('task_%02d' % task_id)
    with lock(directory / '.task.lock'):
        if (directory / 'task.json').exists():
            return checked_task(run, spec, round_index, task_id)
        theta, scores, rounds = load_training_data(run, spec, round_index)
        fits = []
        for fit_id in range(spec['workflow']['fits_per_task']):
            try:
                fits.append(_fit_candidate(run, spec, round_index, task_id, fit_id, theta, scores, rounds))
            except Exception as error:
                atomic_json(directory / ('fit_%02d_error.json' % fit_id),
                            {'fingerprint': record['fingerprint'], 'fit_id': fit_id,
                             'error_type': type(error).__name__, 'message': str(error)})
                raise
        result = {'fingerprint': record['fingerprint'], 'round_index': round_index,
                  'task_id': task_id, 'selected_fit': best_candidate(fits), 'fits': fits}
        atomic_json(directory / 'task.json', result)
    return result
