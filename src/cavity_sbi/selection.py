"""Stable meta-score selection of fit/task winners and next-round proposals."""
from pathlib import Path
import os
import shutil

from .artifacts import file_sha256
from .scoring import scores_from_quantiles
from .workflow import atomic_json, checked_run, lock, read_json, round_path, run_file


def best_candidate(candidates):
    if not candidates:
        raise ValueError('No complete candidates to select')
    # Python max retains the first maximum, matching the original np.argmax.
    return max(range(len(candidates)), key=lambda i: candidates[i]['stored_meta_score'])


def checked_candidate(record, run, identity):
    if record['fingerprint'] != identity:
        raise ValueError('Candidate belongs to another run')
    score = scores_from_quantiles(record['quartiles'])
    if (score.stored_meta_score != record['stored_meta_score'] or score.r25 != record['r25']
            or score.meta_score != record['raw_meta_score']):
        raise ValueError('Candidate score does not match its saved quartiles')
    path = run_file(run, record['posterior_file'])
    if file_sha256(path) != record['posterior_sha256']:
        raise ValueError('Candidate posterior checksum differs')
    return record


def checked_task(run, spec, round_index, task_id):
    identity = checked_run(run, spec)['fingerprint']
    path = round_path(run, round_index) / ('task_%02d' % task_id) / 'task.json'
    record = read_json(path)
    if record['fingerprint'] != identity or record['task_id'] != task_id or record['round_index'] != round_index:
        raise ValueError('Task identity differs')
    fits = record['fits']
    if len(fits) != spec['workflow']['fits_per_task'] or [r['fit_id'] for r in fits] != list(range(len(fits))):
        raise ValueError('Task does not contain both ordered fits')
    for fit in fits:
        checked_candidate(fit, run, identity)
        if (fit['round_index'] != round_index or fit['task_id'] != task_id
                or fit['evaluation_pairs'] != spec['workflow']['evaluation_pairs']):
            raise ValueError('Candidate evaluation ensemble is incomplete')
    if record['selected_fit'] != best_candidate(fits):
        raise ValueError('Stored within-task selection differs')
    return record


def select_round(run, spec, round_index):
    state = checked_run(run, spec)
    directory = round_path(run, round_index)
    with lock(directory / '.selection.lock'):
        tasks = [checked_task(run, spec, round_index, i) for i in range(spec['workflow']['tasks'])]
        candidates = [t['fits'][t['selected_fit']] for t in tasks]
        chosen = best_candidate(candidates)
        winner = candidates[chosen]
        result = {'schema_version': 1, 'fingerprint': state['fingerprint'], 'round_index': round_index,
                  'selected_task': chosen, 'selected_fit': winner['fit_id'],
                  'stored_meta_score': winner['stored_meta_score'], 'raw_meta_score': winner['raw_meta_score'],
                  'r25': winner['r25'], 'quartiles': winner['quartiles'],
                  'posterior_file': str((directory / 'selected_posterior.pkl').relative_to(Path(run))),
                  'posterior_sha256': winner['posterior_sha256'],
                  'proposal_interpretation': 'selected estimator density at the ideal target',
                  'candidate_task_ids': list(range(len(tasks)))}
        destination = directory / 'selection.json'
        if destination.exists():
            if read_json(destination) != result or file_sha256(run_file(run, result['posterior_file'])) != result['posterior_sha256']:
                raise ValueError('Existing round selection differs; preserve it for inspection')
            return result
        source = run_file(run, winner['posterior_file'])
        target = run_file(run, result['posterior_file'])
        temporary = target.with_name(target.name + '.tmp-' + str(os.getpid()))
        try:
            shutil.copyfile(source, temporary)
            if file_sha256(temporary) != result['posterior_sha256']:
                raise ValueError('Selected posterior copy differs')
            os.replace(str(temporary), str(target))
        finally:
            if temporary.exists():
                temporary.unlink()
        atomic_json(destination, result)
    return result
