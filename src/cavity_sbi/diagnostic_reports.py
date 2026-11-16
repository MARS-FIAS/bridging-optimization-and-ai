"""Rebuild complete diagnostic reports from persisted results, without a model."""
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from .diagnostic_stats import calibration_summary, predictive_summary
from .diagnostics import (case_count, case_identity, checked_diagnostic,
                          read_array_record, _validate_theta, _validate_observation)
from .workflow import atomic_json, atomic_npz, lock


def summarize_diagnostic(run):
    """Read every planned case; fail on missing, corrupt or mismatched evidence."""
    run = Path(run)
    record = checked_diagnostic(run)
    spec, identity = record['specification'], record['fingerprint']
    cfg, model = spec['settings'], spec['model']
    count = case_count(spec)
    bank, bank_meta = read_array_record(run / 'parameters.npz', case_identity(record, 'parameters'))
    _validate_theta(bank['theta'], spec, count)
    keys = [p['key'] for p in model['parameters']]
    if bank['parameter_keys'].tolist() != keys:
        raise ValueError('Report parameter order differs')
    scores, ranks, pits, coverage, hashes = [], [], [], [], {}
    weight_hashes = set()
    for i in range(count):
        path = run / 'observations' / ('case_%06d.npz' % i)
        observation, meta = read_array_record(path, case_identity(record, 'observation'), i)
        _validate_observation(observation, meta)
        if meta['parameter_sha256'] != bank_meta['sha256'] or not np.array_equal(observation['theta'], bank['theta'][i]):
            raise ValueError('Report observation differs from the fixed parameter draw')
        scores.append(observation['joint_score'])
        hashes[str(path.relative_to(run))] = meta['sha256']
        if spec['operation'] == 'sbc':
            path = run / 'diagnostics' / ('case_%06d.npz' % i)
            arrays, diagnostic_meta = read_array_record(path, case_identity(record, 'sbc'), i)
            if (diagnostic_meta['observation_sha256'] != meta['sha256']
                    or diagnostic_meta['posterior_draws'] != cfg['posterior_draws_per_case']
                    or not np.array_equal(arrays['truth'], bank['theta'][i])
                    or not np.array_equal(arrays['observation'], observation['joint_score'])):
                raise ValueError('Calibration report truth/observation identity differs')
            low, high = arrays['central_interval_low'], arrays['central_interval_high']
            covered = arrays['central_interval_contains_truth']
            if (low.shape != (len(cfg['coverage_levels']), 26) or high.shape != low.shape
                    or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high)
                    or not np.array_equal(covered, (arrays['truth'] >= low) & (arrays['truth'] <= high))):
                raise ValueError('Stored credible-interval membership is inconsistent')
            ranks.append(arrays['ranks'])
            pits.append(arrays['rank_pit'])
            coverage.append(covered)
            weight_hashes.add(diagnostic_meta['weights_sha256'])
            hashes[str(path.relative_to(run))] = diagnostic_meta['sha256']
    scores = np.stack(scores)
    ids = np.arange(count)
    predictive = predictive_summary(scores, ids, count, spec['operation'])
    if spec['operation'] == 'sbc':
        if len(weight_hashes) != 1:
            raise ValueError('Calibration cases use different frozen neural weights')
        result = calibration_summary(np.stack(ranks), np.stack(pits), np.stack(coverage), ids,
                                     keys, cfg['posterior_draws_per_case'], cfg['coverage_levels'], count, cfg['alpha'])
        result['weights_sha256'] = next(iter(weight_hashes))
        result['prior_predictive_reference'] = predictive
    else:
        result = predictive
    result.update({'fingerprint': identity, 'geometry': spec['geometry']['id'],
                   'estimator_sha256': spec['estimator']['sha256'],
                   'execution_environment': record['execution_environment'],
                   'created_utc': datetime.now(timezone.utc).isoformat(),
                   'parameter_keys': keys, 'score_time_hours': model['score']['time_hours'],
                   'input_sha256': dict(hashes, **{'parameters.npz': bank_meta['sha256']}),
                   'failed_attempt_records': sorted(p.name for p in (run / 'errors').glob('*.json')),
                   'new_simulations': 0, 'new_posterior_draws': 0, 'training_runs': 0})
    with lock(run / 'reports/.report.lock'):
        common = {'case_ids': ids, 'parameter_keys': np.asarray(keys),
                  'score_time_hours': np.asarray(model['score']['time_hours'])}
        atomic_npz(run / 'reports/predictive_scores.npz', scores=scores, **common)
        if spec['operation'] == 'sbc':
            atomic_npz(run / 'reports/sbc_rank_arrays.npz', ranks=np.stack(ranks), rank_pit=np.stack(pits),
                       coverage=np.stack(coverage), coverage_levels=np.asarray(cfg['coverage_levels']),
                       posterior_draws_per_case=np.asarray(cfg['posterior_draws_per_case']), **common)
        atomic_json(run / 'reports/summary.json', result)
    return result
