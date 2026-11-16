"""Reconstruct scores, selection, cell-fate fractions and parameter plot data."""
from collections import Counter
import numpy as np

from .configuration import geometry_layout
from .diagnostic_stats import predictive_summary
from .scoring import scores_from_quantiles
from .selection import best_candidate


def selection_analysis(data, geometries, model):
    q = data['quartiles']
    if q.shape != (400, 49, 3) or not np.array_equal(data['time_h'], model['score']['time_hours']):
        raise ValueError('Expected all 400 retained candidate quartile trajectories')
    if data['parameter_keys'].tolist() != [p['key'] for p in model['parameters']]:
        raise ValueError('Candidate parameter order differs')
    for key in ['geometry', 'round_index', 'task_id', 'stored_S', 'selected', 'estimator_sha256']:
        if data[key].shape != (400,):
            raise ValueError('Candidate identity shape differs: ' + key)
    if data['selected'].dtype.kind != 'b' or data['round_index'].dtype.kind not in 'iu' or data['task_id'].dtype.kind not in 'iu':
        raise ValueError('Invalid candidate selection/index types')
    candidates, groups = [], []
    for i in range(400):
        score = scores_from_quantiles(q[i])
        if score.stored_meta_score != data['stored_S'][i]:
            raise ValueError('Stored S differs from retained quartiles')
        candidates.append({'geometry': str(data['geometry'][i]), 'round_number': int(data['round_index'][i])+1,
            'task_id': int(data['task_id'][i]), 'raw_S': score.meta_score,
            'stored_meta_score': score.stored_meta_score, 'R25': score.r25,
            'selected': bool(data['selected'][i]), 'estimator_sha256': str(data['estimator_sha256'][i])})
    for geometry in geometries:
        for round_number in range(1, 6):
            rows = sorted((r for r in candidates if r['geometry'] == geometry['id'] and r['round_number'] == round_number),
                          key=lambda r: r['task_id'])
            if [r['task_id'] for r in rows] != list(range(10)):
                raise ValueError('Every geometry/round must contain all ten retained task winners')
            winner = rows[best_candidate(rows)]
            if [r['task_id'] for r in rows if r['selected']] != [winner['task_id']]:
                raise ValueError('Retained selected task differs from stored-S first-maximum policy')
            r25_winner = max(rows, key=lambda r: r['R25'])
            gain = r25_winner['R25'] - winner['R25']
            category = 'same' if winner['task_id'] == r25_winner['task_id'] else 'tie_only' if gain == 0 else 'strict_difference'
            groups.append({'geometry': geometry['id'], 'cells': geometry['physical_ICM_cells'], 'round_number': round_number,
                'selected_task': winner['task_id'], 'R25_first_task': r25_winner['task_id'], 'category': category,
                'S': winner['stored_meta_score'], 'raw_S': winner['raw_S'], 'R25': winner['R25'],
                'best_R25': r25_winner['R25'], 'R25_gain': gain,
                'S_tied_tasks': '|'.join(str(r['task_id']) for r in rows if r['stored_meta_score'] == winner['stored_meta_score']),
                'R25_tied_tasks': '|'.join(str(r['task_id']) for r in rows if r['R25'] == r25_winner['R25']),
                'selected_estimator_sha256': winner['estimator_sha256']})
    counts = Counter(row['category'] for row in groups)
    summary = {'retained_candidates': 400, 'groups': 40, 'same_first_task': counts['same'],
               'tie_only': counts['tie_only'], 'strict_difference': counts['strict_difference'],
               'S_winner_maximizes_R25': counts['same']+counts['tie_only'],
               'final_round_agreement': sum(r['category'] != 'strict_difference' for r in groups if r['round_number'] == 5),
               'scope': 'Retrospective comparison of retained task winners on the S-selected sequential path.'}
    return summary, groups, candidates


def fate_counts_from_raw(raw, layers, model):
    """Count classified physical ICM cells; no event simulation or neural loading."""
    layout = geometry_layout(layers)
    values = np.asarray(raw).reshape(2, 3, 196, layout['total_storage_positions'])[:, :, 144:193]
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Invalid molecular observations for fate classification')
    cfg = model['score']
    nt, g = values[:, 0]+values[:, 2], values[:, 1]
    epi = ((nt > cfg['positive_NT']-cfg['cusp_NT']*np.sqrt(cfg['positive_NT']))
           & (g < cfg['negative_G']+cfg['cusp_G']*np.sqrt(cfg['negative_G'])))
    pre = ((nt < cfg['negative_NT']+cfg['cusp_NT']*np.sqrt(cfg['negative_NT']))
           & (g > cfg['positive_G']-cfg['cusp_G']*np.sqrt(cfg['positive_G'])))
    active = layout['active_ICM_indices']
    lower = layers[0]
    groups = [active[:lower], active[lower:]]
    result = np.stack([np.stack([epi[:, :, group].sum(axis=-1), pre[:, :, group].sum(axis=-1)], axis=-1)
                       for group in groups], axis=1)
    return result.astype(np.uint8)


def predictive_analysis(data, model, expected_cases=None, operation='predictive'):
    scores, ids, times = data['scores'], data['case_ids'], data['time_h']
    count = len(ids) if expected_cases is None else expected_cases
    if not np.array_equal(times, model['score']['time_hours']):
        raise ValueError('Predictive evaluation times differ')
    summary = predictive_summary(scores, ids, count, operation)
    intended = None
    if 'fate_counts' in data:
        counts, denominator = data['fate_counts'], data['group_cell_counts']
        if (counts.shape != (count, 2, 2, 49, 2) or counts.dtype.kind not in 'ui'
                or denominator.shape != (2,) or denominator.dtype.kind not in 'ui' or np.any(denominator <= 0)
                or np.any(counts.sum(axis=-1) > denominator[None, None, :, None])):
            raise ValueError('Invalid cell-fate count dimensions or totals')
        if 'condition_rules' in data and not np.array_equal(data['condition_rules'], [0, 1]):
            raise ValueError('Biological condition order differs')
        if 'fate_keys' in data and list(data['fate_keys']) != ['EPI', 'PRE']:
            raise ValueError('Classified fate order differs')
        if 'geometry' in data:
            layers = tuple(map(int, str(data['geometry']).split('_')))
            if not np.array_equal(denominator, [layers[0], sum(layers)-layers[0]]):
                raise ValueError('Physical cell-group counts differ from the declared geometry')
        fractions = counts / denominator[None, None, :, None, None]
        target_nt = np.asarray(model['score']['aim_NT'])
        targets = np.stack([target_nt, 1-target_nt], axis=-1)
        error = np.abs(fractions-targets[None, :, :, None, :]).sum(axis=-1)
        beta = model['score']['beta']
        manual = ((np.exp(-beta*error)-np.exp(-2*beta))/(1-np.exp(-2*beta))).min(axis=(1, 2))
        if not np.allclose(manual, scores, atol=1e-12, rtol=0):
            raise ValueError('Independent reconstruction from fate counts disagrees with joint scores')
        intended = [float(fractions[:, 0, 0, :, 0].mean()), float(fractions[:, 0, 1, :, 0].mean()),
                    float(fractions[:, 1, 0, :, 1].mean()), float(fractions[:, 1, 1, :, 0].mean())]
        summary['mean_intended_fate_fractions'] = dict(zip(
            ['knockout_lower_EPI', 'knockout_upper_EPI', 'wild_type_lower_PRE', 'wild_type_upper_EPI'], intended))
        summary['independent_scores_from_fate_counts_verified'] = count
    hist, edges = np.histogram(scores.mean(axis=1), bins=np.linspace(0, 1, 21))
    timeseries = [{'time_h': float(t), 'mean': summary['pointwise_mean'][i],
                  'q25': summary['pointwise_quantiles_q25_q50_q75'][0][i],
                  'median': summary['pointwise_quantiles_q25_q50_q75'][1][i],
                  'q75': summary['pointwise_quantiles_q25_q50_q75'][2][i],
                  'target_fraction': summary['pointwise_target_attainment'][i]} for i, t in enumerate(times)]
    histogram = [{'bin_lower': float(edges[i]), 'bin_upper': float(edges[i+1]), 'count': int(n)} for i, n in enumerate(hist)]
    return summary, timeseries, histogram, intended


def parameter_analysis(map_rows, reference_rows, model, geometries):
    keys = [p['key'] for p in model['parameters']]
    expected = {(g['id'], key) for g in geometries for key in keys}
    if len(map_rows) != 208 or {(r['geometry'], r['parameter']) for r in map_rows} != expected:
        raise ValueError('Expected all 208 geometry-specific selected MAP coordinates')
    if len(reference_rows) != 26 or [r['parameter'] for r in reference_rows] != keys:
        raise ValueError('Expected all 26 reference-density marginals in model order')
    parameters = {p['key']: p for p in model['parameters']}
    for row in map_rows+reference_rows:
        p = parameters[row['parameter']]
        if float(row['prior_low']) != p['prior_low'] or float(row['prior_high']) != p['prior_high']:
            raise ValueError('Parameter evidence prior differs')
        tags = ['MAP'] if 'MAP_native' in row else ['q025', 'q25', 'median', 'q75', 'q975']
        values = []
        for tag in tags:
            native, normalized = float(row[tag+'_native']), float(row[tag+'_prior_percent'])
            if (not np.isfinite(native) or not p['prior_low'] <= native <= p['prior_high']
                    or not np.isclose(normalized, 100*(native-p['prior_low'])/(p['prior_high']-p['prior_low']), atol=1e-12, rtol=0)):
                raise ValueError('Invalid native/prior-range parameter value')
            values.append(native)
        if any(a > b for a, b in zip(values, values[1:])):
            raise ValueError('Reference marginal quantiles are not ordered')
        if 'draws' in row and int(row['draws']) != 256:
            raise ValueError('Reference summaries must identify the 256 ideal-target draws')
    fl = next(r for r in reference_rows if r['parameter'] == 'FL')
    return {'selected_MAP_coordinates': 208, 'reference_marginals': 26, 'reference_parameter_draws': 256,
            'reference_geometry': '5_3_1', 'FL_IQR_native': [float(fl['q25_native']), float(fl['q75_native'])],
            'FL_IQR_prior_percent': [float(fl['q25_prior_percent']), float(fl['q75_prior_percent'])],
            'interpretation': 'Fitted-density summaries, not a calibrated-coverage or necessity claim.'}
