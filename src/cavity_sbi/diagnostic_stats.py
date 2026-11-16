"""Predictive attainment and frozen-estimator marginal calibration statistics.

These functions only process arrays. They never load a network or simulate.
Rank randomization, central intervals and Holm adjustment retain the completed
CPU diagnostic's arithmetic; nearest quartiles retain the selection policy.
"""
import numpy as np
from .scoring import ensemble_scores


def _levels(levels):
    levels = np.asarray(levels, dtype=float)
    if (levels.ndim != 1 or len(levels) == 0 or not np.isfinite(levels).all()
            or np.any(levels <= 0) or np.any(levels >= 1) or np.any(np.diff(levels) <= 0)):
        raise ValueError('Coverage levels must increase strictly within (0,1)')
    return levels


def _case_ids(case_ids, count):
    ids = np.asarray(case_ids)
    if ids.dtype.kind not in 'iu' or not np.array_equal(ids, np.arange(count)):
        raise ValueError('A complete report requires every planned case in order')


def ranks_and_intervals(samples, truth, levels, rng):
    """Randomize finite-precision ties, jitter ranks and assess central intervals."""
    samples, truth = np.asarray(samples), np.asarray(truth)
    levels = _levels(levels)
    if (samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] == 0
            or truth.shape != (samples.shape[1],) or not np.isfinite(samples).all()
            or not np.isfinite(truth).all()):
        raise ValueError('Expected finite draw-by-parameter samples and one truth')
    lower = np.sum(samples < truth, axis=0)
    equal = np.sum(samples == truth, axis=0)
    ranks = lower + np.asarray([rng.integers(0, int(k) + 1) for k in equal])
    pit = (ranks + rng.uniform(size=len(truth))) / (len(samples) + 1)
    quantile_levels = [(1-c)/2 for c in levels] + [(1+c)/2 for c in levels]
    quantiles = np.quantile(samples, quantile_levels, axis=0)
    lo, hi = quantiles[:len(levels)], quantiles[len(levels):]
    return {'ranks': ranks, 'rank_pit': pit, 'ties': equal,
            'central_interval_low': lo, 'central_interval_high': hi,
            'central_interval_contains_truth': (truth >= lo) & (truth <= hi)}


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Expected finite p-values in [0,1]')
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    return adjusted


def predictive_summary(scores, case_ids, expected_cases, operation='predictive'):
    values = np.asarray(scores, dtype=float)
    if values.shape != (expected_cases, 49):
        raise ValueError('Predictive reports require every planned 49-component score')
    _case_ids(case_ids, expected_cases)
    result = ensemble_scores(values)
    perfect = np.all(np.abs(values - 1) <= 1e-12, axis=1)
    quantiles = np.percentile(values, [25, 50, 75], axis=0, method='nearest')
    return {'operation': operation, 'complete': True, 'completed_parameter_draws': expected_cases,
            'case_ids': list(range(expected_cases)), 'condition_specific_simulations': 2 * expected_cases,
            'mean_score': float(values.mean()), 'S': result.meta_score,
            'stored_S': result.stored_meta_score, 'R25': result.r25,
            'perfect_joint_pairs': int(perfect.sum()),
            'fraction_of_draws_with_target_at_all_evaluated_times': float(perfect.mean()),
            'pointwise_mean': values.mean(axis=0).tolist(),
            'pointwise_quantiles_q25_q50_q75': quantiles.tolist(),
            'pointwise_target_attainment': (np.abs(values - 1) <= 1e-12).mean(axis=0).tolist(),
            'per_draw_mean_score': values.mean(axis=1).tolist(),
            'interpretation': ('Biological target attainment at a fixed selected MAP.' if operation == 'evaluate-map'
                               else 'Biological target attainment under the stated parameter-and-simulation law.'),
            'uncertainty_calibration_assessed': False}


def calibration_summary(ranks, rank_pit, coverage, case_ids, parameter_keys,
                        posterior_draws, levels, expected_cases, alpha=0.05):
    """KS/Holm and interval coverage for all marginals; no joint-calibration claim."""
    from scipy.stats import kstest
    rank, pit, covered = np.asarray(ranks), np.asarray(rank_pit), np.asarray(coverage)
    levels = _levels(levels)
    keys = list(parameter_keys)
    d = len(keys)
    if (type(expected_cases) is not int or expected_cases <= 0 or d == 0 or len(set(keys)) != d
            or type(posterior_draws) is not int or posterior_draws < 2 or not 0 < alpha < 1):
        raise ValueError('Invalid marginal-calibration dimensions or settings')
    _case_ids(case_ids, expected_cases)
    if rank.shape != (expected_cases, d) or rank.dtype.kind not in 'iu':
        raise ValueError('Expected integer case-by-parameter ranks')
    if np.any(rank < 0) or np.any(rank > posterior_draws):
        raise ValueError('Rank outside [0,L]')
    if (pit.shape != rank.shape or not np.isfinite(pit).all()
            or np.any(pit < rank / (posterior_draws + 1))
            or np.any(pit >= (rank + 1) / (posterior_draws + 1))):
        raise ValueError('Rank fractions do not match randomized ranks')
    if covered.shape != (expected_cases, len(levels), d) or covered.dtype.kind != 'b':
        raise ValueError('Expected boolean case-by-level-by-parameter coverage')
    if np.any(np.diff(covered.astype(int), axis=1) < 0):
        raise ValueError('Nested central intervals have inconsistent coverage')
    ks = [kstest(pit[:, j], 'uniform') for j in range(d)]
    p = np.asarray([value.pvalue for value in ks])
    adjusted = holm_adjust(p)
    band = float(np.sqrt(np.log(2 * d / alpha) / (2 * expected_cases)))
    rows = [{'key': key, 'KS_D': float(ks[j].statistic), 'KS_p': float(p[j]),
             'Holm_adjusted_p': float(adjusted[j]),
             'central_interval_coverage': covered[:, :, j].mean(axis=0).tolist(),
             'central_interval_covered_cases': covered[:, :, j].sum(axis=0).tolist()}
            for j, key in enumerate(keys)]
    return {'operation': 'sbc', 'complete': True, 'completed_cases': expected_cases,
            'case_ids': list(range(expected_cases)), 'posterior_draws_per_case': posterior_draws,
            'total_posterior_draws': expected_cases * posterior_draws,
            'coverage_levels': levels.tolist(), 'alpha': alpha, 'parameters': rows,
            'DKW_familywise_band': band, 'Holm_rejections': int(np.sum(adjusted < alpha)),
            'outside_familywise_DKW': sum(row['KS_D'] > band for row in rows),
            'reference_law': 'Independent uniform box prior in native physical coordinates.',
            'scope': 'Global marginal calibration of one frozen conditional estimator under the original prior.',
            'biological_target_attainment_assessed_by': 'Separate forward predictive summaries.'}


def analytic_controls(seed=20261914):
    """Exact conjugate Gaussian and deliberately biased controls, not model results."""
    from scipy.stats import kstest
    rng = np.random.default_rng(seed)
    sample = np.asarray([[0., -1.], [2., .5], [4., 3.]])
    truth = np.asarray([3., 0.])
    result = ranks_and_intervals(sample, truth, [.8], rng)
    reverse = ranks_and_intervals(sample[::-1], truth, [.8], np.random.default_rng(seed))
    if not np.array_equal(result['ranks'], [2, 1]) or not np.array_equal(result['ranks'], reverse['ranks']):
        raise RuntimeError('Hand-rank or sample-order control failed')
    tied = ranks_and_intervals(np.ones((5, 2)), np.ones(2), [.8], rng)
    if not ((tied['ranks'] >= 0) & (tied['ranks'] <= 5)).all():
        raise RuntimeError('Tie-randomization control failed')
    theta = rng.normal(size=(1000, 2))
    observation = theta + rng.normal(size=(1000, 2))
    exact = observation[:, None, :] / 2 + rng.normal(size=(1000, 256, 2)) / np.sqrt(2)
    rank = (exact < theta[:, None, :]).sum(axis=1)
    pit = (rank + rng.uniform(size=rank.shape)) / 257
    calibrated = [float(kstest(pit[:, j], 'uniform').statistic) for j in range(2)]
    rank = (exact + 3 < theta[:, None, :]).sum(axis=1)
    pit = (rank + rng.uniform(size=rank.shape)) / 257
    biased = [float(kstest(pit[:, j], 'uniform').statistic) for j in range(2)]
    if max(calibrated) >= .08 or min(biased) <= .3:
        raise RuntimeError('Gaussian rank-control separation failed')
    return {'hand_ranks': 'passed', 'order_invariance': 'passed', 'tie_bounds': 'passed',
            'exact_gaussian_KS_D': calibrated, 'deliberately_biased_gaussian_KS_D': biased,
            'meaning': 'Implementation controls only; not calibration results for the cavity network.'}
