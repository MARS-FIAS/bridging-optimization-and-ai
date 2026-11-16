"""Diagnostics controls: synthetic observations/test doubles; no RDME or fitting."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cavity_sbi.cli import main
from cavity_sbi import artifacts, diagnostics, model
from cavity_sbi.diagnostic_reports import summarize_diagnostic
from cavity_sbi.diagnostic_stats import (analytic_controls, calibration_summary,
                                          holm_adjust, predictive_summary, ranks_and_intervals)
from cavity_sbi.workflow import fingerprint
from test_foundation import target_data

ROOT = Path(__file__).resolve().parents[1]


class FakePosterior:
    def __init__(self, theta):
        import torch
        self._map = torch.tensor(theta, dtype=torch.float32)
        self.conditioning = []

    def sample(self, shape, x, **kwargs):
        self.conditioning.append(np.array(x))
        return self._map.repeat(shape[0], 1)


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.profile = self.base / 'diagnostics.json'
        cfg = json.loads((ROOT / 'config/diagnostics-small.json').read_text())
        self.profile.write_text(json.dumps(cfg))
        self.specs = {op: diagnostics.diagnostic_specification(op, ROOT, '5_3_1', self.profile)
                      for op in diagnostics.OPERATIONS}
        self.theta = np.asarray([(p['prior_low'] + p['prior_high']) / 2
                                 for p in self.specs['sbc']['model']['parameters']], dtype=np.float32)
        self.posterior = FakePosterior(self.theta)
        self.forward_calls = []

    def bank(self, spec, entropy):
        return np.repeat(self.theta[None], diagnostics.case_count(spec), axis=0)

    def forward(self, theta, layers, **kwargs):
        self.forward_calls.append((np.array(theta), kwargs['condition_seeds']))
        raw = target_data(self.specs['sbc']['geometry'])
        # Differentiate observations from the ideal conditioning target.
        raw[0] = 0
        audits = tuple({'horizon_reached': True, 'last_event_seconds': 48 * 3600 + 1,
                        'event_history_capacity_reached': False} for _ in [0, 1])
        return SimpleNamespace(flat=raw.reshape(-1), geometry=tuple(layers), event_audit=audits)

    def run_check(self, operation, **kwargs):
        run = self.base / operation
        with patch.object(diagnostics, '_parameter_bank', side_effect=self.bank), \
             patch.object(diagnostics, '_load_estimator', return_value=(self.posterior, 'frozen-weights')), \
             patch.object(artifacts, 'weights_sha256', return_value='frozen-weights'), \
             patch.object(model, 'simulate_pair', side_effect=self.forward):
            result = diagnostics.run_diagnostic(ROOT, run, self.specs[operation], **kwargs)
        return run, result

    def test_paper_previews_have_exact_budgets_and_create_nothing(self):
        expected = {'evaluate-map': (50, 100, 0), 'predictive': (256, 512, 256),
                    'sbc': (1000, 2000, 10000000)}
        for operation, (cases, calls, draws) in expected.items():
            out = io.StringIO()
            run = self.base / operation
            with contextlib.redirect_stdout(out):
                main([operation, '--project', str(ROOT), '--geometry', '5_3_1', '--run-dir', str(run)])
            value = json.loads(out.getvalue())
            self.assertEqual(value['planned_cases'], cases)
            self.assertEqual(value['condition_specific_simulations'], calls)
            self.assertEqual(value['planned_conditional_posterior_draws'], draws)
            self.assertFalse(run.exists())

    def test_invalid_case_range_writes_nothing(self):
        with self.assertRaises(ValueError):
            self.run_check('sbc', start=3, stop=1)
        self.assertFalse((self.base / 'sbc').exists())

    def test_sbc_prior_truths_require_no_neural_artifact_loading(self):
        with patch.object(diagnostics, '_load_estimator', side_effect=AssertionError('Prior needs no NN')):
            theta = diagnostics._parameter_bank(self.specs['sbc'], 20261114)
        self.assertEqual(theta.shape, (4, 26))
        self.assertFalse(np.array_equal(theta[0], theta[1]))

    def test_map_preparation_uses_cached_coordinates_without_sampling_or_optimization(self):
        with patch.object(diagnostics, '_load_estimator', return_value=(self.posterior, 'frozen-weights')), \
             patch.object(artifacts, 'weights_sha256', return_value='frozen-weights'), \
             patch.object(self.posterior, 'sample', side_effect=AssertionError('MAP needs no density draws')):
            theta = diagnostics._parameter_bank(self.specs['evaluate-map'], 123)
        np.testing.assert_array_equal(theta, np.repeat(self.theta[None], 2, axis=0))

    def test_predictive_preparation_conditions_on_the_ideal_target(self):
        with patch.object(diagnostics, '_load_estimator', return_value=(self.posterior, 'frozen-weights')), \
             patch.object(artifacts, 'weights_sha256', return_value='frozen-weights'):
            theta = diagnostics._parameter_bank(self.specs['predictive'], 123)
        self.assertEqual(theta.shape, (4, 26))
        np.testing.assert_array_equal(self.posterior.conditioning[0], np.ones(49))

    def test_diagnostic_pairs_and_completed_observations_are_reused(self):
        run, first = self.run_check('predictive')
        _, second = self.run_check('predictive')
        self.assertEqual(first['condition_specific_simulations'], 8)
        self.assertEqual(second['condition_specific_simulations'], 0)
        self.assertEqual(second['verified_existing_pairs'], 4)
        self.assertEqual(len(self.forward_calls), 4)
        self.assertEqual(self.forward_calls[0][1], [[20261113, 99, 0, 0], [20261113, 99, 0, 1]])
        self.assertTrue((run / 'parameters.npz').exists())

    def test_prepared_workers_read_the_fixed_bank_without_taking_preparation_locks(self):
        run, _ = self.run_check('predictive', prepare_only=True)
        for name in ['.diagnostic.lock', 'parameters.lock']:
            (run / name).write_text('another reader')
        _, result = self.run_check('predictive', start=0, stop=2)
        self.assertEqual(result['new_paired_observations'], 2)
        for name in ['.diagnostic.lock', 'parameters.lock']:
            self.assertEqual((run / name).read_text(), 'another reader')

    def test_sbc_conditions_on_each_simulated_score_and_never_retrains(self):
        run, result = self.run_check('sbc')
        self.assertEqual(result['new_sbc_cases'], 4)
        self.assertEqual(len(self.posterior.conditioning), 4)
        for observation in self.posterior.conditioning:
            self.assertEqual(observation.shape, (49,))
            self.assertTrue(np.all(observation < 1))
        report = summarize_diagnostic(run)
        self.assertEqual(report['completed_cases'], 4)
        self.assertEqual(report['total_posterior_draws'], 128)
        self.assertEqual(report['training_runs'], 0)

    def test_sampler_failure_retries_same_truth_and_reuses_forward_observation(self):
        original = diagnostics._conditional_samples
        with patch.object(diagnostics, '_conditional_samples', side_effect=TimeoutError('synthetic timeout')):
            with self.assertRaises(TimeoutError):
                self.run_check('sbc')
        run = self.base / 'sbc'
        self.assertEqual(len(self.forward_calls), 1)
        self.assertEqual(len(list((run / 'errors').glob('*.json'))), 1)
        with patch.object(diagnostics, '_conditional_samples', side_effect=original):
            _, result = self.run_check('sbc')
        self.assertEqual(result['verified_existing_pairs'], 1)
        self.assertEqual(len(self.forward_calls), 4)
        report = summarize_diagnostic(run)
        self.assertEqual(len(report['failed_attempt_records']), 1)

    def test_complete_sbc_cases_reuse_ranks_without_sampling(self):
        self.run_check('sbc')
        with patch.object(diagnostics, '_conditional_samples', side_effect=AssertionError('No resampling')):
            _, result = self.run_check('sbc')
        self.assertEqual(result['new_sbc_cases'], 0)
        self.assertEqual(result['condition_specific_simulations'], 0)

    def test_saved_draws_resume_after_interrupted_rank_calculation(self):
        from cavity_sbi import diagnostic_stats
        spec = self.specs['sbc']
        spec['settings']['keep_posterior_samples'] = True
        with patch.object(diagnostic_stats, 'ranks_and_intervals', side_effect=RuntimeError('synthetic rank interruption')):
            with self.assertRaises(RuntimeError):
                self.run_check('sbc', stop=1)
        run = self.base / 'sbc'
        self.assertTrue((run / 'posterior/case_000000.npz').exists())
        with patch.object(diagnostics, '_conditional_samples', side_effect=AssertionError('Reuse stored draws')):
            _, result = self.run_check('sbc', stop=1)
        self.assertEqual(result['new_sbc_cases'], 1)
        self.assertEqual(len(self.forward_calls), 1)

    def test_real_frozen_final_loader_and_cached_map_need_no_posterior_draws(self):
        posterior, weight_hash = diagnostics._load_estimator(self.specs['evaluate-map'])
        self.assertEqual(len(weight_hash), 64)
        self.assertTrue(all(not p.requires_grad for p in posterior.posterior_estimator.parameters()))
        with patch.object(posterior, 'sample', side_effect=AssertionError('No posterior sampling')), \
             patch.object(diagnostics, '_load_estimator', return_value=(posterior, weight_hash)):
            bank = diagnostics._parameter_bank(self.specs['evaluate-map'], 123)
        self.assertEqual(bank.shape, (2, 26))
        np.testing.assert_array_equal(bank[0], posterior._map.detach().cpu().numpy())

    def test_summary_never_calls_the_neural_estimator_or_simulator(self):
        run, _ = self.run_check('sbc')
        with patch.object(diagnostics, '_load_estimator', side_effect=AssertionError('No NN')), \
             patch.object(model, 'simulate_pair', side_effect=AssertionError('No simulations')):
            report = summarize_diagnostic(run)
        self.assertEqual(report['new_simulations'], 0)
        self.assertEqual(report['new_posterior_draws'], 0)

    def test_incomplete_cases_cannot_be_reported_as_complete(self):
        run, _ = self.run_check('sbc', stop=2)
        with self.assertRaises(FileNotFoundError):
            summarize_diagnostic(run)
        self.assertFalse((run / 'reports/summary.json').exists())

    def test_incomplete_condition_horizon_is_rejected_and_logged(self):
        original = self.forward

        def incomplete(*args, **kwargs):
            pair = original(*args, **kwargs)
            pair.event_audit[1]['horizon_reached'] = False
            return pair

        with patch.object(self, 'forward', side_effect=incomplete):
            with self.assertRaises(ValueError):
                self.run_check('predictive')
        self.assertFalse((self.base / 'predictive/observations/case_000000.npz').exists())

    def test_corruption_and_cross_operation_outputs_are_rejected(self):
        run, _ = self.run_check('sbc')
        path = run / 'diagnostics/case_000000.npz'
        path.write_bytes(path.read_bytes() + b'corrupt')
        with self.assertRaises(ValueError):
            summarize_diagnostic(run)
        with self.assertRaises(ValueError):
            diagnostics.ensure_diagnostic(ROOT, run, self.specs['predictive'])

    def test_diagnostic_configuration_drift_is_rejected(self):
        run, _ = self.run_check('predictive', prepare_only=True)
        changed = copy.deepcopy(self.specs['predictive'])
        changed['settings']['predictive_pairs'] += 1
        with self.assertRaises(ValueError):
            diagnostics.ensure_diagnostic(ROOT, run, changed)

    def test_separate_condition_streams_preserve_simulator_rule_order(self):
        spec = self.specs['predictive']
        raw = target_data(spec['geometry'])
        rules = []

        def numerical_stub(*args, **kwargs):
            rules.append(kwargs['parameter_set_rules'])
            return raw[kwargs['parameter_set_rules'][0]].reshape(-1)

        with patch.object(model._model, 'simulator', side_effect=numerical_stub):
            pair = model.simulate_pair(self.theta, tuple(spec['geometry']['layers']),
                                       config=spec['model'], condition_seeds=[[12, 0], [12, 1]])
        self.assertEqual(rules, [[0], [1]])
        np.testing.assert_array_equal(pair.flat, raw.reshape(-1))

    def test_hand_ranks_ties_sample_order_and_holm(self):
        draws = np.asarray([[0., 1.], [2., 1.], [4., 1.]])
        first = ranks_and_intervals(draws, [3., 1.], [.5, .9], np.random.default_rng(9))
        second = ranks_and_intervals(draws[::-1], [3., 1.], [.5, .9], np.random.default_rng(9))
        self.assertEqual(first['ranks'][0], 2)
        self.assertTrue(0 <= first['ranks'][1] <= 3)
        np.testing.assert_array_equal(first['rank_pit'], second['rank_pit'])
        np.testing.assert_allclose(holm_adjust([.04, .01, .03]), [.06, .03, .06], rtol=0, atol=1e-15)

    def test_impossible_rank_fraction_and_missing_case_are_rejected(self):
        with self.assertRaises(ValueError):
            calibration_summary(np.zeros((2, 1), dtype=int), np.ones((2, 1)) / 2,
                                np.ones((2, 1, 1), dtype=bool), [0, 1], ['x'], 10, [.9], 2)
        with self.assertRaises(ValueError):
            predictive_summary(np.ones((2, 49)), [0, 2], 2)

    def test_exact_and_deliberately_biased_gaussian_controls(self):
        result = analytic_controls()
        self.assertLess(max(result['exact_gaussian_KS_D']), .08)
        self.assertGreater(min(result['deliberately_biased_gaussian_KS_D']), .3)

    def test_predictive_target_attainment_is_distinct_from_meta_score(self):
        scores = np.full((10, 49), .2)
        scores[-3:] = 1
        result = predictive_summary(scores, np.arange(10), 10)
        self.assertEqual(result['stored_S'], 1)
        self.assertEqual(result['perfect_joint_pairs'], 3)
        self.assertAlmostEqual(result['R25'], .2)
        self.assertFalse(result['uncertainty_calibration_assessed'])


if __name__ == '__main__':
    unittest.main()
