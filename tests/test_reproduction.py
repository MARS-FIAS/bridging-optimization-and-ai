"""Compact-evidence arithmetic and plot interfaces; no real model simulation."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cavity_sbi.cli import main
from cavity_sbi.configuration import geometry_configs, model_config
from cavity_sbi.correlations import conditional_correlations, correlation_plan
from cavity_sbi.diagnostic_stats import calibration_summary
from cavity_sbi.evidence import evidence_arrays, evidence_inventory, evidence_rows
from cavity_sbi.reproduction import figure_config, plot_saved_diagnostic, reproduction_plan
from cavity_sbi.result_analysis import (fate_counts_from_raw, parameter_analysis,
                                        predictive_analysis, selection_analysis)
from test_foundation import target_data

ROOT = Path(__file__).resolve().parents[1]


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.model = model_config(ROOT)
        self.geometries = geometry_configs(ROOT)

    def test_previews_verify_evidence_and_plan_work_without_outputs(self):
        plan = reproduction_plan(ROOT, self.base / 'plots')
        self.assertEqual(plan['evidence_files_verified'], 8)
        self.assertEqual(plan['evidence_bytes'], 345111)
        self.assertFalse((self.base / 'plots').exists())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(['reproduce', '--project', str(ROOT), '--output-dir', str(self.base / 'plots')])
        self.assertEqual(json.loads(out.getvalue())['mode'], 'preview')
        self.assertFalse((self.base / 'plots').exists())

    def test_evidence_tampering_is_rejected(self):
        project = self.base / 'source'
        shutil.copytree(ROOT / 'config', project / 'config')
        shutil.copytree(ROOT / 'evidence', project / 'evidence')
        path = project / 'evidence/goal_predictive.npz'
        path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaises(ValueError):
            evidence_inventory(project)

    def test_recomputed_selection_retains_every_candidate_tie_and_final_round(self):
        data = evidence_arrays(ROOT, 'selection_candidates.npz')
        summary, groups, candidates = selection_analysis(data, self.geometries, self.model)
        self.assertEqual(len(candidates), 400)
        self.assertEqual(len(groups), 40)
        self.assertEqual([summary[k] for k in ['same_first_task', 'tie_only', 'strict_difference']], [34, 2, 4])
        self.assertEqual(summary['final_round_agreement'], 8)
        self.assertEqual(sum(r['S'] == 1 for r in groups if r['round_number'] == 5), 7)
        altered = copy.deepcopy(data)
        altered['selected'][0] = not altered['selected'][0]
        with self.assertRaises(ValueError):
            selection_analysis(altered, self.geometries, self.model)

    def test_all_predictive_scores_reconstruct_from_integer_cell_fate_counts(self):
        data = evidence_arrays(ROOT, 'goal_predictive.npz')
        summary, time, histogram, intended = predictive_analysis(data, self.model, 256)
        self.assertEqual(summary['independent_scores_from_fate_counts_verified'], 256)
        self.assertEqual(summary['perfect_joint_pairs'], 107)
        self.assertEqual(len(time), 49)
        self.assertEqual(sum(r['count'] for r in histogram), 256)
        self.assertEqual(len(intended), 4)
        altered = copy.deepcopy(data)
        altered['fate_counts'][0] = 0
        with self.assertRaises(ValueError):
            predictive_analysis(altered, self.model, 256)

    def test_fate_classifier_uses_phosphorylated_nanog_and_physical_cells(self):
        for geometry in self.geometries:
            raw = target_data(geometry, phosphorylated=True)
            counts = fate_counts_from_raw(raw, geometry['layers'], self.model)
            self.assertEqual(counts.shape, (2, 2, 49, 2))
            np.testing.assert_array_equal(counts[0, 0, :, 0], [geometry['layers'][0]]*49)
            np.testing.assert_array_equal(counts[1, 0, :, 1], [geometry['layers'][0]]*49)
            np.testing.assert_array_equal(counts[1, 1, :, 0], [sum(geometry['layers'][1:])]*49)

    def test_existing_prior_scores_supply_the_baseline_without_new_simulations(self):
        summary, _, _, _ = predictive_analysis(evidence_arrays(ROOT, 'prior_predictive.npz'), self.model, 1000, 'prior-predictive')
        self.assertAlmostEqual(summary['mean_score'], .05352619069799055, places=15)
        self.assertEqual(summary['perfect_joint_pairs'], 0)

    def test_all_sbc_parameters_and_intervals_reconcile(self):
        data = evidence_arrays(ROOT, 'marginal_sbc.npz')
        summary = calibration_summary(data['ranks'], data['rank_pit'], data['coverage'], data['case_ids'],
            data['parameter_keys'], int(data['posterior_draws_per_case']), data['coverage_levels'], 1000)
        self.assertEqual(len(summary['parameters']), 26)
        self.assertEqual(summary['Holm_rejections'], 26)
        cover = [r['central_interval_coverage'][2] for r in summary['parameters']]
        self.assertEqual([min(cover), max(cover)], [.082, .819])

    def test_parameter_boxes_and_dots_retain_native_coordinates_and_normalization(self):
        summary = parameter_analysis(evidence_rows(ROOT, 'selected_MAP_parameters.csv'),
            evidence_rows(ROOT, 'reference_density_quantiles_256.csv'), self.model, self.geometries)
        self.assertEqual(summary['selected_MAP_coordinates'], 208)
        self.assertEqual(summary['reference_marginals'], 26)
        self.assertAlmostEqual(summary['FL_IQR_prior_percent'][0], 55.605564453125, places=10)
        self.assertEqual(sum(len(g['parameters']) for g in figure_config(ROOT)['parameter_groups']), 26)

    def test_correlation_preview_reports_the_full_cost_without_execution(self):
        plan = correlation_plan(ROOT, '5_3_1', self.base / 'matrix')
        self.assertEqual(plan['grid_density_evaluations'], 20312500)
        self.assertEqual(plan['new_posterior_draws'], 0)
        self.assertFalse((self.base / 'matrix').exists())

    def test_conditional_grid_matches_a_known_gaussian_without_sampling(self):
        import torch
        low = np.asarray([p['prior_low'] for p in self.model['parameters']])
        high = np.asarray([p['prior_high'] for p in self.model['parameters']])

        class GaussianDensity:
            _map = torch.tensor((low+high)/2, dtype=torch.float32)
            normalized = []

            def log_prob(self, theta, x, norm_posterior):
                self.normalized.append(norm_posterior)
                np.testing.assert_array_equal(x.numpy(), np.ones(49))
                z = (theta[:, :2]-500)/100
                return -(z[:, 0]**2-1.4*z[:, 0]*z[:, 1]+z[:, 1]**2)/(2*(1-.7**2))

            def sample(self, *args, **kwargs):
                raise AssertionError('Grid correlation must not draw samples')

        density = GaussianDensity()
        matrix, condition = conditional_correlations(density, self.model, ['G_G', 'N_N'], resolution=60)
        self.assertAlmostEqual(matrix[0, 1], .7, places=3)
        np.testing.assert_array_equal(np.diag(matrix), [1, 1])
        self.assertTrue(density.normalized and not any(density.normalized))
        self.assertEqual(condition.shape, (26,))

    def test_new_run_plotting_reuses_completed_observations_without_neural_work(self):
        import test_diagnostics as fixture_module
        from cavity_sbi import diagnostics, model
        fixture = fixture_module.DiagnosticTests('test_diagnostic_pairs_and_completed_observations_are_reused')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.specs['predictive']['settings']['keep_raw_trajectories'] = True
        run, _ = fixture.run_check('predictive')
        with patch.object(diagnostics, '_load_estimator', side_effect=AssertionError('No neural loading')), \
             patch.object(model, 'simulate_pair', side_effect=AssertionError('No new simulations')):
            result = plot_saved_diagnostic(ROOT, run, self.base / 'own-run-plots')
        self.assertEqual(result['new_simulations'], 0)
        self.assertEqual(result['new_posterior_draws'], 0)
        self.assertTrue((self.base / 'own-run-plots/diagnostic_predictive.pdf').exists())

    def test_existing_plot_directory_fails_before_refreshing_input_reports(self):
        import test_diagnostics as fixture_module
        from cavity_sbi import diagnostic_reports
        fixture = fixture_module.DiagnosticTests('test_diagnostic_pairs_and_completed_observations_are_reused')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        run, _ = fixture.run_check('predictive', prepare_only=True)
        output = self.base / 'existing'
        output.mkdir()
        with patch.object(diagnostic_reports, 'summarize_diagnostic', side_effect=AssertionError('No report refresh')):
            with self.assertRaises(FileExistsError):
                plot_saved_diagnostic(ROOT, run, output)


if __name__ == '__main__':
    unittest.main()
