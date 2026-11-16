"""Workflow controls with synthetic observations and test-double estimators only."""
import contextlib
import copy
import io
import json
import shutil
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cavity_sbi.cli import main
from cavity_sbi.configuration import model_config, get_geometry
from cavity_sbi import datasets, training
from cavity_sbi.selection import best_candidate, select_round
from cavity_sbi.workflow import (atomic_json, checked_run, fingerprint, plan,
                                 round_path, specification, load_run_posterior)
from cavity_sbi.artifacts import file_sha256
from test_foundation import target_data

ROOT = Path(__file__).resolve().parents[1]


class FakePosterior:
    def __init__(self, theta):
        import torch
        self._map = torch.tensor(theta, dtype=torch.float32)

    def set_default_x(self, target):
        self.default_x = target

    def sample(self, shape, **kwargs):
        return self._map.repeat(shape[0], 1)

    def map(self, **kwargs):
        return self._map


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.run = self.base / 'run'
        cfg = json.loads((ROOT / 'config/workflow-small.json').read_text())
        cfg.update(simulation_pairs_per_round=3, tasks=2, evaluation_pairs=3,
                   posterior_summary_draws=8, map_num_init_samples=8)
        self.profile = self.base / 'workflow.json'
        self.profile.write_text(json.dumps(cfg))
        self.spec = specification(ROOT, '1_1', self.profile)
        self.theta = np.asarray([(p['prior_low'] + p['prior_high']) / 2
                                 for p in self.spec['model']['parameters']], dtype=np.float32)
        self.forward_calls = []

    def sample_bank(self, spec, run, round_index, entropy):
        return np.repeat(self.theta[None], spec['workflow']['simulation_pairs_per_round'], axis=0)

    def forward(self, theta, layers, **kwargs):
        self.forward_calls.append(np.asarray(theta).copy())
        geometry = get_geometry(layers, ROOT)
        return SimpleNamespace(flat=target_data(geometry).reshape(-1), geometry=tuple(layers))

    def generate(self, index):
        with patch.object(datasets, '_sample_parameters', side_effect=self.sample_bank), \
             patch.object(datasets, 'simulate_pair', side_effect=self.forward):
            return datasets.generate(ROOT, self.run, self.spec, index)

    def fake_fit(self, theta, scores, model, workflow, seed, log_directory=None):
        return FakePosterior(self.theta), self.theta.copy(), self.theta.copy()

    def train(self, index, task):
        with patch.object(training, 'fit_estimator', side_effect=self.fake_fit) as fit, \
             patch.object(training, 'simulate_pair', side_effect=self.forward):
            result = training.train_task(self.run, self.spec, index, task)
            return result, fit.call_count

    def complete_round(self, index):
        self.generate(index)
        for task in range(2):
            self.train(index, task)
        return select_round(self.run, self.spec, index)

    def test_preview_has_paper_budgets_and_writes_nothing(self):
        result = plan('train', ROOT, '5_3_1', 2, self.run)
        self.assertEqual(result['fitting_data_rounds'], [1, 2])
        self.assertEqual(result['training_pairs_when_complete'], 200000)
        self.assertEqual(result['evaluation_condition_specific_calls_per_task'], 200)
        self.assertFalse(self.run.exists())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(['generate', '--project', str(ROOT), '--geometry', '1_1', '--round', '0',
                  '--run-dir', str(self.run), '--dry-run'])
        self.assertEqual(json.loads(output.getvalue())['mode'], 'preview')
        self.assertFalse(self.run.exists())

    def test_generation_restart_reuses_same_cases_without_resimulation(self):
        first = self.generate(0)
        self.assertEqual(first['new_pairs'], 3)
        self.assertEqual(len(self.forward_calls), 3)
        second = self.generate(0)
        self.assertEqual(second['verified_existing_pairs'], 3)
        self.assertEqual(len(self.forward_calls), 3)

    def test_initial_prior_sampling_needs_no_neural_artifact(self):
        with patch.object(datasets, 'load_run_posterior', side_effect=AssertionError('No neural model needed')):
            theta = datasets._sample_parameters(self.spec, self.run, 0, 123)
        self.assertEqual(theta.shape, (3, 26))

    def test_run_posterior_loader_accepts_the_released_native_format(self):
        path = ROOT / 'artifacts/final/1_1.pkl'
        posterior = load_run_posterior(path, file_sha256(path), self.spec['model'])
        self.assertEqual(tuple(posterior._map.shape), (26,))

    def test_case_corruption_is_rejected(self):
        self.generate(0)
        path = round_path(self.run, 0) / 'cases/case_000000.npz'
        path.write_bytes(path.read_bytes() + b'corruption')
        with self.assertRaises(ValueError):
            self.generate(0)

    def test_case_from_another_round_cannot_be_reused_even_at_identical_theta(self):
        self.complete_round(0)
        self.generate(1)
        source = round_path(self.run, 0) / 'cases/case_000000'
        target = round_path(self.run, 1) / 'cases/case_000000'
        for suffix in ['.npz', '.json']:
            shutil.copyfile(source.with_suffix(suffix), target.with_suffix(suffix))
        with self.assertRaises(ValueError):
            datasets.load_training_data(self.run, self.spec, 1)

    def test_changed_model_or_profile_cannot_mix_into_existing_run(self):
        self.generate(0)
        changed = copy.deepcopy(self.spec)
        changed['workflow']['evaluation_pairs'] += 1
        with self.assertRaises(ValueError):
            checked_run(self.run, changed)

    def test_next_round_requires_previous_selection(self):
        self.generate(0)
        with self.assertRaises(FileNotFoundError):
            self.generate(1)

    def test_pool_uses_only_current_and_immediately_previous_round(self):
        self.complete_round(0)
        self.complete_round(1)
        self.generate(2)
        theta, scores, rounds = datasets.load_training_data(self.run, self.spec, 2)
        self.assertEqual(rounds, [1, 2])
        self.assertEqual(theta.shape, (6, 26))
        self.assertEqual(scores.shape, (6, 49))
        self.assertEqual(scores.dtype, np.float32)

    def test_two_fits_ties_and_task_restart(self):
        self.generate(0)
        first, calls = self.train(0, 0)
        self.assertEqual(calls, 2)
        self.assertEqual(first['selected_fit'], 0)
        for fit in first['fits']:
            self.assertEqual(fit['evaluation_pairs'], 3)
            self.assertEqual(fit['stored_meta_score'], 1.0)
        _, calls = self.train(0, 0)
        self.assertEqual(calls, 0)
        self.train(0, 1)
        result = select_round(self.run, self.spec, 0)
        self.assertEqual(result['selected_task'], 0)
        self.assertEqual(result['selected_fit'], 0)
        self.assertEqual(result['posterior_sha256'], first['fits'][0]['posterior_sha256'])

    def test_interrupted_evaluation_reuses_trained_checkpoint(self):
        self.generate(0)
        calls = [0]

        def interrupted(theta, layers, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError('Synthetic interruption')
            return self.forward(theta, layers, **kwargs)

        with patch.object(training, 'fit_estimator', side_effect=self.fake_fit) as fit, \
             patch.object(training, 'simulate_pair', side_effect=interrupted):
            with self.assertRaises(RuntimeError):
                training.train_task(self.run, self.spec, 0, 0)
            self.assertEqual(fit.call_count, 1)
        with patch.object(training, 'load_run_posterior', return_value=FakePosterior(self.theta)), \
             patch.object(training, 'fit_estimator', side_effect=self.fake_fit) as fit, \
             patch.object(training, 'simulate_pair', side_effect=self.forward):
            training.train_task(self.run, self.spec, 0, 0)
            self.assertEqual(fit.call_count, 1)  # Only the second, previously unstarted fit.

    def test_round_selection_requires_every_task(self):
        self.generate(0)
        self.train(0, 0)
        with self.assertRaises(FileNotFoundError):
            select_round(self.run, self.spec, 0)

    def test_selection_compares_rounded_S_and_retains_first_tie(self):
        rows = [{'stored_meta_score': 0.9, 'raw_meta_score': 0.90001},
                {'stored_meta_score': 0.9, 'raw_meta_score': 0.90004}]
        self.assertEqual(best_candidate(rows), 0)

    def test_fitting_passes_None_and_a_fresh_conditional_estimator(self):
        captured = {}
        posterior = FakePosterior(self.theta)

        class FakeInference:
            def append_simulations(self, theta, scores, proposal):
                captured.update(proposal=proposal, theta_shape=tuple(theta.shape), score_shape=tuple(scores.shape))
                return self

            def train(self, **kwargs):
                captured['train_kwargs'] = kwargs
                return 'test-double network'

            def build_posterior(self, network):
                captured['network'] = network
                return posterior

        with patch.object(training, '_make_inference', return_value=FakeInference()) as factory:
            result, mape, _ = training.fit_estimator(np.repeat(self.theta[None], 3, 0),
                np.ones((3, 49)), self.spec['model'], self.spec['workflow'], 123)
        self.assertEqual(factory.call_count, 1)
        self.assertIsNone(captured['proposal'])
        self.assertEqual(captured['theta_shape'], (3, 26))
        self.assertEqual(captured['score_shape'], (3, 49))
        np.testing.assert_array_equal(mape, self.theta)

    def test_real_inference_logger_is_scoped_without_fitting_a_network(self):
        import torch
        from sbi.utils import BoxUniform
        from torch.utils.tensorboard import SummaryWriter
        log_dir = self.base / 'run/round_000/task_00/fit_00/tensorboard'
        prior = BoxUniform(low=torch.zeros(26), high=torch.ones(26))
        with SummaryWriter(log_dir=str(log_dir)) as writer:
            inference = training._make_inference(prior, summary_writer=writer)
            self.assertIs(inference._summary_writer, writer)
            self.assertEqual(Path(writer.log_dir).resolve(), log_dir.resolve())
        self.assertTrue(log_dir.exists())


if __name__ == '__main__':
    unittest.main()
