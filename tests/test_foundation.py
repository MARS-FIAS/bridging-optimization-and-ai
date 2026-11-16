"""Scientific interface checks using synthetic observations; no RDME runs or fitting."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from cavity_sbi.configuration import model_config, geometry_configs, parameter_declaration
from cavity_sbi.scoring import joint_pattern_scores, ensemble_scores, scores_from_quantiles
from cavity_sbi import __version__

ROOT = Path(__file__).resolve().parents[1]


def target_data(geometry, phosphorylated=False, only_window=False):
    """Exact biological targets in the original molecular-observation layout."""
    data = np.zeros((2, 3, 196, geometry['total_storage_positions']), dtype=float)
    frames = slice(144, 193) if only_window else slice(None)
    active = geometry['active_ICM_indices']
    lower = active[:geometry['layers'][0]]
    upper = active[geometry['layers'][0]:]
    nanog_channel = 2 if phosphorylated else 0
    for cell in active:
        data[0, nanog_channel, frames, cell] = 1000
    for cell in lower:
        data[1, 1, frames, cell] = 2000
    for cell in upper:
        data[1, nanog_channel, frames, cell] = 1000
    return data


class FoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = model_config(ROOT)
        cls.geometries = geometry_configs(ROOT)

    def test_both_biological_targets_in_all_eight_geometries(self):
        for geometry in self.geometries:
            score = joint_pattern_scores(target_data(geometry).reshape(-1), geometry['layers'], config=self.config)
            np.testing.assert_array_equal(score, np.ones((1, 49)))

    def test_violation_of_either_condition_reduces_joint_target_score(self):
        geometry = self.geometries[5]
        for condition in (0, 1):
            data = target_data(geometry)
            data[condition] = 0  # Undifferentiated throughout one condition.
            score = joint_pattern_scores(data.reshape(-1), geometry['layers'], config=self.config)
            np.testing.assert_allclose(score, np.full((1, 49), 1 / (np.exp(1) + 1)), rtol=0, atol=1e-15)

    def test_opposing_fates_score_zero(self):
        geometry = self.geometries[5]
        data = target_data(geometry)
        data[0] = 0
        for cell in geometry['active_ICM_indices']:
            data[0, 1, :, cell] = 2000  # PRE throughout the knockout condition.
        score = joint_pattern_scores(data.reshape(-1), geometry['layers'], config=self.config)
        np.testing.assert_array_equal(score, np.zeros((1, 49)))

    def test_total_nanog_includes_phosphorylated_nanog(self):
        geometry = self.geometries[5]
        score = joint_pattern_scores(target_data(geometry, phosphorylated=True).reshape(-1), geometry['layers'], config=self.config)
        np.testing.assert_array_equal(score, np.ones((1, 49)))

    def test_only_36_to_48_hours_enters_the_joint_score(self):
        geometry = self.geometries[5]
        score = joint_pattern_scores(target_data(geometry, only_window=True).reshape(-1), geometry['layers'], config=self.config)
        np.testing.assert_array_equal(score, np.ones((1, 49)))

    def test_ensemble_meta_score_is_distinct_from_lower_quartile(self):
        q = np.tile([0.2, 0.2, 1.0], (49, 1))
        result = scores_from_quantiles(q)
        self.assertEqual(result.stored_meta_score, 1.0)
        self.assertAlmostEqual(result.r25, 0.2)
        values = np.asarray([[0.0] * 49, [0.2] * 49, [0.3] * 49, [1.0] * 49])
        from cavity_sbi._numerics.objective import ObjectiveFunPanelRule
        objective = ObjectiveFunPanelRule(np.array(None), ['N', 'G', 'NP'])
        objective.data_objective = values[:, None, :, None]
        original_quantiles, original_s = objective.appraise()
        result = ensemble_scores(values)
        self.assertEqual(result.stored_meta_score, original_s)
        self.assertEqual(result, scores_from_quantiles(original_quantiles))

    def test_model_assembly_preserves_reported_dimensions(self):
        import torch
        from cavity_sbi.model import construct_stem
        from cavity_sbi._numerics.utils import make_para_fun
        declaration = parameter_declaration(self.config)
        theta = torch.as_tensor([p['reference_value'] for p in self.config['parameters']], dtype=torch.float32)
        stem = construct_stem(theta, declaration, make_para_fun(theta, declaration, 0))
        self.assertEqual(len(stem.initial_state), 67)
        self.assertEqual(len(stem.reactions), 121)
        self.assertTrue(stem._assembled)

    def test_numerical_definition_hashes_match_extracted_source(self):
        manifest = json.loads((ROOT / 'provenance/CORE_EXTRACTION.json').read_text())
        for entry in manifest['records']:
            tree = ast.parse((ROOT / entry['destination']).read_text())
            actual = {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
                      for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            self.assertEqual(actual, entry['definition_AST_hashes'])

    def test_import_starts_no_scientific_work_or_plotting(self):
        code = '''import sys
import cavity_sbi
from cavity_sbi import model, scoring
assert not any(n in sys.modules for n in ['torch', 'sbi', 'matplotlib.pyplot'])
print(cavity_sbi.__version__)
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), __version__)

    def test_incomplete_event_history_is_rejected(self):
        from cavity_sbi import model

        class SimulatedHistory:
            def __init__(self, last):
                self.epoch_mat = np.asarray([[0.0], [last]])
                self.step_cells = np.asarray([2])
                self.cells, self.steps = 1, 10

            def meth_direct(self, *args):
                return None

        with patch.object(model._model, 'BiochemSimulSwift', SimulatedHistory):
            with model._audit_horizon([]):
                with self.assertRaises(RuntimeError):
                    model._model.BiochemSimulSwift(48 * 3600 - 1).meth_direct()
                model._model.BiochemSimulSwift(48 * 3600 + 1).meth_direct()

    def test_optional_seed_keeps_independent_streams_and_keyword_API(self):
        from cavity_sbi.model import independent_streams
        original = np.random.default_rng
        results = []
        for _ in range(2):
            with independent_streams(123):
                first = np.random.default_rng(seed=None).normal(size=8)
                second = np.random.default_rng(seed=None).normal(size=8)
                explicit = np.random.default_rng(seed=456).normal(size=8)
                results.append((first, second))
            np.testing.assert_array_equal(explicit, original(seed=456).normal(size=8))
        np.testing.assert_array_equal(results[0], results[1])
        self.assertFalse(np.array_equal(*results[0]))
        self.assertIs(np.random.default_rng, original)


if __name__ == '__main__':
    unittest.main()
