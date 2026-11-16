"""Explicit conditional-correlation recipe from the retained sbi analysis service."""
from pathlib import Path
from .artifacts import checked_inventory, file_sha256
from . import __version__
from .configuration import get_geometry, model_config, project_file
from .reproduction import figure_config, write_csv
from .workflow import atomic_json, validate_run_location


def correlation_plan(project, geometry, output, parameters=None, resolution=250, condition='source-rounded-map'):
    cfg = model_config(project)
    keys = [p['key'] for p in cfg['parameters']]
    requested = figure_config(project)['correlation_display_order'] if parameters is None else list(parameters)
    if len(requested) < 2 or len(set(requested)) != len(requested) or set(requested)-set(keys):
        raise ValueError('Choose at least two distinct model parameter keys')
    if type(resolution) is not int or resolution < 3:
        raise ValueError('Correlation grids require at least three points per coordinate')
    if condition not in ['source-rounded-map', 'saved-map']:
        raise ValueError('Unknown conditional MAP convention')
    get_geometry(geometry, project)
    inventory = checked_inventory(project)
    row = next(r for r in inventory['artifacts'] if r['geometry'] == geometry)
    pairs = len(requested)*(len(requested)-1)//2
    return {'operation': 'conditional-correlations', 'mode': 'preview', 'geometry': geometry,
            'parameters': requested, 'resolution': resolution, 'condition': condition,
            'output_directory': str(validate_run_location(project, output)),
            'estimator_sha256': row['sha256'], 'pairwise_grids': pairs,
            'grid_density_evaluations': pairs*resolution**2,
            'method': 'sbi 0.19.2 analysis.conditional_corrcoeff over the declared prior bounds',
            'conditioning_target': 'Ideal 49-component all-ones score',
            'new_simulations': 0, 'new_posterior_draws': 0, 'training_runs': 0,
            'execute_required': True, 'files_written': False,
            'article_matrix_comparison': 'Original numeric matrix is a separate input; this recipe does not assert numerical identity.'}


def conditional_correlations(posterior, model, parameters, resolution=250, condition='source-rounded-map'):
    """Grid Pearson coefficients; hold other native coordinates at the selected MAP.

    The source analysis rounds MAP coordinates above 1 to integers and others
    to two decimals. saved-map explicitly selects unrounded cached coordinates.
    An unnormalized log-density avoids normalization sampling; its constant
    cancels in the grid correlation calculation.
    """
    import numpy as np
    import torch
    from sbi import analysis
    keys = [p['key'] for p in model['parameters']]
    if len(parameters) < 2 or len(set(parameters)) != len(parameters) or set(parameters)-set(keys):
        raise ValueError('Invalid conditional-correlation parameter subset')
    if type(resolution) is not int or resolution < 3 or condition not in ['source-rounded-map', 'saved-map']:
        raise ValueError('Invalid grid or MAP condition')
    cached = posterior._map.detach().cpu().numpy().copy()
    if condition == 'source-rounded-map':
        cached = np.asarray([int(round(float(v))) if v > 1 else round(float(v), 2) for v in cached], dtype=np.float32)
    bounds = np.asarray([[p['prior_low'], p['prior_high']] for p in model['parameters']])
    if cached.shape != (len(keys),) or not np.isfinite(cached).all() or np.any(cached < bounds[:, 0]) or np.any(cached > bounds[:, 1]):
        raise ValueError('Invalid MAP conditioning coordinates')
    native_indices = sorted(keys.index(k) for k in parameters)
    permutation = [native_indices.index(keys.index(k)) for k in parameters]

    class DensityView:
        _device = 'cpu'

        def log_prob(self, theta):
            return posterior.log_prob(theta, x=torch.ones(49), norm_posterior=False)

    with torch.no_grad():
        matrix = analysis.conditional_corrcoeff(density=DensityView(),
            limits=torch.as_tensor(bounds, dtype=torch.float32),
            condition=torch.as_tensor(cached, dtype=torch.float32).reshape(1, -1),
            subset=native_indices, resolution=resolution).detach().cpu().numpy()
    matrix = matrix[np.ix_(permutation, permutation)]
    if (not np.isfinite(matrix).all() or np.any(np.abs(matrix) > 1+1e-5)
            or not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 1)):
        raise ValueError('Invalid conditional-correlation matrix')
    return matrix, cached


def run_correlations(project, geometry, output, parameters=None, resolution=250, condition='source-rounded-map'):
    from .artifacts import load_final, weights_sha256
    from .cli import environment_report
    import torch
    result = correlation_plan(project, geometry, output, parameters, resolution, condition)
    environment = environment_report(check_runtime=True)
    if environment['reference_version_differences']:
        raise RuntimeError('Use the pinned CPU environment')
    output = Path(result['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    posterior = load_final(geometry, project)
    before = weights_sha256(posterior)
    matrix, fixed = conditional_correlations(posterior, model_config(project), result['parameters'], resolution, condition)
    if weights_sha256(posterior) != before:
        raise ValueError('Frozen weights changed')
    keys = result['parameters']
    rows = [dict(parameter=key, **{other: float(matrix[i, j]) for j, other in enumerate(keys)}) for i, key in enumerate(keys)]
    path = write_csv(output / 'conditional_correlations.csv', rows)
    result.update(mode='executed', files_written=True, weights_sha256=before,
                  package_version=__version__, code_sha256=file_sha256(Path(__file__)),
                  model_configuration_sha256=file_sha256(project_file(project, 'config/model.json')),
                  figure_configuration_sha256=file_sha256(project_file(project, 'config/figures.json')),
                  fixed_native_coordinates=fixed.tolist(), parameter_order=[p['key'] for p in model_config(project)['parameters']],
                  normalization='Raw conditional log density; a multiplicative normalization constant cancels.',
                  interpretation='Conditional linear dependence at fixed MAP coordinates; not global identifiability or absence of compensation.',
                  environment=environment, files={path.name: file_sha256(path)})
    atomic_json(output / 'correlation_manifest.json', result)
    return result
