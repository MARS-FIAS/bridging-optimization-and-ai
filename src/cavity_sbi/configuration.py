"""Validate portable project configuration without importing scientific packages."""
import json
import math
from pathlib import Path


def project_file(project, relative):
    root = Path(project).resolve()
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def read_json(project, relative):
    return json.loads(project_file(project, relative).read_text())


def validate_model(config):
    parameters = config['parameters']
    keys = [p['key'] for p in parameters]
    if len(keys) != 26 or len(set(keys)) != 26:
        raise ValueError('Expected 26 unique, ordered model parameters')
    for p in parameters:
        lo, hi, value = p['prior_low'], p['prior_high'], p['reference_value']
        if not all(math.isfinite(x) for x in (lo, hi, value)) or not lo < hi or not lo <= value <= hi:
            raise ValueError('Invalid parameter bounds: ' + p['key'])
        if p['coordinate'] != 'linear':
            raise ValueError('The reported estimator uses native linear coordinates')
    if config['parameter_mode'] != 0 or config['faces'] != 6:
        raise ValueError('Unexpected parameter mapping or face convention')
    if config['simulation_horizon_hours'] != 48 or config['event_history_capacity'] <= 0:
        raise ValueError('Invalid horizon or event-history capacity')
    if [c['rule'] for c in config['conditions']] != [0, 1]:
        raise ValueError('The joint observation requires knockout then wild type')
    score = config['score']
    if score['L'] != 1 or score['window_hours'] != [36, 48] or score['beta'] != 1:
        raise ValueError('Unexpected joint-score configuration')
    if score['time_hours'] != [36 + i / 4 for i in range(49)] or score['target'] != [1.0] * 49:
        raise ValueError('Expected the 49-component score grid and ideal target')
    if score['quantile_interpolation'] != 'nearest' or score['meta_score_rounding_digits'] != 4:
        raise ValueError('Unexpected ensemble-meta-score convention')
    if score['aim_NT'] != [[1, 1], [0, 1]] or score['aim_G'] != [[0, 0], [1, 0]]:
        raise ValueError('Unexpected lower/upper biological targets')
    if config['species'] != ['N', 'G', 'NP']:
        raise ValueError('Expected NANOG, GATA6 and phosphorylated NANOG observations')
    if config['observation_time_hours'] != {'start': 0, 'stop_exclusive': 49, 'step': 0.25}:
        raise ValueError('Unexpected recorded observation-grid convention')
    return config


def model_config(project='.'):
    return validate_model(read_json(project, 'config/model.json'))


def geometry_layout(layers):
    if not isinstance(layers, (list, tuple)) or len(layers) < 2:
        raise ValueError('A cavity geometry requires at least two ICM layers')
    if any(type(n) is not int or n <= 0 for n in layers):
        raise ValueError('Layer sizes must be positive integers')
    width = max(layers)
    active = [row * width + (width - size) // 2 + i
              for row, size in enumerate(layers) for i in range(size)]
    rectangular = len(layers) * width
    return {'id': '_'.join(map(str, layers)), 'layers': list(layers),
            'physical_ICM_cells': sum(layers), 'ICM_storage_positions': rectangular,
            'inactive_padding_positions': rectangular - sum(layers),
            'total_storage_positions': rectangular + 1, 'active_ICM_indices': active,
            'shared_reservoir_index': rectangular}


def geometry_configs(project='.'):
    rows = read_json(project, 'config/geometries.json')['geometries']
    if len(rows) != 8 or len({g['id'] for g in rows}) != 8:
        raise ValueError('Expected the eight distinct paper geometries')
    for g in rows:
        if geometry_layout(g['layers']) != g:
            raise ValueError('Inconsistent geometry: ' + g['id'])
    return rows


def get_geometry(identifier, project='.'):
    key = identifier if isinstance(identifier, str) else '_'.join(map(str, identifier))
    for g in geometry_configs(project):
        if g['id'] == key:
            return g
    raise ValueError('Geometry is not in config/geometries.json: ' + key)


def parameter_declaration(config):
    return {p['key']: (p['reference_value'], (p['prior_low'], p['prior_high']))
            for p in validate_model(config)['parameters']}
