"""Validate compact evidence bytes without importing scientific libraries."""
from .artifacts import file_sha256
from .configuration import geometry_configs, model_config, project_file, read_json


def evidence_inventory(project='.'):
    manifest = read_json(project, 'evidence/manifest.json')
    config = model_config(project)
    if manifest['schema_version'] != 1 or manifest['parameter_keys'] != [p['key'] for p in config['parameters']]:
        raise ValueError('Evidence parameter declaration differs')
    if manifest['geometry_order'] != [g['id'] for g in geometry_configs(project)]:
        raise ValueError('Evidence geometry declaration differs')
    required = {'selection_candidates.npz', 'goal_predictive.npz', 'marginal_sbc.npz', 'prior_predictive.npz',
                'selected_MAP_parameters.csv', 'reference_density_quantiles_256.csv',
                'selected_final_artifacts.csv', 'expected_results.json'}
    if set(manifest['files']) != required:
        raise ValueError('Unexpected compact evidence inventory')
    for name, item in manifest['files'].items():
        path = project_file(project, 'evidence/' + name)
        if path.stat().st_size != item['bytes'] or file_sha256(path) != item['sha256']:
            raise ValueError('Compact evidence checksum/size differs: ' + name)
    return manifest


def evidence_arrays(project, name):
    import numpy as np
    with np.load(project_file(project, 'evidence/' + name), allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


def evidence_rows(project, name):
    import csv
    with project_file(project, 'evidence/' + name).open() as stream:
        return list(csv.DictReader(stream))
