"""Hash checks and explicit CPU loading of the eight selected estimators."""
import hashlib
from importlib import metadata
from .configuration import model_config, geometry_configs, project_file, read_json


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_inventory(project='.'):
    config = model_config(project)
    geometries = geometry_configs(project)
    manifest = read_json(project, 'artifacts/manifest.json')
    rows = manifest['artifacts']
    if manifest['parameter_keys'] != [p['key'] for p in config['parameters']]:
        raise ValueError('Artifact/model parameter order differs')
    if manifest['conditioning_target'] != config['score']['target']:
        raise ValueError('Artifact/model targets differ')
    if len(rows) != 8 or {r['geometry'] for r in rows} != {g['id'] for g in geometries}:
        raise ValueError('Expected one final estimator for every geometry')
    for row in rows:
        path = project_file(project, row['file'])
        if not path.is_file() or path.stat().st_size != row['bytes'] or file_sha256(path) != row['sha256']:
            raise ValueError('Artifact hash/size mismatch: ' + row['geometry'])
    return manifest


def weights_sha256(posterior):
    digest = hashlib.sha256()
    for name, value in posterior.posterior_estimator.state_dict().items():
        arr = value.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def load_final(geometry, project='.'):
    """Load a manifest-checked released model on CPU; never sample or run MAP search."""
    manifest = checked_inventory(project)
    for package, expected in [('sbi', manifest['sbi_version']), ('torch', manifest['torch_version'])]:
        if metadata.version(package).split('+')[0] != expected:
            raise RuntimeError('Use the pinned legacy environment for ' + package)
    row = next((r for r in manifest['artifacts'] if r['geometry'] == geometry), None)
    if row is None:
        raise ValueError('Unknown final-estimator geometry: ' + str(geometry))
    import pickle
    import numpy as np
    import torch
    path = project_file(project, row['file'])
    with path.open('rb') as stream:
        posterior = pickle.load(stream)
    if type(posterior).__name__ != 'DirectPosterior':
        raise ValueError('Expected a legacy sbi DirectPosterior')
    config = model_config(project)
    if list(posterior.parameter_set_true) != manifest['parameter_keys']:
        raise ValueError('Loaded estimator parameter order differs')
    prior = getattr(posterior.prior, 'base_dist', posterior.prior)
    for attr, key in [('low', 'prior_low'), ('high', 'prior_high')]:
        if not np.array_equal(getattr(prior, attr).detach().cpu().numpy(), [p[key] for p in config['parameters']]):
            raise ValueError('Loaded estimator prior bounds differ')
    posterior.posterior_estimator.to('cpu').eval()
    for value in posterior.posterior_estimator.parameters():
        value.requires_grad_(False)
    cached_map = getattr(posterior, '_map', None)
    if cached_map is None or tuple(cached_map.shape) != (26,) or not torch.isfinite(cached_map).all():
        raise ValueError('Expected the saved selected MAP coordinates')
    if 'weights_sha256' in row and weights_sha256(posterior) != row['weights_sha256']:
        raise ValueError('Loaded estimator weight hash differs')
    return posterior
