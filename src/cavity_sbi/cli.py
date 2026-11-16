"""Reproduce the cavity workflow; computation requires an explicit --execute."""
import argparse
import importlib
import json
import platform
from importlib import metadata
from . import __version__
from .artifacts import checked_inventory, weights_sha256
from .configuration import model_config, geometry_configs

REFERENCE = {'numpy': '1.24.3', 'scipy': '1.10.1', 'numba': '0.58.1',
             'torch': '1.11.0', 'sbi': '0.19.2', 'nflows': '0.14',
             'pyknos': '0.15.1', 'pyro-ppl': '1.8.1', 'matplotlib': '3.6.2'}


def environment_report(check_runtime=False):
    actual = {}
    for package in REFERENCE:
        try:
            actual[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual[package] = None
    metadata_versions = actual.copy()
    if check_runtime:
        for package in REFERENCE:
            module = importlib.import_module({'pyro-ppl': 'pyro'}.get(package, package))
            actual[package] = getattr(module, '__version__', actual[package])
    differences = [p for p, v in REFERENCE.items() if actual[p] is None or actual[p].split('+')[0] != v]
    if platform.python_version_tuple()[:2] != ('3', '8'):
        differences.append('python')
    return {'python': platform.python_version(), 'packages': actual,
            'metadata_packages': metadata_versions, 'runtime_versions_checked': check_runtime,
            'reference_versions': REFERENCE, 'reference_version_differences': differences}


def doctor(project='.', load_models=False, strict_versions=False):
    config = model_config(project)
    geometries = geometry_configs(project)
    artifacts = checked_inventory(project)
    environment = environment_report(check_runtime=load_models)
    if strict_versions and environment['reference_version_differences']:
        raise RuntimeError('Environment differs from the reference: ' + ', '.join(environment['reference_version_differences']))
    result = {'package': 'cavity_sbi', 'version': __version__,
              'parameters': len(config['parameters']), 'geometries': geometries,
              'joint_score_times': len(config['score']['time_hours']),
              'simulation_horizon_hours': config['simulation_horizon_hours'],
              'artifact_hashes_verified': len(artifacts['artifacts']),
              'artifact_total_bytes': sum(r['bytes'] for r in artifacts['artifacts']),
              'models_loaded': [], 'environment': environment,
              'simulation_calls': 0, 'posterior_samples': 0, 'training_runs': 0,
              'minimum_history_buffers_per_instance_bytes': {
                  g['id']: 38 * config['event_history_capacity'] * g['total_storage_positions']
                  for g in geometries}}
    if load_models:
        import torch
        from .artifacts import load_final
        torch.set_num_threads(1)
        for row in artifacts['artifacts']:
            posterior = load_final(row['geometry'], project)
            result['models_loaded'].append({'geometry': row['geometry'],
                'weights_sha256': weights_sha256(posterior),
                'cached_MAP_dimensions': list(posterior._map.shape),
                'device': str(next(posterior.posterior_estimator.parameters()).device)})
            del posterior
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('doctor', help='Validate configuration and artifact bytes; no simulations')
    check.add_argument('--project', default='.', help='Path to the source checkout with config/ and artifacts/')
    check.add_argument('--load-models', action='store_true', help='Additionally load all eight trusted final estimators on CPU')
    check.add_argument('--strict-versions', action='store_true', help='Require the pinned numerical package versions')
    geometry = sub.add_parser('geometries', help='Print physical geometry and storage dimensions')
    geometry.add_argument('--project', default='.')
    for operation in ['generate', 'train', 'select']:
        stage = sub.add_parser(operation, help='Preview or execute the ' + operation + ' stage')
        stage.add_argument('--project', default='.')
        stage.add_argument('--geometry', required=True, help='Geometry ID, e.g. 5_3_1')
        stage.add_argument('--round', type=int, required=True, dest='round_index', help='Zero-based round index')
        stage.add_argument('--run-dir', required=True, help='Separate local data/output directory')
        stage.add_argument('--workflow', default='config/workflow.json', help='Paper or custom workflow JSON')
        mode = stage.add_mutually_exclusive_group()
        mode.add_argument('--execute', action='store_true', help='Perform computation and write generated outputs')
        mode.add_argument('--dry-run', action='store_true', help='Print the plan only (default)')
        if operation == 'generate':
            stage.add_argument('--start', type=int, default=0, help='First case index for this worker')
            stage.add_argument('--stop', type=int, help='Exclusive last case index')
            stage.add_argument('--prepare-only', action='store_true', help='Fix all parameter draws without forward simulations')
        if operation == 'train':
            stage.add_argument('--task', type=int, default=0, help='Inference-task index')
    for operation in ['evaluate-map', 'predictive', 'sbc']:
        stage = sub.add_parser(operation, help='Preview or execute a frozen-estimator diagnostic')
        stage.add_argument('--project', default='.')
        stage.add_argument('--geometry', required=True)
        stage.add_argument('--run-dir', required=True, help='Separate directory for this diagnostic')
        stage.add_argument('--diagnostics', default='config/diagnostics.json', help='Paper or custom diagnostic JSON')
        stage.add_argument('--estimator', help='Optional frozen native estimator from a regenerated workflow')
        stage.add_argument('--estimator-sha256', help='Required checksum for --estimator')
        stage.add_argument('--start', type=int, default=0)
        stage.add_argument('--stop', type=int)
        stage.add_argument('--prepare-only', action='store_true', help='Fix parameters without forward simulations')
        mode = stage.add_mutually_exclusive_group()
        mode.add_argument('--execute', action='store_true')
        mode.add_argument('--dry-run', action='store_true', help='Preview only (default)')
    report = sub.add_parser('diagnostic-report', help='Summarize complete saved results; no simulation or neural sampling')
    report.add_argument('--run-dir', required=True)
    mode = report.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Read complete cases and write summaries')
    mode.add_argument('--dry-run', action='store_true', help='Preview only (default)')
    controls = sub.add_parser('diagnostic-controls', help='Analytic Gaussian rank/coverage implementation checks')
    controls.add_argument('--execute', action='store_true', help='Run only analytic controls; no cavity network/simulator')
    reproduce = sub.add_parser('reproduce', help='Rebuild numerical plots and tables from compact supplied evidence')
    reproduce.add_argument('--project', default='.')
    reproduce.add_argument('--component', choices=['all', 'selection', 'predictive', 'calibration', 'parameters'], default='all')
    reproduce.add_argument('--output-dir', default='outputs/reproduction')
    mode = reproduce.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true', help='Preview only (default)')
    plot = sub.add_parser('plot-diagnostic', help='Plot a completed diagnostic run without new simulations or neural sampling')
    plot.add_argument('--project', default='.')
    plot.add_argument('--run-dir', required=True)
    plot.add_argument('--output-dir', required=True)
    mode = plot.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true', help='Preview only (default)')
    correlations = sub.add_parser('conditional-correlations', help='Grid evaluation of a frozen density; no simulation or sampling')
    correlations.add_argument('--project', default='.')
    correlations.add_argument('--geometry', default='5_3_1')
    correlations.add_argument('--parameters', help='Comma-separated keys; default: all 26 in the source display order')
    correlations.add_argument('--resolution', type=int, help='Default: config/figures.json')
    correlations.add_argument('--condition', choices=['source-rounded-map', 'saved-map'], help='Default: config/figures.json')
    correlations.add_argument('--output-dir', default='outputs/conditional-correlations')
    mode = correlations.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true', help='Preview only (default)')
    args = parser.parse_args(argv)
    try:
        if args.command == 'doctor':
            result = doctor(args.project, args.load_models, args.strict_versions)
        elif args.command == 'geometries':
            result = geometry_configs(args.project)
        elif args.command == 'reproduce':
            from .reproduction import reproduce, reproduction_plan
            result = reproduction_plan(args.project, args.output_dir, args.component)
            if args.execute:
                result = reproduce(args.project, args.output_dir, args.component)
        elif args.command == 'plot-diagnostic':
            from .diagnostics import checked_diagnostic
            record = checked_diagnostic(args.run_dir)
            result = {'operation': 'plot-diagnostic', 'mode': 'preview', 'diagnostic': record['specification']['operation'],
                      'new_simulations': 0, 'new_posterior_draws': 0, 'training_runs': 0, 'files_written': False}
            if args.execute:
                from .reproduction import plot_saved_diagnostic
                result = plot_saved_diagnostic(args.project, args.run_dir, args.output_dir)
        elif args.command == 'conditional-correlations':
            from .correlations import correlation_plan, run_correlations
            from .reproduction import figure_config
            figure_settings = figure_config(args.project)
            resolution = figure_settings['correlation_grid_resolution'] if args.resolution is None else args.resolution
            condition = figure_settings['correlation_condition'] if args.condition is None else args.condition
            parameters = None if args.parameters is None else [p.strip() for p in args.parameters.split(',')]
            result = correlation_plan(args.project, args.geometry, args.output_dir, parameters, resolution, condition)
            if args.execute:
                result = run_correlations(args.project, args.geometry, args.output_dir, parameters, resolution, condition)
        elif args.command in ['evaluate-map', 'predictive', 'sbc']:
            from .diagnostics import diagnostic_plan, diagnostic_specification, run_diagnostic
            spec = diagnostic_specification(args.command, args.project, args.geometry, args.diagnostics,
                                            args.estimator, args.estimator_sha256)
            result = diagnostic_plan(spec, args.project, args.run_dir, args.start, args.stop, args.prepare_only)
            if args.execute:
                differences = environment_report(check_runtime=True)['reference_version_differences']
                if differences:
                    raise RuntimeError('Use the pinned execution environment: ' + ', '.join(differences))
                result = run_diagnostic(args.project, args.run_dir, spec, args.start, args.stop, args.prepare_only)
        elif args.command == 'diagnostic-report':
            from .diagnostics import checked_diagnostic, case_count
            record = checked_diagnostic(args.run_dir)
            result = {'operation': 'diagnostic-report', 'mode': 'preview',
                      'diagnostic': record['specification']['operation'],
                      'planned_cases': case_count(record['specification']),
                      'new_simulations': 0, 'new_posterior_draws': 0, 'training_runs': 0,
                      'files_written': False}
            if args.execute:
                from .diagnostic_reports import summarize_diagnostic
                result = summarize_diagnostic(args.run_dir)
        elif args.command == 'diagnostic-controls':
            result = {'operation': 'diagnostic-controls', 'mode': 'preview',
                      'new_cavity_simulations': 0, 'new_cavity_posterior_draws': 0, 'training_runs': 0}
            if args.execute:
                from .diagnostic_stats import analytic_controls
                result.update(mode='executed', controls=analytic_controls())
        else:
            from .workflow import plan, specification
            result = plan(args.command, args.project, args.geometry, args.round_index, args.run_dir,
                          args.workflow, task=getattr(args, 'task', 0),
                          start=getattr(args, 'start', 0), stop=getattr(args, 'stop', None))
            if getattr(args, 'prepare_only', False):
                result['prepare_only'] = True
                result['generation_condition_specific_calls'] = 0
            if args.execute:
                spec = specification(args.project, args.geometry, args.workflow)
                if args.command in ['generate', 'train']:
                    differences = environment_report(check_runtime=True)['reference_version_differences']
                    if differences:
                        raise RuntimeError('Use the pinned execution environment: ' + ', '.join(differences))
                if args.command == 'generate':
                    from .datasets import generate
                    result = generate(args.project, args.run_dir, spec, args.round_index,
                                      args.start, args.stop, args.prepare_only)
                elif args.command == 'train':
                    from .training import train_task
                    result = train_task(args.run_dir, spec, args.round_index, args.task)
                else:
                    from .selection import select_round
                    result = select_round(args.run_dir, spec, args.round_index)
                result['operation'] = args.command
                result['mode'] = 'executed' if not getattr(args, 'prepare_only', False) else result['mode']
    except (ValueError, OSError, KeyError, RuntimeError, metadata.PackageNotFoundError) as error:
        parser.exit(1, 'Preflight failed: %s\n' % error)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
