"""Explicit reproduction from compact evidence; no neural/simulator execution."""
import csv
from pathlib import Path

from .artifacts import file_sha256
from .configuration import geometry_configs, model_config, project_file, read_json
from .evidence import evidence_inventory
from .workflow import atomic_json, validate_run_location

COMPONENTS = ('all', 'selection', 'predictive', 'calibration', 'parameters')


def figure_config(project='.'):
    cfg = read_json(project, 'config/figures.json')
    if type(cfg['rank_histogram_bins']) is not int or cfg['rank_histogram_bins'] < 2:
        raise ValueError('Invalid rank histogram bin count')
    if not 0 < cfg['rank_bin_reference_level'] < 1 or not 0 < cfg['coverage_color_limit_percentage_points'] <= 100:
        raise ValueError('Invalid plot reference or color scale')
    keys = [p['key'] for p in model_config(project)['parameters']]
    if sorted(k for group in cfg['parameter_groups'] for k in group['parameters']) != sorted(keys):
        raise ValueError('Parameter panels must contain all model parameters exactly once')
    if sorted(cfg['correlation_display_order']) != sorted(keys):
        raise ValueError('Correlation display order differs from model parameters')
    return cfg


def reproduction_plan(project, output, component='all'):
    if component not in COMPONENTS:
        raise ValueError('Unknown reproduction component')
    manifest = evidence_inventory(project)
    figure_config(project)
    return {'operation': 'reproduce', 'mode': 'preview', 'component': component,
            'output_directory': str(validate_run_location(project, output)),
            'evidence_files_verified': len(manifest['files']),
            'evidence_bytes': sum(r['bytes'] for r in manifest['files'].values()),
            'input': 'Supplied compact derived evidence, not simulation or posterior-draw banks',
            'outputs': 'Recomputed JSON/CSV summaries and reference PDF/SVG/PNG plots',
            'neural_models_loaded': 0, 'new_simulations': 0, 'new_posterior_draws': 0,
            'training_runs': 0, 'files_written': False, 'execute_required': True}


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot write an empty derived table')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return Path(path)


def verify_values(actual, expected):
    import numpy as np
    for key, value in expected.items():
        other = actual[key]
        if isinstance(value, (int, float)):
            if not np.isclose(other, value, rtol=0, atol=1e-12):
                raise ValueError('Recomputed result differs: ' + key)
        elif other != value:
            raise ValueError('Recomputed result differs: ' + key)


def _output_manifest(project, output, files, result):
    import platform
    import numpy as np
    import scipy
    import matplotlib
    result.update({'software': {'python': platform.python_version(), 'numpy': np.__version__,
                               'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__},
        'files': {str(p.relative_to(output)): {'bytes': p.stat().st_size, 'sha256': file_sha256(p)} for p in files},
        'code_sha256': {str(p.relative_to(Path(__file__).parent)): file_sha256(p)
                       for p in sorted(Path(__file__).parent.rglob('*.py'))}})
    atomic_json(output / 'reproduction_manifest.json', result)
    return result


def reproduce(project, output, component='all'):
    from .diagnostic_stats import calibration_summary
    from .evidence import evidence_arrays, evidence_rows
    from .result_analysis import parameter_analysis, predictive_analysis, selection_analysis
    from .result_plots import plot_calibration, plot_parameters, plot_predictive, plot_selection
    import numpy as np
    plan = reproduction_plan(project, output, component)
    output = Path(plan['output_directory'])
    cfg, geometries, figures = model_config(project), geometry_configs(project), figure_config(project)
    expected = read_json(project, 'evidence/expected_results.json')
    output.mkdir(parents=True, exist_ok=False)
    files, summaries = [], {}
    selected = set(COMPONENTS[1:]) if component == 'all' else {component}
    if 'selection' in selected:
        summary, groups, candidates = selection_analysis(evidence_arrays(project, 'selection_candidates.npz'), geometries, cfg)
        verify_values(summary, expected['selection'])
        summaries['selection'] = summary
        files += [write_csv(output / 'candidate_scores.csv', candidates), write_csv(output / 'selection_groups.csv', groups)]
        files.append(write_csv(output / 'score_progression.csv',
            [{k: row[k] for k in ['geometry', 'cells', 'round_number', 'selected_task', 'S', 'R25']} for row in groups]))
        files += plot_selection(groups, geometries, output)
    if 'predictive' in selected:
        data = evidence_arrays(project, 'goal_predictive.npz')
        summary, times, histogram, intended = predictive_analysis(data, cfg, 256)
        verify_values(summary, expected['predictive'])
        summaries['predictive'] = summary
        files += [write_csv(output / 'predictive_score_timeseries.csv', times),
                  write_csv(output / 'predictive_score_histogram.csv', histogram),
                  write_csv(output / 'predictive_intended_fates.csv',
                            [{'cell_group': key, 'mean_fraction': value} for key, value in summary['mean_intended_fate_fractions'].items()])]
        files += plot_predictive(summary, data['scores'], data['time_h'], intended, output)
        baseline, baseline_times, _, _ = predictive_analysis(evidence_arrays(project, 'prior_predictive.npz'), cfg, 1000, 'prior-predictive')
        verify_values(baseline, expected['prior_predictive'])
        summaries['prior_predictive'] = baseline
        files.append(write_csv(output / 'prior_predictive_score_timeseries.csv', baseline_times))
    if 'calibration' in selected:
        data = evidence_arrays(project, 'marginal_sbc.npz')
        keys = data['parameter_keys'].tolist()
        summary = calibration_summary(data['ranks'], data['rank_pit'], data['coverage'], data['case_ids'], keys,
                                      int(data['posterior_draws_per_case']), data['coverage_levels'], 1000)
        for actual, original in zip(summary['parameters'], expected['calibration']['parameters']):
            verify_values(actual, original)
        verify_values(summary, {'Holm_rejections': expected['calibration']['Holm_rejections'],
                               'DKW_familywise_band': expected['calibration']['familywise_DKW_band']})
        summaries['calibration'] = summary
        rows, histograms = [], []
        edges = np.linspace(0, 1, figures['rank_histogram_bins']+1)
        for j, row in enumerate(summary['parameters']):
            for level, empirical in zip(summary['coverage_levels'], row['central_interval_coverage']):
                rows.append({'parameter': row['key'], 'nominal': level, 'empirical': empirical,
                             'empirical_minus_nominal': empirical-level})
            counts, _ = np.histogram(data['rank_pit'][:, j], bins=edges)
            histograms.extend({'parameter': row['key'], 'bin_lower': float(edges[k]), 'bin_upper': float(edges[k+1]),
                               'count': int(count), 'uniform_expected_count': 1000/figures['rank_histogram_bins']}
                              for k, count in enumerate(counts))
        parameter_rows = [{key: row[key] for key in ['key', 'KS_D', 'KS_p', 'Holm_adjusted_p']} for row in summary['parameters']]
        files += [write_csv(output / 'calibration_parameters.csv', parameter_rows),
                  write_csv(output / 'calibration_coverage.csv', rows), write_csv(output / 'rank_histograms.csv', histograms)]
        files += plot_calibration(summary, data['rank_pit'], data['coverage'], keys, output, figures)
    if 'parameters' in selected:
        maps = evidence_rows(project, 'selected_MAP_parameters.csv')
        reference = evidence_rows(project, 'reference_density_quantiles_256.csv')
        summaries['parameters'] = parameter_analysis(maps, reference, cfg, geometries)
        files += [write_csv(output / 'selected_MAP_parameters.csv', maps),
                  write_csv(output / 'reference_density_quantiles_256.csv', reference),
                  write_csv(output / 'parameter_priors.csv', cfg['parameters'])]
        dimensions = [{key: (str(value) if isinstance(value, list) else value) for key, value in g.items()} for g in geometries]
        files.append(write_csv(output / 'geometry_dimensions.csv', dimensions))
        files += plot_parameters(maps, reference, geometries, figures, output)
    atomic_json(output / 'results.json', summaries)
    files.append(output / 'results.json')
    result = dict(plan, mode='executed', files_written=True,
        evidence_manifest_sha256=file_sha256(project_file(project, 'evidence/manifest.json')),
        model_configuration_sha256=file_sha256(project_file(project, 'config/model.json')),
        figure_configuration_sha256=file_sha256(project_file(project, 'config/figures.json')),
        interpretation='Reference numerical plots from computed evidence; article artwork is a separate asset.',
        components=sorted(selected))
    return _output_manifest(project, output, files, result)


def plot_saved_diagnostic(project, run, output):
    """Refresh a complete public-format report and plot it without neural work."""
    from .diagnostic_reports import summarize_diagnostic
    from .diagnostics import case_count, case_identity, checked_diagnostic, read_array_record
    from .result_analysis import fate_counts_from_raw, predictive_analysis
    from .result_plots import plot_calibration, plot_predictive
    import numpy as np
    run = Path(run).resolve()
    record = checked_diagnostic(run)
    spec = record['specification']
    output = validate_run_location(project, output)
    if output == run or output in run.parents:
        raise ValueError('Use a separate plot output directory, not an input directory or its parent')
    if output.exists():
        raise FileExistsError('Use a new plot output directory: ' + str(output))
    summary = summarize_diagnostic(run)
    output.mkdir(parents=True, exist_ok=False)
    count = case_count(spec)
    if spec['operation'] == 'sbc':
        with np.load(run / 'reports/sbc_rank_arrays.npz', allow_pickle=False) as data:
            files = plot_calibration(summary, data['rank_pit'], data['coverage'], data['parameter_keys'].tolist(), output, figure_config(project))
    else:
        with np.load(run / 'reports/predictive_scores.npz', allow_pickle=False) as source:
            data = {'scores': np.array(source['scores']), 'time_h': np.array(source['score_time_hours']), 'case_ids': np.array(source['case_ids'])}
        if spec['settings']['keep_raw_trajectories']:
            counts = []
            for i in range(count):
                arrays, _ = read_array_record(run / 'observations' / ('case_%06d.npz' % i), case_identity(record, 'observation'), i)
                counts.append(fate_counts_from_raw(arrays['raw_trajectories'], spec['geometry']['layers'], spec['model']))
            lower = spec['geometry']['layers'][0]
            data.update(fate_counts=np.stack(counts), group_cell_counts=np.asarray([lower, spec['geometry']['physical_ICM_cells']-lower]))
        summary, _, _, intended = predictive_analysis(data, spec['model'], count, spec['operation'])
        files = plot_predictive(summary, data['scores'], data['time_h'], intended, output,
                                stem='diagnostic_predictive', geometry=spec['geometry']['id'])
    atomic_json(output / 'results.json', summary)
    files.append(output / 'results.json')
    return _output_manifest(project, output, files, {'operation': 'plot-diagnostic', 'mode': 'executed',
        'diagnostic_fingerprint': record['fingerprint'], 'new_simulations': 0, 'new_posterior_draws': 0,
        'neural_models_loaded': 0, 'training_runs': 0})
