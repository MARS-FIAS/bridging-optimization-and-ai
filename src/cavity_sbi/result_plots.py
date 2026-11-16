"""Reference numerical plots; called explicitly, independently of article artwork."""
import os
from pathlib import Path
import numpy as np


def plotting(output):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(output) / 'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


STYLE = {'font.size': 10, 'axes.titlesize': 11, 'axes.spines.top': False,
         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
         'svg.fonttype': 'none', 'text.usetex': False, 'savefig.facecolor': 'white'}


def save(plt, figure, output, stem):
    paths = []
    for suffix in ['pdf', 'svg', 'png']:
        path = Path(output) / (stem + '.' + suffix)
        figure.savefig(str(path), dpi=150)
        paths.append(path)
    plt.close(figure)
    return paths


def plot_selection(groups, geometries, output):
    plt = plotting(output)
    from matplotlib.patches import Patch
    files = []
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
        for g in geometries:
            rows = [r for r in groups if r['geometry'] == g['id']]
            ax.plot([r['round_number'] for r in rows], [r['S'] for r in rows], marker='o',
                    label='(%s), %s cells' % (','.join(map(str, g['layers'])), g['physical_ICM_cells']))
        ax.axhline(1, color='0.4', linestyle='--', linewidth=1, label='Upper bound S = 1')
        ax.set(xlabel='Inference round', ylabel='Ensemble meta-score S', ylim=(0, 1.06), xticks=range(1, 6),
               title='Computed selected-MAP ensemble progression')
        ax.legend(fontsize=9, ncol=2, loc='lower right')
        files += save(plt, fig, output, 'score_progression')
        codes = {'same': 0, 'tie_only': 1, 'strict_difference': 2}
        colors = ['#E5F0F7', '#F0F1F3', '#FBE7D3']
        from matplotlib.colors import ListedColormap
        matrix = np.asarray([[codes[next(r['category'] for r in groups if r['geometry'] == g['id'] and r['round_number'] == k)]
                              for k in range(1, 6)] for g in geometries])
        fig, ax = plt.subplots(figsize=(9, 6.5))
        fig.subplots_adjust(left=.19, right=.97, top=.78, bottom=.23)
        ax.imshow(matrix, cmap=ListedColormap(colors), vmin=0, vmax=2, aspect='auto')
        for row in groups:
            i = next(i for i, g in enumerate(geometries) if g['id'] == row['geometry'])
            text = '=' if row['category'] == 'same' else 'tie' if row['category'] == 'tie_only' else '+%.4f' % row['R25_gain']
            ax.text(row['round_number']-1, i, text, ha='center', va='center', fontsize=11)
        ax.set(xticks=range(5), xticklabels=range(1, 6), yticks=range(8),
               yticklabels=['(%s)' % ','.join(map(str, g['layers'])) for g in geometries],
               xlabel='Inference round', ylabel='Geometry')
        fig.suptitle('Agreement of meta-score selection policies\nS-selected candidate also maximizes R₂₅: 36/40 groups (90%)\nAgreement in all eight final rounds', fontsize=12)
        labels = ['Same first task (34 groups)', 'Different first task; S-selected task also maximizes R₂₅ (2 groups)',
                  'Higher R₂₅ available; gain shown (4 groups)']
        fig.legend([Patch(facecolor=c) for c in colors], labels, loc='lower center', fontsize=9)
        files += save(plt, fig, output, 'score_policy_agreement')
    return files


def plot_predictive(summary, scores, times, intended, output, stem='goal_predictive', geometry='5_3_1'):
    plt = plotting(output)
    with plt.rc_context(STYLE):
        if intended is None:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
            a, b, d = axes
        else:
            fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)
            a, b, c, d = axes.flat
        q25, median, q75 = np.asarray(summary['pointwise_quantiles_q25_q50_q75'])
        a.fill_between(times, q25, q75, color='#2166AC', alpha=.2, label='25th–75th percentiles')
        a.plot(times, median, color='#2166AC', linewidth=2, label='Median')
        a.plot(times, summary['pointwise_mean'], color='#D97924', label='Mean')
        a.axhline(1, color='0.4', linestyle='--', linewidth=1, label='Ideal target')
        a.set(title='A  Joint pattern score', xlabel='Time (h)', ylabel='Score', ylim=(0, 1.06))
        a.legend(fontsize=8, ncol=2, loc='lower right')
        b.hist(np.asarray(scores).mean(axis=1), bins=np.linspace(0, 1, 21), color='#2166AC', edgecolor='white')
        count = summary['completed_parameter_draws']
        fraction = summary['fraction_of_draws_with_target_at_all_evaluated_times']
        b.text(.03, .95, 'Complete target at all 49 times:\n%d/%d pairs (%.1f%%)' % (summary['perfect_joint_pairs'], count, 100*fraction),
               transform=b.transAxes, va='top', fontsize=9, bbox={'facecolor': 'white', 'edgecolor': '.85'})
        b.set(title='B  Variation across paired predictions', xlabel='Time-averaged joint score', ylabel='Pairs', xlim=(0, 1.03))
        b.text(.03, .07, 'Highest bin includes near-perfect trajectories', transform=b.transAxes, fontsize=8)
        if intended is not None:
            labels = ['Knockout\nlower EPI', 'Knockout\nupper EPI', 'Wild type\nlower PRE', 'Wild type\nupper EPI']
            bars = c.bar(range(4), intended, color=['#2166AC', '#2166AC', '#D97924', '#2166AC'])
            for bar, value in zip(bars, intended):
                c.text(bar.get_x()+bar.get_width()/2, value+.018, '%.1f%%' % (100*value), ha='center', fontsize=9)
            c.axhline(1, color='0.4', linestyle='--', linewidth=1)
            c.set(xticks=range(4), xticklabels=labels, ylabel='Mean fraction', ylim=(0, 1.14), title='C  Cells attaining the intended fate')
            c.text(.03, .12, 'Means across draws and evaluated times', transform=c.transAxes, fontsize=9)
        d.plot(times, summary['pointwise_target_attainment'], color='#258A7D', linewidth=2)
        d.axhline(fraction, color='#D97924', linestyle='--', label='Complete target throughout window: %.1f%%' % (100*fraction))
        d.set(title='D  Complete joint target attainment', xlabel='Time (h)', ylabel='Fraction of pairs', ylim=(0, 1.05))
        d.legend(fontsize=8, loc='lower right')
        title = 'Biological target attainment across the fitted density' if summary['operation'] == 'predictive' else 'Biological target attainment at the selected MAP'
        fig.suptitle('%s\n(%s); %d pairs; S = %.4f; R₂₅ = %.4f' %
                     (title, geometry.replace('_', ','), count, summary['S'], summary['R25']), fontsize=12)
        return save(plt, fig, output, stem)


def plot_calibration(summary, pit, coverage, keys, output, figure_config):
    plt = plotting(output)
    from scipy.stats import binom
    files = []
    n, d = np.asarray(pit).shape
    bins = figure_config['rank_histogram_bins']
    reference = figure_config['rank_bin_reference_level']
    color_limit = figure_config['coverage_color_limit_percentage_points']
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(6, 5, figsize=(13, 16), constrained_layout=True)
        lower, upper = binom.ppf([(1-reference)/2, (1+reference)/2], n, 1/bins)
        for j, ax in enumerate(axes.flat):
            if j >= d:
                ax.set_axis_off()
                continue
            ax.axhspan(lower, upper, color='.9')
            ax.axhline(n/bins, color='.4', linestyle='--', linewidth=.8)
            ax.hist(pit[:, j], bins=np.linspace(0, 1, bins+1), color='#2166AC', edgecolor='white', linewidth=.4)
            ax.set(title=keys[j], xlim=(0, 1), xlabel='Normalized randomized rank', ylabel='Count')
            ax.tick_params(labelsize=9)
        fig.suptitle('Frozen-estimator marginal SBC under the original prior\n%d cases; %d conditional draws per case; all %d parameters\nGray: %g%% reference per bin, not simultaneous; Holm rejections: %d/%d' %
                     (n, summary['posterior_draws_per_case'], d, 100*reference, summary['Holm_rejections'], d), fontsize=12)
        files += save(plt, fig, output, 'sbc_ranks')
        levels = np.asarray(summary['coverage_levels'])
        empirical = np.asarray(coverage).mean(axis=0).T
        fig, ax = plt.subplots(figsize=(8, 11), constrained_layout=True)
        mesh = ax.imshow(100*(empirical-levels), cmap='RdBu_r', vmin=-color_limit, vmax=color_limit, aspect='auto')
        ax.set(xticks=range(len(levels)), xticklabels=['%g%%' % (100*c) for c in levels],
               yticks=range(d), yticklabels=keys, xlabel='Nominal central-interval coverage',
               title='Empirical marginal interval coverage\n%d original-prior cases' % n)
        for j in range(d):
            for k in range(len(levels)):
                ax.text(k, j, '%.1f%%' % (100*empirical[j, k]), ha='center', va='center', fontsize=9,
                        color='white' if abs(empirical[j, k]-levels[k]) > .16 else 'black')
        fig.colorbar(mesh, ax=ax, label='Empirical − nominal (percentage points)', extend='both', shrink=.65)
        fig.supxlabel('Colors clipped at ±%g percentage points; numbers show full coverage' % color_limit, fontsize=9)
        files += save(plt, fig, output, 'sbc_coverage')
        fig, axes = plt.subplots(6, 5, figsize=(13, 16), constrained_layout=True)
        grid = np.linspace(0, 1, 300)
        band = summary['DKW_familywise_band']
        for j, ax in enumerate(axes.flat):
            if j >= d:
                ax.set_axis_off()
                continue
            ax.fill_between(grid, np.maximum(0, grid-band), np.minimum(1, grid+band), color='.9')
            ax.plot(grid, grid, color='.4', linestyle='--', linewidth=.8)
            ax.plot(np.sort(pit[:, j]), np.arange(1, n+1)/n, color='#2166AC')
            ax.set(title='%s · D=%.3f' % (keys[j], summary['parameters'][j]['KS_D']), xlim=(0, 1), ylim=(0, 1),
                   xlabel='Normalized randomized rank', ylabel='Empirical CDF')
        fig.suptitle('Marginal SBC rank CDFs · all %d parameters\nGray: %g%% familywise DKW reference; original-prior law' % (d, 100*(1-summary['alpha'])), fontsize=12)
        files += save(plt, fig, output, 'sbc_rank_cdf')
    return files


def plot_parameters(map_rows, reference_rows, geometries, figure_config, output):
    plt = plotting(output)
    groups = figure_config['parameter_groups']
    reference = {r['parameter']: r for r in reference_rows}
    maps = {(r['geometry'], r['parameter']): float(r['MAP_prior_percent']) for r in map_rows}
    if len(groups) != 4 or sorted(k for group in groups for k in group['parameters']) != sorted(reference):
        raise ValueError('Parameter figure must retain the four groups and all 26 parameters')
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:pink']
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.subplots_adjust(left=.12, right=.98, bottom=.17, top=.86, wspace=.42, hspace=.45)
        for ax, group in zip(axes.flat, groups):
            stats = []
            for key in group['parameters']:
                row = reference[key]
                stats.append({'label': 'FGF4-LUMEN (FL)' if key == 'FL' else key,
                              'whislo': float(row['q025_prior_percent']), 'q1': float(row['q25_prior_percent']),
                              'med': float(row['median_prior_percent']), 'q3': float(row['q75_prior_percent']),
                              'whishi': float(row['q975_prior_percent']), 'fliers': []})
            ax.bxp(stats, vert=False, positions=np.arange(len(stats)), widths=.55, showfliers=False,
                   boxprops={'color': '.5'}, whiskerprops={'color': '.5'}, capprops={'color': '.5'}, medianprops={'color': '.2'})
            for g, marker, color in zip(geometries, markers, colors):
                ax.scatter([maps[g['id'], k] for k in group['parameters']], np.arange(len(stats)),
                           marker=marker, color=color, s=28, zorder=3, label='(%s): %d cells' % (g['id'].replace('_', ','), g['physical_ICM_cells']))
            ax.invert_yaxis()
            ax.set(xlim=(0, 100), xticks=[0, 25, 50, 75, 100], xlabel='Parameter position within prior range (%)', title=group['title'])
            ax.grid(axis='x', color='.93', zorder=0)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9)
        fig.suptitle('Selected MAP parameterizations and fitted reference ranges\nDots: geometry-specific MAP; reference: (5,3,1), 256 ideal-target draws\nBox: 25–75%; line: median; whiskers: 2.5–97.5% of the fitted density', fontsize=12)
        return save(plt, fig, output, 'parameter_reference')
