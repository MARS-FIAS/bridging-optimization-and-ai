"""Check an installed plotting or CPU environment without biological simulation.

Run after installation from a source checkout. Preview is default; --execute
runs dependency checks, reference plots and (CPU mode) analytic/unit controls.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--mode', choices=['plots', 'cpu'], default='plots')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    from cavity_sbi.workflow import validate_run_location
    project = args.project.resolve()
    output = validate_run_location(project, args.output_dir)
    python = [sys.executable, '-I', '-B']
    cli = python + ['-m', 'cavity_sbi']
    doctor = cli + ['doctor', '--project', str(project)]
    if args.mode == 'cpu':
        doctor += ['--load-models', '--strict-versions']
    commands = [('dependencies', python+['-m', 'pip', 'check']),
                ('version', cli+['--version']), ('configuration-and-artifacts', doctor),
                ('reference-plots', cli+['reproduce', '--project', str(project), '--output-dir', str(output/'reference-plots'), '--execute'])]
    if args.mode == 'cpu':
        commands += [('analytic-controls', cli+['diagnostic-controls', '--execute']),
                     ('unit-controls', python+['-m', 'unittest', 'discover', '-s', str(project/'tests'), '-v'])]
        for operation in ['generate', 'train', 'select']:
            commands.append((operation+'-preview', cli+[operation, '--project', str(project), '--geometry', '5_3_1',
                '--round', '0', '--run-dir', str(output/'planned-workflow')]))
        for operation in ['evaluate-map', 'predictive', 'sbc']:
            commands.append((operation+'-preview', cli+[operation, '--project', str(project), '--geometry', '5_3_1',
                '--run-dir', str(output/('planned-'+operation))]))
    result = {'mode': args.mode, 'execution': 'executed' if args.execute else 'preview',
              'new_biological_simulations': 0, 'new_trained_network_posterior_draws': 0, 'neural_training_runs': 0,
              'commands': [{'name': name, 'argv': argv} for name, argv in commands]}
    if not args.execute:
        print(json.dumps(result, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    cache = output/'cache'
    for name in ['tmp', 'matplotlib', 'numba', 'torch']:
        (cache/name).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(cache/'tmp'), MPLBACKEND='Agg',
               MPLCONFIGDIR=str(cache/'matplotlib'), NUMBA_CACHE_DIR=str(cache/'numba'),
               XDG_CACHE_HOME=str(cache), TORCH_HOME=str(cache/'torch'), PIP_DISABLE_PIP_VERSION_CHECK='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for name, argv in commands:
        with (output/(name+'.stdout')).open('w') as out, (output/(name+'.stderr')).open('w') as err:
            process = subprocess.run(argv, cwd=project, env=env, stdout=out, stderr=err)
        next(item for item in result['commands'] if item['name'] == name)['exit_code'] = process.returncode
        result['passed'] = process.returncode == 0
        (output/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
        print(('PASS ' if process.returncode == 0 else 'FAIL ') + name, flush=True)
        if process.returncode:
            return process.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
