import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.experiment_tooling import (DEFAULT_ANALYSIS_ROOT,
                                            DEFAULT_DATA_ROOT,
                                            DEFAULT_OUTPUT_ROOT,
                                            METHOD_SPECS, TASK_SPECS,
                                            build_experiment_plan,
                                            default_client_model,
                                            default_server_model)


def _profile_defaults(profile: str):
    if profile == 'smoke':
        return ['gsm8k'], [
            'local-only', 'rawfedavg', 'fedprox', 'fedopt', 'offsite-tuning',
            'pfedme', 'ditto', 'kafa-full'
        ], [42]
    if profile == 'paper':
        return list(TASK_SPECS.keys()), list(METHOD_SPECS.keys()), [42, 43, 44]
    return ['gsm8k'], ['local-only', 'rawfedavg', 'fedprox', 'fedopt',
                       'offsite-tuning', 'pfedme', 'ditto', 'kafa-full'], [42]


def main():
    parser = argparse.ArgumentParser(
        description='Generate and optionally execute KAFA experiment matrix.')
    parser.add_argument('--profile',
                        choices=['smoke', 'paper'],
                        default='smoke',
                        help='Preset matrix size.')
    parser.add_argument('--tasks', nargs='+', default=None)
    parser.add_argument('--methods', nargs='+', default=None)
    parser.add_argument('--seeds',
                        nargs='+',
                        type=int,
                        default=None)
    parser.add_argument('--data-root',
                        default=str(DEFAULT_DATA_ROOT),
                        help='Dataset root directory.')
    parser.add_argument('--output-root',
                        default=str(DEFAULT_OUTPUT_ROOT),
                        help='Experiment output root directory.')
    parser.add_argument('--server-model',
                        default=default_server_model(),
                        help='Server-side model path.')
    parser.add_argument('--client-model',
                        default=default_client_model(),
                        help='Client-side compressed model path.')
    parser.add_argument('--write-configs',
                        action='store_true',
                        help='Write YAML files under the output root.')
    parser.add_argument('--execute',
                        action='store_true',
                        help='Execute the generated commands.')
    args = parser.parse_args()

    default_tasks, default_methods, default_seeds = _profile_defaults(
        args.profile)
    tasks = args.tasks or default_tasks
    methods = args.methods or default_methods
    seeds = args.seeds or default_seeds

    plan = build_experiment_plan(tasks=tasks,
                                 methods=methods,
                                 seeds=seeds,
                                 data_root=args.data_root,
                                 output_root=args.output_root,
                                 server_model=args.server_model,
                                 client_model=args.client_model,
                                 write_configs=args.write_configs)

    print('Planned commands:')
    for _, row in plan.iterrows():
        note = f" [{row['status']}]" if row['status'] != 'ready' else ''
        print(
            f"- [{row['task']}] {row['method']} seed={row['seed']}{note}: {row['command']}"
        )

    print(f'\nPlan saved under: {Path(args.output_root).resolve()}')
    print(
        f'Analysis root: {DEFAULT_ANALYSIS_ROOT.resolve()} (report is generated there)'
    )

    if not args.execute:
        print('\nDry run only. Add --execute to launch experiments.')
        return

    root = Path(__file__).resolve().parents[1]
    for _, row in plan.iterrows():
        print(f"\n>>> Running {row['task']} / {row['method']} / seed={row['seed']}")
        subprocess.run(row['command'].split(), cwd=str(root), check=True)


if __name__ == '__main__':
    main()
