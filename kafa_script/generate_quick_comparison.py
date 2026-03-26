import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kafa_script.experiment_tooling import (DEFAULT_ANALYSIS_ROOT,
                                            METHOD_SPECS, TASK_SPECS,
                                            build_and_save_artifacts,
                                            default_client_model,
                                            default_server_model)


def main():
    parser = argparse.ArgumentParser(
        description='Generate KAFA real/synthetic summaries and report.')
    parser.add_argument('--tasks',
                        nargs='+',
                        default=list(TASK_SPECS.keys()),
                        help='Tasks to summarize.')
    parser.add_argument('--methods',
                        nargs='+',
                        default=list(METHOD_SPECS.keys()),
                        help='Methods to include.')
    parser.add_argument('--seeds',
                        nargs='+',
                        type=int,
                        default=[42, 43, 44],
                        help='Synthetic seeds / expected seeds.')
    parser.add_argument('--exp-root',
                        default=str(Path('exp')),
                        help='Root directory containing experiment logs.')
    parser.add_argument('--analysis-root',
                        default=str(DEFAULT_ANALYSIS_ROOT),
                        help='Directory to write analysis outputs.')
    parser.add_argument('--data-root',
                        default='data',
                        help='Dataset root directory.')
    parser.add_argument('--server-model',
                        default=default_server_model(),
                        help='Server model path.')
    parser.add_argument('--client-model',
                        default=default_client_model(),
                        help='Client model path.')
    parser.add_argument('--write-configs',
                        action='store_true',
                        help='Also emit experiment configs.')
    args = parser.parse_args()

    outputs = build_and_save_artifacts(tasks=args.tasks,
                                       methods=args.methods,
                                       seeds=args.seeds,
                                       exp_root=Path(args.exp_root),
                                       analysis_root=Path(args.analysis_root),
                                       data_root=args.data_root,
                                       server_model=args.server_model,
                                       client_model=args.client_model,
                                       write_configs=args.write_configs)

    print('Generated outputs:')
    for name, path in outputs.items():
        print(f'- {name}: {path}')


if __name__ == '__main__':
    main()
