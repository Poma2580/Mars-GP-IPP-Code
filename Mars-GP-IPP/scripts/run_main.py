import _bootstrap
import argparse
from _experiments import expand
from mars_ipp.utils.config import load_config
from mars_ipp.training.trainer import train
from mars_ipp.visualization.comparison import plot_results


def main(default='configs/main/original.yaml'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(_bootstrap.ROOT / default))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--formats', nargs='+', choices=('png', 'pdf'), default=['png'])
    args = parser.parse_args()
    for config in expand(load_config(args.config)):
        print(config['experiment'], 'seed', config['seed'], 'episodes', config['training']['episodes'], flush=True)
        if not args.dry_run:
            plot_results(train(config), formats=args.formats)


if __name__ == '__main__':
    main()
