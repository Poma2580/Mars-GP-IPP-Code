import _bootstrap
import argparse
from mars_ipp.utils.config import load_config
from mars_ipp.training.trainer import train
from mars_ipp.visualization.comparison import plot_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(_bootstrap.ROOT / 'configs/smoke.yaml'))
    parser.add_argument('--output')
    parser.add_argument('--formats', nargs='+', choices=('png', 'pdf'), default=['png'])
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = train(cfg, args.output)
    plot_results(output, formats=args.formats)
    print(output)


if __name__ == '__main__':
    main()
