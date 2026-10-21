"""Masked V3.3.4/V3.5 experiments; defaults to read-only preflight."""
import _bootstrap
import argparse
import logging
from pathlib import Path
from threadpoolctl import threadpool_limits
from mars_ipp.experiments.v3_masked import run
from mars_ipp.utils.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='preflight', choices=('preflight', 'smoke', 'train', 'evaluate', 'plot'))
    parser.add_argument('--train-root', type=Path, default=ROOT / 'results/main')
    parser.add_argument('--test-root', type=Path, default=ROOT / 'results/comparison')
    parser.add_argument('--output', type=Path, help='New directory for CSV-only plots')
    parser.add_argument('--smoke-results', action='store_true', help='Replot smoke CSVs')
    args = parser.parse_args()
    if (args.stage == 'plot') != bool(args.output):
        parser.error('plot requires --output; other stages do not accept it')
    if args.smoke_results and args.stage != 'plot':
        parser.error('--smoke-results is only supported for plot')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    with threadpool_limits(limits=1):
        run(args.stage, args.train_root.resolve(), args.test_root.resolve(), args.output, args.smoke_results)


if __name__ == '__main__':
    main()
