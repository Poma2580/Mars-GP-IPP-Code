"""Seven V3.3/V3.4 weight experiments; default is read-only preflight."""
import _bootstrap
import argparse
import logging
from pathlib import Path
from threadpoolctl import threadpool_limits
from mars_ipp.experiments.v3_weights import run
from mars_ipp.utils.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='preflight',
                        choices=('preflight', 'smoke', 'train', 'evaluate', 'plot'))
    parser.add_argument('--train-root', type=Path, default=ROOT / 'results/main',
                        help='Parent containing both v3_*_weight_{smoke,train} groups')
    parser.add_argument('--test-root', type=Path, default=ROOT / 'results/comparison')
    parser.add_argument('--output', type=Path, help='New destination for plot only')
    parser.add_argument('--smoke-results', action='store_true', help='Read smoke results for plot only')
    args = parser.parse_args()
    if args.smoke_results and args.stage != 'plot':
        parser.error('--smoke-results is supported only for plot')
    if args.stage == 'plot' and not args.output:
        parser.error('plot requires --output')
    if args.output and args.stage != 'plot':
        parser.error('--output is supported only for plot')
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger('mars_ipp.experiments').setLevel(logging.INFO)
    with threadpool_limits(limits=1):
        run(args.stage, args.train_root.resolve(), args.test_root.resolve(), args.output, args.smoke_results)


if __name__ == '__main__':
    main()
