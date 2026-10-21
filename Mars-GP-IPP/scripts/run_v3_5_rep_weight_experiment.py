"""V3.5 repeat-weight batches; default is read-only preflight of versions 1--4."""
import _bootstrap
import argparse
import logging
from pathlib import Path
from threadpoolctl import threadpool_limits
from mars_ipp.experiments.v3_rep_weights import run
from mars_ipp.utils.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='preflight', choices=('preflight', 'smoke', 'train', 'evaluate', 'plot'))
    parser.add_argument('--train-root', type=Path, default=ROOT / 'results/main')
    parser.add_argument('--test-root', type=Path, default=ROOT / 'results/comparison')
    parser.add_argument('--output', type=Path, help='New output directory for CSV-only plotting')
    parser.add_argument('--smoke-results', action='store_true')
    parser.add_argument('--batch', choices=('1-4', '5-10', '11-12'), default='1-4')
    args = parser.parse_args()
    if (args.stage == 'plot') != bool(args.output):
        parser.error('plot requires --output; other stages do not accept it')
    if args.smoke_results and args.stage != 'plot':
        parser.error('--smoke-results is only supported for plot')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger('GP').setLevel(logging.WARNING)
    with threadpool_limits(limits=1):
        run(args.stage, args.train_root, args.test_root, args.output, args.smoke_results, args.batch)


if __name__ == '__main__':
    main()
