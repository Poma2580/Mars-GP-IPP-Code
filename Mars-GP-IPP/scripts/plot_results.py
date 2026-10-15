import _bootstrap
import argparse
from mars_ipp.visualization.comparison import plot_results
from mars_ipp.visualization.aggregate import aggregate_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+')
    parser.add_argument('--output')
    parser.add_argument('--formats', nargs='+', choices=('png', 'pdf'), default=['png'])
    args = parser.parse_args()
    if len(args.directories) > 1:
        if not args.output:
            parser.error('Multiple runs require --output for a new aggregate directory')
        aggregate_results(args.directories, args.output, args.formats)
    else:
        plot_results(args.directories[0], args.output, args.formats)
