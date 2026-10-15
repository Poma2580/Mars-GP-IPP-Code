import _bootstrap
import argparse
from mars_ipp.evaluation.no_b_diagnostics import run_diagnostics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare final Full/No-B evaluations without retraining')
    defaults = {
        'full_zhurong': 'full_rbf_eval_zhurong_seed_0',
        'full_perseverance': 'full_rbf_eval_seed_0',
        'no_b_zhurong': 'no_b_rbf_eval_zhurong_seed_0',
        'no_b_perseverance': 'no_b_rbf_eval_perseverance_seed_0',
        'full_training': 'full_rbf/seed_0', 'no_b_training': 'no_b_rbf/seed_0', 'output': 'no_b_rbf'}
    for name, path in defaults.items():
        parser.add_argument('--' + name.replace('_', '-'), default=str(_bootstrap.ROOT / 'results/main' / path))
    args = parser.parse_args()
    run_diagnostics(**vars(args))
    print(args.output)
