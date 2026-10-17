"""Independent V3.1–V3.4 experiments; defaults to read-only preflight."""
import _bootstrap
import argparse
import logging
from pathlib import Path
from threadpoolctl import threadpool_limits
from mars_ipp.experiments.v3_pipeline import (
    preflight, prepare_data, protect_history, verify_hashes, copy_history,
    train_variant, test_policies, replot, replot_training_rewards, VERSIONS)
from mars_ipp.utils.config import ROOT, load_config
from mars_ipp.utils.logger import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preflight','smoke','train','evaluate','plot','plot-training'), nargs='?', default='preflight')
    parser.add_argument('--train-root', type=Path)
    parser.add_argument('--test-root', type=Path)
    parser.add_argument('--output', type=Path, help='New output directory for plot stage; plot-training updates only reward PNGs in train-root')
    args = parser.parse_args()
    smoke = args.stage == 'smoke'
    train_root = args.train_root or ROOT / ('results/main/v3_1_v3_4_smoke' if smoke else 'results/main/v3_1_v3_4_train')
    test_root = args.test_root or ROOT / ('results/comparison/v3_1_v3_4_smoke' if smoke else 'results/comparison/v3_1_v3_4_test')
    if args.stage == 'plot' and not args.output:
        parser.error('plot requires --output pointing to a new directory')
    if args.stage == 'preflight':
        report=preflight()
        print('Preflight passed; CUDA:',report['cuda_available'])
        print('V3.1:',report['historical_checkpoint'],report['historical_sha256'])
        print('10 maps verified; no files copied, no training started.')
        return
    (ROOT/'logs').mkdir(exist_ok=True)
    handlers=[logging.StreamHandler(), logging.FileHandler(ROOT/f'logs/v3_1_v3_4_{args.stage}.log')]
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s',handlers=handlers)
    logging.getLogger('mars_ipp.experiments.v3_pipeline').setLevel(logging.INFO)
    with threadpool_limits(limits=1):
        protected=protect_history()
        if args.stage == 'plot-training':
            targets = {(train_root/v/'figures/training.png').resolve() for v in ('v3_1',*VERSIONS)}
            protected = {p: h for p, h in protected.items() if Path(p).resolve() not in targets}
        try:
            if args.stage == 'plot-training':
                replot_training_rewards(train_root)
            if args.stage in ('smoke','train'):
                preflight()
                # Reject all output conflicts before any training starts.
                for path in [*(train_root/v for v in ('v3_1',*VERSIONS)), *([test_root] if smoke else [])]:
                    if path.exists() and any(path.iterdir()):
                        raise FileExistsError(path)
                data=prepare_data(copy=True)
                copy_history(train_root/'v3_1')
                write_json(train_root/'others/data_manifest.json',data)
                for version in VERSIONS:
                    vh=logging.FileHandler(ROOT/f'logs/v3_1_v3_4_{version}_{args.stage}.log')
                    logging.getLogger().addHandler(vh)
                    try:
                        train_variant(load_config(ROOT/f'configs/main/{version}.yaml'),train_root/version,smoke)
                    finally:
                        logging.getLogger().removeHandler(vh);vh.close()
            if args.stage in ('smoke','evaluate'):
                test_policies(train_root,test_root,smoke)
            if args.stage == 'plot':
                replot(train_root,test_root,args.output)
        finally:
            verify_hashes(protected)
        if args.stage in ('smoke','train'):
            write_json(train_root/'others/protected_inputs.json',protected)
            write_json(train_root/'others/acceptance.json',{'stage':args.stage,'completed':True,
                'protected_files_unchanged':len(protected),'formal_training_started':not smoke,
                'note':'冒烟验收，非正式性能结果' if smoke else 'Formal training'})


if __name__ == '__main__':
    main()
