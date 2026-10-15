import _bootstrap
import argparse
from mars_ipp.utils.config import load_config
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.utils.seed import seed_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(_bootstrap.ROOT / 'configs/main/matern.yaml'))
    args = parser.parse_args()
    config = load_config(args.config)
    seed_all(config['seed'])
    for name, field in prepare_fields(config).items():
        print(name, field.source, field.parameters)
