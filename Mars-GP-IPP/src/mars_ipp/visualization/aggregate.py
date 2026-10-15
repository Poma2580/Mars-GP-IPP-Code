"""Aggregate explicitly selected runs; preserve run identity and field provenance."""
from pathlib import Path
from mars_ipp.utils.config import validate_output_path
import hashlib
import json
import yaml
from .learning_curves import read_csv
from .comparison import plot_results
from mars_ipp.utils.logger import write_csv, write_json


def aggregate_results(directories, output, formats=('png',)):
    directories = list(map(Path, directories))
    if any(not (directory / 'eval_metrics.csv').exists() for directory in directories):
        raise ValueError('Multi-run aggregation requires formal evaluation eval_metrics.csv; run scripts/evaluate.py first')
    output = validate_output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, sources = [], []
    identities = set()
    for directory in map(Path, directories):
        config = yaml.safe_load((directory / 'config.yaml').read_text())
        identity = (config['experiment'], config['seed'])
        if identity in identities:
            raise ValueError(f'Duplicate experiment/seed {identity}; do not count reruns as independent seeds')
        identities.add(identity)
        path = directory / 'eval_metrics.csv'
        run_rows = read_csv(path)
        last = max(int(r['episode']) for r in run_rows)
        for row in run_rows:
            if int(row['episode']) == last:
                row['algorithm'] = config['experiment'] + ':' + row['algorithm']
                row['training_seed'] = config['seed']
                rows.append(row)
        metadata_path = directory / 'metadata.json'
        sources.append({'directory': str(directory.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'experiment': config['experiment'], 'training_seed': config['seed'], 'episode': last,
                        'fields': json.loads(metadata_path.read_text()).get('fields') if metadata_path.exists() else None})
    if len({s['episode'] for s in sources}) != 1:
        raise ValueError('Aggregate only checkpoints evaluated at the same training episode')
    write_csv(output / 'eval_metrics.csv', rows)
    write_json(output / 'sources.json', sources)
    plot_results(output, formats=formats)
