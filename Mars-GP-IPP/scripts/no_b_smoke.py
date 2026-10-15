"""Isolated two-episode validation; never writes official experiment results."""
import _bootstrap
import argparse
from copy import deepcopy
from pathlib import Path
import tempfile
import numpy as np
from mars_ipp.utils.config import load_config, merge
from mars_ipp.training.trainer import train
from mars_ipp.training.checkpoint import read_checkpoint
from mars_ipp.visualization.comparison import plot_results
from evaluate import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp-root', help='Optional parent for a new temporary directory')
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix='mars_no_b_smoke_', dir=args.temp_root))
    print(f'Smoke output: {root}', flush=True)
    config = merge(load_config(_bootstrap.ROOT / 'configs/ablation/no_b_rbf.yaml'), {
        'training': {'episodes': 2, 'eval_every': 2, 'checkpoint_every': 2},
        'evaluation': {'seeds': [1000]}})
    directory = train(config, root / 'training')
    plot_results(directory, formats=('png', 'pdf'))
    payload = read_checkpoint(directory / 'checkpoint.pt')
    assert payload['progress'] == {'episode': 2, 'steps': 140, 'completed': True}
    assert payload['config']['ppo']['observation_mode'] == 'position_only'
    assert payload['config']['ppo']['rollout_steps'] == 2000
    assert payload['config']['ppo']['update_epochs'] == 80
    for scenario in ('zhurong', 'perseverance'):
        switched = deepcopy(payload['config'])
        switched['field']['test_scenario'] = scenario
        output = root / scenario
        rows, paths = run(switched, output, str(directory / 'checkpoint.pt'), formats=('png', 'pdf'))
        assert len(rows) == 72 and len(paths) == 70
        assert np.isfinite([float(row['rmse']) for row in rows]).all()
        assert np.any(np.asarray([[row[key] for key in ('Br', 'B2', 'B3')] for row in paths]) != 0)
    for output in (directory, root / 'zhurong', root / 'perseverance'):
        pngs, pdfs = list((output / 'figures').glob('*.png')), list((output / 'figures').glob('*.pdf'))
        assert pngs and {p.stem for p in pngs} == {p.stem for p in pdfs}
    print(f'PASS: two episodes, PPO update, checkpoint, both scenarios, raw magnetic output and PNG/PDF: {root}')


if __name__ == '__main__':
    main()
