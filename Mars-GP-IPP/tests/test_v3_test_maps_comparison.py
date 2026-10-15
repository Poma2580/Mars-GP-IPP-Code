import importlib
from pathlib import Path
import numpy as np
import pytest
import matplotlib.pyplot as plt

from mars_ipp.utils.config import ROOT, load_config
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.envs.observations import make_scaler
from mars_ipp.field.field_model import FieldModel
from mars_ipp.training.checkpoint import save_checkpoint, read_checkpoint


@pytest.fixture
def runner(monkeypatch, small_field, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    cli = importlib.import_module('run_v3_test_maps_comparison')
    monkeypatch.setattr(cli, 'prepare_fields', lambda _: {'train1': small_field})
    checkpoints = {}
    for method, filename in [('v3', 'v3_gp_trace_train1.yaml'),
                             ('v3_1', 'v3_1_gp_trace_global_raw_train1.yaml')]:
        c = load_config(ROOT / 'configs/main' / filename)
        c['ppo']['device'] = 'cpu'
        c['v3']['sha256'] = small_field.sha256
        agent = PPO(c['ppo'], make_scaler({'train1': small_field}, c))
        path = tmp_path / (method + '.pt')
        save_checkpoint(path, agent, c, {'episode': 10000, 'steps': 700000, 'completed': True})
        checkpoints[method] = path
    monkeypatch.setattr(cli, 'CHECKPOINTS', checkpoints)
    return cli


@pytest.mark.parametrize('method', ['v3', 'v3_1'])
def test_frozen_policy_cross_map_keeps_original_scaler_and_checkpoint(runner, small_field, method):
    before = runner.sha256(runner.CHECKPOINTS[method])
    rows = runner.evaluate_policy(method, 'test2', small_field, [1000])
    assert len(rows) == 72 and rows[1]['stage'] == 'reset'
    assert all(r['algorithm'] == method for r in rows)
    assert runner.sha256(runner.CHECKPOINTS[method]) == before
    _, payload = runner.frozen_agent(method)
    assert payload['config']['field']['test_scenario'] == 'train1'


def test_corrupt_checkpoint_completion_rejected(runner):
    import torch
    path = runner.CHECKPOINTS['v3_1']
    payload = read_checkpoint(path)
    payload['progress']['completed'] = False
    torch.save(payload, path)
    with pytest.raises(ValueError, match='completed'):
        runner.frozen_agent('v3_1')


def test_real_gp_trace_reuse_read_only(runner, monkeypatch):
    field = FieldModel.load(runner.DEFAULT_DATA / 'test2.pkl')
    original = ROOT / 'results/comparison/test_maps_seed0/test2/gp_trace/eval_metrics.csv'
    before = runner.sha256(original)
    def unexpected_evaluation(*args, **kwargs):
        raise AssertionError('Verified historical GP-Trace should be reused')
    monkeypatch.setattr(runner, 'evaluate', unexpected_evaluation)
    rows = runner.gp_trace_rows('test2', field, list(range(1000, 1050)))
    assert len(rows) == 3600 and runner.sha256(original) == before


def test_fallback_and_three_file_plots_match_table(runner, small_field, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    runs = {method: runner.evaluate_policy(method, 'test2', small_field, [1000])
            for method in ('v3', 'v3_1')}
    runs['gp_trace'] = runner.gp_trace_rows('test2', small_field, [1000])
    figures = {}
    original_save = runner.save_figure
    def capture(fig, prefix, formats):
        figures[Path(prefix).name] = {
            'labels': [line.get_label() for line in fig.axes[0].lines],
            'endpoints': [line.get_ydata()[-1] for line in fig.axes[0].lines],
            'x': [line.get_xdata().copy() for line in fig.axes[0].lines],
            'bands': len(fig.axes[0].collections), 'title': fig.axes[0].get_title()}
        original_save(fig, prefix, formats)
    monkeypatch.setattr(runner, 'save_figure', capture)
    output = tmp_path / 'new/test2'
    summary = runner.save_scene('test2', runs, [1000], output)
    assert {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} == {
        'figures/rmse.png', 'figures/trr.png', 'comparison.tex'}
    text = (output / 'comparison.tex').read_text()
    for metric, figure in figures.items():
        assert figure['labels'] == ['V3 (n=1)', 'V3.1 (n=1)', 'GP-Trace (n=1)']
        assert figure['bands'] == 3 and figure['title'] == 'test2'
        for index, method in enumerate(runner.METHODS):
            np.testing.assert_array_equal(figure['x'][index], np.arange(71))
            assert figure['endpoints'][index] == summary[method][metric][0]
            mean, std = summary[method][metric]
            assert f'{mean:.4f} \\pm {std:.4f}' in text
    with pytest.raises(FileExistsError):
        runner.save_scene('test2', runs, [1000], output)
    runs['v3_1'][1]['trace_current'] += 1.
    with pytest.raises(AssertionError):
        runner.paired_statistics(runs, [1000])


def test_run_rejects_nonempty_output_before_evaluation(runner, tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'keep.txt'
    marker.write_text('keep')
    with pytest.raises(FileExistsError):
        runner.run(tmp_path, output, ('test2',), (1000,))
    assert marker.read_text() == 'keep'
