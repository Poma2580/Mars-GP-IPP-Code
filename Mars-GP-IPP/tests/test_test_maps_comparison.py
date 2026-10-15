import importlib
import json
import numpy as np
import pytest
from mars_ipp.utils.config import ROOT, load_config, merge
from mars_ipp.envs.observations import make_scaler
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.agents.baselines.msr import Msr
from mars_ipp.training.checkpoint import save_checkpoint, read_checkpoint


@pytest.fixture
def runner(monkeypatch, small_field, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    module = importlib.import_module('run_test_maps_comparison')
    fields = {'zhurong': small_field, 'perseverance': small_field}
    monkeypatch.setattr(module, 'prepare_fields', lambda _: fields)
    checkpoints = {}
    for method, filename in (('msr', 'msr.yaml'), ('v2', 'bnorm_no_mh_local_global_3x3_rbf.yaml'),
                             ('v2_global_only', 'bnorm_no_mh_local_global_3x3_rbf_no_local.yaml')):
        cfg = merge(load_config(ROOT / 'configs/main' / filename), {'ppo': {'device': 'cpu'}})
        scaler = make_scaler(fields, cfg)
        agent = (Msr if method == 'msr' else PPO)(cfg['ppo'], scaler)
        path = tmp_path / f'{method}.pt'
        save_checkpoint(path, agent, cfg, {'episode': 10000, 'steps': 700000, 'completed': True})
        checkpoints[method] = path
    monkeypatch.setattr(module, 'CHECKPOINTS', checkpoints)
    return module


@pytest.mark.parametrize('method', ['msr', 'v2', 'v2_global_only'])
def test_new_scene_scaler_keeps_original_restore_and_policy(runner, small_field, method):
    original = read_checkpoint(runner.CHECKPOINTS[method])
    cfg, agent, scaler, payload = runner.evaluation_setup(method, 'test2', small_field, [1000])
    assert payload['config'] == original['config']
    assert agent.scaler.to_dict() == original['scaler']
    assert cfg['field']['test_scenario'] == 'test2'
    assert cfg['env']['unique_samples'] and cfg['env']['online_noise_variance'] == 1e-6
    assert scaler.coordinate_scale == original['scaler']['coordinate_scale']
    if method == 'msr':
        assert scaler is agent.scaler and agent.input_dim == 2
    else:
        assert scaler.mode == original['scaler']['mode']
        assert scaler.scene == 'test2' and set(scaler.scenes) == {'test2'}
        assert 'test2' not in agent.scaler.scenes
        np.testing.assert_array_equal(scaler.transform(np.r_[15, 15, np.zeros(20)])[:2], [1, 1])
    for key, value in original['network'].items():
        assert value.equal(agent.network.state_dict()[key])


@pytest.mark.parametrize('method', ['gp_greedy', 'gp_trace', 'msr', 'cover', 'v2', 'v2_global_only'])
def test_six_method_evaluation_reuse_and_corruption_recovery(runner, small_field, tmp_path, method):
    target = tmp_path / 'test2' / method
    before = {name: runner.sha256(path) for name, path in runner.CHECKPOINTS.items()}
    rows, paths, saved = runner.evaluate_method(method, 'test2', small_field, [90, 15], target, [1000])
    assert len(rows) == 72 and len(paths) == 70 and saved == target
    meta = json.loads((saved / 'metadata.json').read_text())
    assert meta['field']['origin_degrees'] == [90, 15]
    assert meta['evaluation_scaler']['coordinate_scale'] == 15
    assert meta['c_global'] is not None if method.startswith('v2') else meta['c_global'] is None
    _, _, reused = runner.evaluate_method(method, 'test2', small_field, [90, 15], target, [1000])
    assert reused == target
    trajectory = target / 'trajectories.csv'
    trajectory.write_text(trajectory.read_text().splitlines()[0] + '\n')
    _, _, rerun = runner.evaluate_method(method, 'test2', small_field, [90, 15], target, [1000])
    assert rerun.name == method + '_rerun_1' and target.exists()
    assert before == {name: runner.sha256(path) for name, path in runner.CHECKPOINTS.items()}


def test_checkpoint_completion_guard(runner):
    import torch
    path = runner.CHECKPOINTS['v2']
    payload = read_checkpoint(path)
    payload['progress']['completed'] = False
    torch.save(payload, path)
    with pytest.raises(ValueError, match='completed seed-0'):
        runner.frozen_agent('v2')
