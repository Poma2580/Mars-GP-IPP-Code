import importlib
import json
import numpy as np
import pytest
from mars_ipp.utils.config import ROOT, load_config, merge
from mars_ipp.envs.observations import make_scaler, scaler_from_dict
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.evaluation.evaluator import env_config, evaluate
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.training.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint


def test_bnorm_is_only_v1_observation_ablation(small_field, tmp_path):
    base = load_config(ROOT / 'configs/main/bnorm_no_mh_uncertain.yaml')
    cfg = load_config(ROOT / 'configs/main/bnorm_no_mh.yaml')
    assert cfg == merge(base, {'experiment': 'bnorm_no_mh_rbf',
                               'ppo': {'observation_mode': 'full_bnorm'}})
    cfg = merge(cfg, {'ppo': {'device': 'cpu'}, 'env': {'max_steps': 2},
                      'evaluation': {'seeds': [1000]}})
    fields = {'zhurong': small_field, 'perseverance': small_field}
    scaler = make_scaler(fields, cfg)
    reference = make_scaler(fields, base)
    raw = np.array([15, 15, 2, -4, 0.])
    np.testing.assert_array_equal(scaler.transform(raw), reference.transform(np.r_[raw, np.zeros(8)])[:5])
    assert scaler_from_dict(scaler.to_dict()).to_dict() == scaler.to_dict()
    assert scaler.for_scene('perseverance').mode == scaler.mode
    with pytest.raises(ValueError, match='5D'):
        scaler.transform(np.r_[raw, np.zeros(8)])
    envs = [GridEnv(small_field, sc, env_config(c)) for sc, c in
            ((scaler, cfg), (reference, merge(cfg, {'ppo': {'observation_mode': 'full_uncertainty'}})))]
    for env in envs:
        env.reset(seed=1000)
    agent = PPO(cfg['ppo'], scaler)
    assert agent.input_dim == 5
    for action in (0, 3):
        a, b = [env.step(action) for env in envs]
        np.testing.assert_array_equal(a[0], b[0][:5])
        assert a[1:4] == b[1:4]
        for key in ('r_tr', 'r_rep', 'r_out', 'rmse', 'trr'):
            if key in a[4]:
                assert a[4][key] == b[4][key]
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, agent, cfg, {'episode': 2})
    restored = PPO(cfg['ppo'], scaler)
    restore_checkpoint(read_checkpoint(path), restored)
    for scene in fields:
        rows, _, _ = evaluate(small_field, scaler, merge(cfg, {'field': {'test_scenario': scene}}), restored)
        assert len(rows) == 4 and restored.scaler.scene == 'zhurong'


def test_explicit_cross_grid_scaler_override(small_field, scaler, config, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    script = importlib.import_module('evaluate')
    cfg = merge(config, {'ppo': {'device': 'cpu'}, 'env': {'grid_size': 3, 'max_steps': 2},
                         'evaluation': {'seeds': [1000]}})
    checkpoint = tmp_path / 'checkpoint.pt'
    save_checkpoint(checkpoint, PPO(cfg['ppo'], scaler), cfg, {'episode': 2})
    switched = merge(cfg, {'env': {'grid_size': 4, 'online_noise_variance': 1e-6, 'unique_samples': True}})
    monkeypatch.setattr(script, 'prepare_fields', lambda _: {'zhurong': small_field, 'perseverance': small_field})
    rejected = tmp_path / 'rejected'
    with pytest.raises(ValueError, match='normalization differs'):
        script.run(switched, rejected, checkpoint, plot=False)
    assert not rejected.exists()
    output = tmp_path / 'accepted'
    rows, paths = script.run(switched, output, checkpoint, plot=False, preserve_checkpoint_scaler=True)
    meta = json.loads((output / 'metadata.json').read_text())
    assert len(rows) == 4 and len(paths) == 2
    assert meta['scaler'] == scaler.to_dict()
    assert meta['preserve_checkpoint_scaler'] is True
    assert meta['checkpoint_training_config']['env']['grid_size'] == 3
    assert meta['evaluation_overrides']['env']['grid_size'] == 4
