from copy import deepcopy
import numpy as np
import pytest
from mars_ipp.utils.config import ROOT, load_config, merge
from mars_ipp.envs.actions import ACTIONS, legal_neighbors
from mars_ipp.envs.observations import SceneObservationScaler, make_scaler
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.gp.estimator import OnlineGP
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.evaluation.evaluator import env_config, evaluate
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.agents.baselines.coverage import Coverage
from mars_ipp.training.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint


def new_config():
    cfg = load_config(ROOT/'configs/main/bnorm_no_mh_uncertain.yaml')
    cfg['ppo']['device'] = 'cpu'
    return cfg


@pytest.mark.parametrize('scene', ['zhurong', 'perseverance'])
def test_real_scene_71_samples_numerics(scene):
    cfg = new_config()
    field = prepare_fields(cfg)[scene]
    gp = OnlineGP(field.parameters, 1e-6, True)
    points = np.array([(x*2., y*2.) for x in range(16) for y in range(16)])[:71]
    values = field.truth(points)
    prior = gp.predict(points)[1]
    assert np.all(prior == field.parameters['variance'])
    for point, value in zip(points, values):
        assert gp.add(point, value)
        assert not gp.add(point, value + 100)
        np.linalg.cholesky(gp.kernel.K(gp.X) + np.eye(len(gp.X))*gp.noise)
        mean, variance = gp.predict(points)
        assert np.isfinite(mean).all() and (variance >= 0).all() and (variance <= prior).all()
    assert len(gp.X) == 71 and gp.noise == 1e-6
    assert float(gp.model.kern.variance[0]) == field.parameters['variance']
    assert float(gp.model.kern.lengthscale[0]) == field.parameters['lengthscale']
    assert abs(gp.jitter) < 1e-8 * field.parameters['variance']
    np.testing.assert_allclose(gp.model.predict(points)[1] - gp.predict(points)[1], 1e-6, atol=1e-10)


def test_scene_extrema_and_missing_signs():
    class Field:
        def __init__(self, values): self.values = values
        def truth(self, points): return np.tile(self.values, (len(points), 1))
    fields = {'zhurong': Field([2, -4, 0]), 'perseverance': Field([8, -16, 0])}
    scaler = SceneObservationScaler.from_fields(fields, 16, 2, 'zhurong')
    raw = np.array([15, 15, 2, -4, 0] + [0.5]*8)
    np.testing.assert_allclose(scaler.transform(raw), [1,1,1,-1,0] + [.5]*8)
    np.testing.assert_allclose(scaler.for_scene('perseverance').transform(raw)[2:5], [.25,-.25,0])
    assert scaler.scenes['zhurong']['negative'][0] == 1
    assert scaler.scenes['zhurong']['positive'][1] == 1
    assert scaler.scenes['zhurong']['positive'][2] == 1


@pytest.mark.parametrize('start', [[0,0], [0,15], [15,0], [15,15], [8,8], [0,8]])
def test_action_features_and_eight_neighbor_reward(small_field, start):
    cfg = new_config()
    scaler = make_scaler({'zhurong':small_field, 'perseverance':small_field}, cfg)
    env = GridEnv(small_field, scaler, env_config(cfg), metrics_mode='terminal')
    obs, info = env.reset(options={'start':start})
    assert obs.shape == (13,) and 'rmse' not in info
    actions, positions = legal_neighbors(start, 16)
    ratios = np.zeros(8)
    ratios[actions] = env.gp.predict(positions*2)[1][:,0] / small_field.parameters['variance']
    np.testing.assert_allclose(obs[5:], ratios)
    action = int(actions[0]); next_position = np.array(start) + ACTIONS[action]
    neighbors = legal_neighbors(next_position,16)[1]*2
    before = env.gp.predict(neighbors)[1].sum()
    old_obs = obs.copy()
    obs, reward, _, _, info = env.step(action)
    after = env.gp.predict(neighbors)[1].sum()
    assert info['r_tr'] == pytest.approx((before-after)/(before+1e-12))
    assert reward == pytest.approx(info['r_tr']/3) and 'r_mh' not in info
    np.testing.assert_array_equal(old_obs, np.r_[start, small_field.truth([np.array(start)*2])[0], ratios])
    reverse = int(np.flatnonzero(np.all(ACTIONS == -ACTIONS[action], axis=1))[0])
    obs, reward, _, _, info = env.step(reverse)
    assert info['r_tr'] == 0 and reward == pytest.approx(-1/3)
    assert not info['sampled'] and info['samples'] == info['unique_positions'] == 2
    assert info['repeated_visits'] == 1
    if len(actions) < 8:
        outside = next(a for a in range(8) if a not in actions)
        x, y = env.gp.X.copy(), env.gp.Y.copy()
        _, reward, _, _, info = env.step(outside)
        assert reward == pytest.approx(-1/3) and info['outside_actions'] == 1 and not info['sampled']
        np.testing.assert_array_equal(env.gp.X, x); np.testing.assert_array_equal(env.gp.Y, y)


def test_cover_all_starts_and_original_diagonal_rule():
    for x in range(16):
        for y in range(16):
            policy = Coverage(random_start=True)
            position = np.array([x,y]); path = [position.copy()]
            for _ in range(70):
                action = policy.select_action(position)
                position = position + ACTIONS[action]
                assert ((position >= 0) & (position < 16)).all(), (x,y,path)
                path.append(position.copy())
            assert len(path) == 71
    policy = Coverage(random_start=True)
    position = np.array([14,14]); path = [position.copy()]
    for _ in range(70):
        position = position + ACTIONS[policy.select_action(position)]
        path.append(position.copy())
    np.testing.assert_array_equal(path, Coverage().path())


def test_new_policy_update_checkpoint_and_scene_isolation(small_field, tmp_path):
    cfg = new_config(); cfg['evaluation']['seeds'] = [1000,1001]
    fields = {'zhurong':small_field,'perseverance':small_field}
    scaler = make_scaler(fields,cfg)
    agent = PPO(cfg['ppo'],scaler)
    env = GridEnv(small_field,scaler,env_config(cfg))
    observation,_ = env.reset(seed=0)
    action,logprob,value,state = agent.act(observation)
    _,reward,_,_,_ = env.step(action)
    agent.buffer.add(state,action,logprob,value,reward,0,True,True)
    assert state.shape == (13,) and np.isfinite(agent.update()['loss'])
    assert agent.network.actor[0].in_features == agent.network.critic[0].in_features == 13
    checkpoint = tmp_path/'new.pt'
    save_checkpoint(checkpoint,agent,cfg,{'episode':40,'completed':True})
    payload = read_checkpoint(checkpoint)
    restored = PPO(cfg['ppo'],scaler); restore_checkpoint(payload,restored)
    bad = deepcopy(payload); bad['scaler']['scenes']['perseverance']['positive'][0] += 1
    with pytest.raises(ValueError,match='normalization'): restore_checkpoint(bad,restored)
    wrong = merge(cfg,{'ppo':{'observation_mode':'full'}})
    with pytest.raises(ValueError): restore_checkpoint(payload,PPO(wrong['ppo'],scaler))
    rows, _, _ = evaluate(small_field,scaler,cfg,agent,metrics_mode='terminal')
    assert len(rows) == 2 and agent.scaler.scene == 'zhurong'
    cover_rows, trajectories, _ = evaluate(small_field,scaler,cfg,algorithm='cover')
    assert len(cover_rows) == 144 and len(trajectories) == 140
    for seed in cfg['evaluation']['seeds']:
        ppo_row = next(r for r in rows if r['seed']==seed)
        cover_row = next(r for r in cover_rows if r['seed']==seed)
        assert (ppo_row['start_i'],ppo_row['start_j']) == (cover_row['start_i'],cover_row['start_j'])


def test_msr_reuse_overrides_without_policy_change(monkeypatch):
    import importlib
    import sys
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    cli = importlib.import_module('run_comparison')
    base = load_config(ROOT/'configs/comparison/bnorm_baselines.yaml')
    base['field']['test_scenario'] = 'zhurong'
    checkpoint = ROOT/'results/main/msr/seed_0/checkpoint.pt'
    old = read_checkpoint(checkpoint)
    effective = cli.checkpoint_config(checkpoint,base,'msr')
    assert effective['env']['online_noise_variance'] == 1e-6 and effective['env']['unique_samples']
    assert effective['ppo'] == old['config']['ppo'] and effective['reward'] == old['config']['reward']
    assert effective['field']['test_scenario'] == 'zhurong'
    assert read_checkpoint(checkpoint)['config'] == old['config']


def test_variance_corruption_is_rejected(small_field, monkeypatch):
    gp = OnlineGP(small_field.parameters, 1e-6, True)
    gp.add([0,0], [1,2,3])
    for value in (-.01, 3., np.nan):
        monkeypatch.setattr(gp.model, 'predict_noiseless', lambda p, v=value: (np.zeros((len(p),3)), np.full((len(p),1),v)))
        with pytest.raises(FloatingPointError): gp.predict([[1,1]])


def test_random_cover_statistics_and_actual_start_validation(tmp_path):
    import yaml
    from test_comparison import saved_runs
    from mars_ipp.visualization.learning_curves import read_csv
    from mars_ipp.utils.logger import write_csv
    from mars_ipp.evaluation.statistics import load_runs, summarize_runs
    directories = saved_runs(tmp_path)
    for directory in directories:
        cfg = yaml.safe_load((directory/'config.yaml').read_text())
        cfg['evaluation']['cover_random_start'] = True
        cfg['field']['test_scenario'] = 'zhurong'
        (directory/'config.yaml').write_text(yaml.safe_dump(cfg))
        rows = read_csv(directory/'eval_metrics.csv')
        if directory.name == 'cover':
            rows += [{**r, 'seed':1001, 'rmse':20} for r in rows]
            paths = read_csv(directory/'trajectories.csv')
            paths += [{**r, 'seed':1001} for r in paths]
            write_csv(directory/'trajectories.csv', paths)
        for row in rows:
            row.update(start_i=int(row['seed'])-1000, start_j=2)
        write_csv(directory/'eval_metrics.csv',rows)
    runs = load_runs(directories)
    table = summarize_runs(runs,tmp_path/'summary')
    assert table[-1]['runs'] == 2 and isinstance(table[-1]['rmse_std'],float)
    assert table[-1]['rmse_std'] > 0
    cover = directories[-1]; rows = read_csv(cover/'eval_metrics.csv')
    next(r for r in rows if r['stage']=='reset')['start_i'] = 15
    write_csv(cover/'eval_metrics.csv',rows)
    with pytest.raises(ValueError,match='actual initial positions'): load_runs(directories)
