"""Repeat-weight isolation, compatibility, reward and saved-data validation."""
from copy import deepcopy
from unittest.mock import patch
import json
import numpy as np
import pytest
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.experiments import v3_rep_weights as rep
from mars_ipp.experiments.v3_variants import VariantEnv, scaler_for, validate_config, WEIGHT_EXPERIMENTS
from mars_ipp.experiments.v3_pipeline import load_policy
from mars_ipp.training.checkpoint import save_checkpoint
from mars_ipp.evaluation.evaluator import evaluate
from mars_ipp.utils.logger import write_csv


@pytest.mark.parametrize('name,weight', rep.WEIGHTS.items())
def test_config_reward_and_checkpoint(name, weight, small_field, tmp_path):
    base, variants = rep.configs(); c = variants[name]
    expected = deepcopy(base); expected['experiment'] = name; expected['reward']['weights'][1] = weight
    assert c == expected and name not in WEIGHT_EXPERIMENTS
    assert base['reward']['weights'] == [1., .014, .02]
    c['ppo']['device'] = 'cpu'; env = VariantEnv(small_field, scaler_for(c), c)
    obs, _ = env.reset(options={'start': [2, 2]}); assert obs.shape == (14,)
    for action in (1, 4, 6):
        revision = env.gp.revision
        _, reward, _, _, info = env.step(action)
        assert reward == pytest.approx(info['r_gp'] + weight*info['r_rep'] + .02*info['r_mh'])
        if action == 6:
            assert reward == -weight and env.gp.revision == revision
            assert info['r_gp'] == info['r_mh'] == 0
    agent = PPO(c['ppo'], scaler_for(c)); checkpoint = tmp_path/'checkpoint.pt'
    save_checkpoint(checkpoint, agent, c, {'episode':30, 'steps':2100, 'completed':True})
    restored, payload = load_policy(checkpoint)
    assert restored.action_mask_enabled and payload['config']['experiment'] == name
    c['evaluation']['seeds'] = [1000, 1001]
    rows, paths, _ = evaluate(small_field, agent.scaler, c, restored)
    rep.audit_method(rep.LABELS[name], rows, paths, [1000, 1001], weight)
    bad = deepcopy(rows); next(r for r in bad if r['stage']=='step')['weighted_r_rep'] += .1
    with pytest.raises((AssertionError, ValueError)):
        rep.audit_method(rep.LABELS[name], bad, paths, [1000, 1001], weight)
    for weights in ([1., .014, .02], [1., weight, .03]):
        bad = deepcopy(c); bad['reward']['weights'] = weights
        with pytest.raises(ValueError):
            validate_config(bad)
    bad = deepcopy(c); bad['experiment'] = 'V3_5_99'
    with pytest.raises(ValueError):
        validate_config(bad)


def test_paths_and_nonempty_protection(tmp_path):
    train, test = rep.directories(tmp_path/'main', tmp_path/'comparison')
    assert train.name == 'v_3.5.1_v3.5.4_train' and test.name == 'v3.5.1_v.3.5.4_test'
    assert rep.directories(tmp_path/'main', tmp_path/'comparison', True)[0] != train
    train.mkdir(parents=True); (train/'protected').touch()
    with pytest.raises(FileExistsError):
        rep.check_outputs([train])
    with pytest.raises(ValueError):
        rep.check_outputs([tmp_path, test])


def test_six_method_saved_csv_report(tmp_path, small_field):
    _, variants = rep.configs(); c = variants['V3_5_1']; c['ppo']['device'] = 'cpu'; c['evaluation']['seeds'] = [1000, 1001]
    agent = PPO(c['ppo'], scaler_for(c)); rows, paths, _ = evaluate(small_field, agent.scaler, c, agent)
    runs = {m: deepcopy(rows) for m in rep.METHODS}; trajectories = {m: deepcopy(paths) for m in rep.METHODS}
    scene = tmp_path/'comparison/test1'
    for method in rep.METHODS:
        source = scene/'others'/method
        write_csv(source/'eval_metrics.csv', runs[method]); write_csv(source/'trajectories.csv', trajectories[method])
        runs[method] = rep.typed_rows(source/'eval_metrics.csv')
    with patch.object(rep, 'load_policy', side_effect=AssertionError('must not load policy')), patch.object(rep.FieldModel, 'load', side_effect=AssertionError('must not load field')):
        summary = rep.report_scene('test1', runs, trajectories, scene, [1000, 1001], True)
    assert len(summary) == 18 and len(list((scene/'figure').glob('*.png'))) == 3
    for item in summary:
        values = [r['rmse'] for r in rows if r['stage']=='step' and r['step']==item['step']]
        assert item['rmse_mean'] == pytest.approx(np.mean(values))
        assert item['rmse_std'] == pytest.approx(np.std(values, ddof=0))
    table = scene/'table/comparison.md'; table.write_text(table.read_text().replace('预算30步表现', '错误', 1))
    with pytest.raises(ValueError):
        rep.audit_summary(scene, runs)


def test_default_preflight_no_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rep, 'preflight', lambda *a: {'checked':True})
    rep.run('preflight', tmp_path/'main', tmp_path/'comparison')
    assert json.loads(capsys.readouterr().out)['checked'] and not list(tmp_path.iterdir())


@pytest.mark.parametrize('name,weight', rep.BATCH_WEIGHTS['5-10'].items())
def test_new_batch_config_rewards_and_checkpoint(name, weight, small_field, tmp_path):
    base, variants = rep.configs('5-10'); c = variants[name]
    expected = deepcopy(base); expected['experiment'] = name; expected['reward']['weights'][1] = weight
    assert c == expected and c['version'] == c['ppo']['variant'] == 'v3_5'
    c['ppo']['device'] = 'cpu'; agent = PPO(c['ppo'], scaler_for(c))
    env = VariantEnv(small_field, agent.scaler, c)
    env.reset(options={'start':[2, 2]})
    for action in (1, 4, 6):
        revision = env.gp.revision
        _, reward, _, _, info = env.step(action)
        assert reward == pytest.approx(info['r_gp'] + weight * info['r_rep'] + .02 * info['r_mh'])
        if action == 6:
            assert reward == -weight and env.gp.revision == revision
            assert info['r_gp'] == info['r_mh'] == 0
    path = tmp_path/'checkpoint.pt'; save_checkpoint(path, agent, c, {'episode':30,'steps':2100,'completed':True})
    restored, payload = load_policy(path)
    assert restored.action_mask_enabled and payload['config']['experiment'] == name
    c['evaluation']['seeds'] = [1000, 1001]
    rows, paths, _ = evaluate(small_field, agent.scaler, c, restored)
    rep.audit_method(name.replace('_','.'), rows, paths, [1000,1001], weight)
    bad=deepcopy(c); bad['reward']['weights'][1]=.0687
    with pytest.raises(ValueError):
        validate_config(bad)


def test_batch_paths_order_legacy_compatibility_and_protection(tmp_path):
    methods=rep.batch_methods('5-10')
    assert methods == ('GP-Trace','V3.5',*(f'V3.5.{i}' for i in range(5,11)))
    assert rep.batch_methods() == rep.METHODS
    assert list(rep.configs()[1]) == list(rep.WEIGHTS)
    assert list(rep.configs('5-10')[1]) == [f'V3_5_{i}' for i in range(5,11)]
    train,test=rep.directories(tmp_path/'main',tmp_path/'comparison',batch='5-10')
    assert train.name == 'v_3.5.5_v3.5.10_train' and test.name == 'v3.5.5_v.3.5.10_test'
    assert rep.directories(tmp_path/'main',tmp_path/'comparison',True,'5-10')[0] != train
    train.mkdir(parents=True); protected=train/'checkpoint.pt'; protected.write_bytes(b'frozen')
    with pytest.raises(FileExistsError):
        rep.check_outputs([train])
    assert protected.read_bytes()==b'frozen'


def test_eight_method_report_layout_ranking_and_csv_only(tmp_path, small_field):
    c=rep.configs('5-10')[1]['V3_5_5']; c['ppo']['device']='cpu'; c['evaluation']['seeds']=[1000,1001]
    agent=PPO(c['ppo'],scaler_for(c)); rows,paths,_=evaluate(small_field,agent.scaler,c,agent)
    methods=rep.batch_methods('5-10')
    runs={m:deepcopy(rows) for m in methods}; trajectories={m:deepcopy(paths) for m in methods}
    # Distinct endpoint Trr values make the ranking independent of method order.
    for index, method in enumerate(methods):
        for row in runs[method]:
            if row['stage']=='step' and row['step']==70:
                row['trr']=80-index
                row['trace_current']=row['trace_prior']*row['trr']/100
    import matplotlib.pyplot as plt
    original=plt.subplots
    layouts=[]
    def record_layout(*args,**kwargs):
        layouts.append(args[:2]); return original(*args,**kwargs)
    scene=tmp_path/'comparison/test1'
    with patch.object(rep,'load_policy',side_effect=AssertionError('CSV only')), patch.object(rep.FieldModel,'load',side_effect=AssertionError('CSV only')), patch.object(plt,'subplots',side_effect=record_layout):
        summary=rep.report_scene('test1',runs,trajectories,scene,[1000,1001],True,methods)
    assert len(summary)==24 and (2,4) in layouts
    assert [r['method'] for r in summary[:8]]==list(methods)
    for i in range(2,6):
        write_csv(tmp_path/f'comparison/test{i}/others/summary.csv',summary)
    rep.summary_report(tmp_path/'comparison',True,methods)
    selection=json.loads((tmp_path/'comparison/others/selection.json').read_text())
    assert selection['best_new']['method']=='V3.5.10'
    assert selection['best_all']['method']=='V3.5.10'
    assert selection['independent_validation'] is False
    assert len(rep.typed_rows(tmp_path/'comparison/others/summary_70.csv'))==48
    assert len(rep.typed_rows(tmp_path/'comparison/others/trr_ranking.csv'))==8
    table=scene/'others/summary.csv'; corrupt=rep.read_csv(table)
    corrupt[0]['trr_mean']=float(corrupt[0]['trr_mean'])+1; write_csv(table,corrupt)
    with pytest.raises(AssertionError):
        rep.audit_summary(scene,runs)


@pytest.mark.parametrize('name,weight', rep.BATCH_WEIGHTS['11-12'].items())
def test_extreme_weights_rewards_finite_update_and_checkpoint(name, weight, small_field, tmp_path):
    import torch
    from mars_ipp.utils.seed import seed_all
    seed_all(0)
    base, variants = rep.configs('11-12'); c = variants[name]
    expected = deepcopy(base); expected['experiment'] = name; expected['reward']['weights'][1] = weight
    assert c == expected
    assert weight == pytest.approx(.006871 * (50 if name == 'V3_5_11' else 100), abs=1e-15)
    c['ppo']['device'] = 'cpu'; c['ppo']['readonly_diagnostics'] = True
    agent = PPO(c['ppo'], scaler_for(c)); env = VariantEnv(small_field, agent.scaler, c)
    env.reset(options={'start':[2, 2]})
    for action in (1, 4, 6):
        revision = env.gp.revision
        _, reward, _, _, info = env.step(action)
        assert reward == pytest.approx(info['r_gp'] + weight*info['r_rep'] + .02*info['r_mh'])
        if action == 6:
            assert reward == -weight and env.gp.revision == revision
            assert info['r_gp'] == info['r_mh'] == 0
    obs,_ = env.reset(options={'start':[0, 0]})
    repeats = 0
    for _ in range(70):
        mask = agent.mask_for(obs)
        action, lp, value, state = agent.act(obs)
        nxt, reward, done, truncated, info = env.step(action)
        assert mask[action] and info['outside_actions'] == 0
        repeats += -info['r_rep']
        agent.buffer.add(state, action, lp, value, reward, 0. if done else agent.value(nxt), done, done or truncated, action_mask=mask)
        obs = nxt
    assert repeats > 0
    report = agent.update()
    assert all(np.isfinite(v) for v in report.values() if v is not None)
    assert all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all()) for p in agent.network.parameters())
    path=tmp_path/'checkpoint.pt'; save_checkpoint(path, agent, c, {'episode':30,'steps':2100,'completed':True})
    restored,payload=load_policy(path)
    assert restored.action_mask_enabled and payload['config']['reward']['weights'][1] == weight
    c['evaluation']['seeds']=[1000,1001]
    rows,paths,_=evaluate(small_field,agent.scaler,c,restored)
    rep.audit_method(name.replace('_','.'),rows,paths,[1000,1001],weight)
    bad=deepcopy(c); bad['reward']['weights'][1]=round(weight,4) if name=='V3_5_11' else .34355
    with pytest.raises(ValueError):
        validate_config(bad)


def test_extreme_batch_paths_and_legacy_compatibility(tmp_path):
    assert rep.batch_methods('11-12') == ('GP-Trace','V3.5','V3.5.11','V3.5.12')
    assert rep.batch_methods() == rep.METHODS
    assert len(rep.configs('5-10')[1]) == 6 and len(rep.configs()[1]) == 4
    train,test=rep.directories(tmp_path/'main',tmp_path/'comparison',batch='11-12')
    assert train.name=='v_3.5.11_v3.5.12_train' and test.name=='v3.5.11_v.3.5.12_test'
    assert rep.directories(tmp_path/'main',tmp_path/'comparison',True,'11-12')[0] != train
    train.mkdir(parents=True); (train/'checkpoint.pt').write_bytes(b'protected')
    with pytest.raises(FileExistsError):
        rep.check_outputs([train])
    assert (train/'checkpoint.pt').read_bytes()==b'protected'


def test_four_method_report_layout_and_ranking(tmp_path, small_field):
    import matplotlib.pyplot as plt
    c=rep.configs('11-12')[1]['V3_5_11']; c['ppo']['device']='cpu'; c['evaluation']['seeds']=[1000,1001]
    agent=PPO(c['ppo'],scaler_for(c)); rows,paths,_=evaluate(small_field,agent.scaler,c,agent)
    methods=rep.batch_methods('11-12'); runs={m:deepcopy(rows) for m in methods}; trajectories={m:deepcopy(paths) for m in methods}
    for index,method in enumerate(methods):
        for row in runs[method]:
            if row['stage']=='step' and row['step']==70:
                row['trr']=80-index; row['trace_current']=row['trace_prior']*row['trr']/100
    original=plt.subplots; layouts=[]
    def record(*args,**kwargs):
        layouts.append(args[:2]); return original(*args,**kwargs)
    scene=tmp_path/'comparison/test1'
    with patch.object(rep,'load_policy',side_effect=AssertionError('CSV only')), patch.object(rep.FieldModel,'load',side_effect=AssertionError('CSV only')), patch.object(plt,'subplots',side_effect=record):
        summary=rep.report_scene('test1',runs,trajectories,scene,[1000,1001],True,methods)
    assert len(summary)==12 and (1,4) in layouts
    assert [r['method'] for r in summary[:4]]==list(methods)
    for i in range(2,6):
        write_csv(tmp_path/f'comparison/test{i}/others/summary.csv',summary)
    rep.summary_report(tmp_path/'comparison',True,methods)
    selection=json.loads((tmp_path/'comparison/others/selection.json').read_text())
    assert selection['best_new']['method']=='V3.5.12' and selection['smoke']
    assert len(rep.typed_rows(tmp_path/'comparison/others/summary_70.csv'))==24
    assert len(rep.typed_rows(tmp_path/'comparison/others/trr_ranking.csv'))==4
    assert '| V3.5.11 | 0.34355 |' in (tmp_path/'comparison/others/weight_comparison.md').read_text()
