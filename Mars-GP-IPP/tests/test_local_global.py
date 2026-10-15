from copy import deepcopy
import warnings
from unittest.mock import patch
import numpy as np
import pytest
from mars_ipp.utils.config import ROOT, load_config, merge
from mars_ipp.envs.observations import make_scaler, region_indices, LOCAL_GLOBAL_MODE
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.envs.actions import ACTIONS, legal_neighbors
from mars_ipp.gp.estimator import OnlineGP
from mars_ipp.gp.uncertainty import expected_trace_reduction
from mars_ipp.field.dataset import evaluation_grid
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.evaluation.evaluator import env_config, evaluate
from mars_ipp.training.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from mars_ipp.agents.ppo.ppo import PPO


def cfg(global_only=False):
    name = 'bnorm_no_mh_local_global_3x3_rbf' + ('_no_local' if global_only else '')
    c = load_config(ROOT/'configs/main'/f'{name}.yaml')
    c['ppo']['device'] = 'cpu'
    return c


def environment(field, global_only=False, metrics_mode='terminal'):
    c = cfg(global_only)
    scaler = make_scaler({'zhurong':field, 'perseverance':field}, c)
    return GridEnv(field, scaler, env_config(c), metrics_mode=metrics_mode)


def test_partition_order_and_full_coverage():
    indices = region_indices()
    assert [len(r) for r in indices] == [36,30,30,30,25,25,30,25,25]
    np.testing.assert_array_equal(np.sort(np.concatenate(indices)), np.arange(256))
    assert indices[0][0] == 0 and indices[1][0] == 6*16 and indices[3][0] == 6


@pytest.mark.parametrize('start', [[0,0], [0,15], [15,15], [8,8], [0,8]])
def test_features_and_cache(small_field, start):
    env = environment(small_field)
    obs, info = env.reset(options={'start':start})
    assert obs.shape == (22,) and np.isfinite(obs).all()
    variance = env.gp.predict(env.sampling_points)[1][:,0] / env.gp.kernel.Kdiag(env.sampling_points)
    np.testing.assert_allclose(obs[13:], [variance[r].mean() for r in region_indices()])
    actions, positions = legal_neighbors(start, 16)
    expected = np.zeros(8); expected[actions] = variance[positions[:,0]*16+positions[:,1]]
    np.testing.assert_allclose(obs[5:13], expected)
    state = env.scaler.transform(obs)
    np.testing.assert_allclose(state[:2], np.array(start)/15)
    np.testing.assert_allclose(state[5:], obs[5:], rtol=1e-6)
    with patch.object(env.gp, 'predict', wraps=env.gp.predict) as predict:
        env._observation(); env._info('step', False, dict.fromkeys(env.reward_names,0.))
        assert predict.call_count == 0
        env.step(int(actions[0]))
        assert sum(len(call.args[0]) == 256 for call in predict.call_args_list) == 1
    assert env._variance_cache[0] == env.gp.revision == 2
    env.reset(options={'start':start})
    assert env._variance_cache[0] == env.gp.revision == 1


def test_repeat_and_outside_short_circuit(small_field):
    env=environment(small_field)
    env.reset(options={'start':[0,0]}); env.step(0)
    x,y=env.gp.X.copy(),env.gp.Y.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with patch('mars_ipp.envs.grid_env.expected_trace_reduction', side_effect=AssertionError('global called')), patch('mars_ipp.envs.grid_env.trace', side_effect=AssertionError('trace called')):
            for action in (2,2):
                _,reward,_,_,info=env.step(action)
                assert info['r_local'] == info['r_global'] == 0
                assert reward == -.25 and not info['sampled']
    np.testing.assert_array_equal(env.gp.X,x); np.testing.assert_array_equal(env.gp.Y,y)
    assert info['outside_actions']==1 and info['repeated_visits']==1


@pytest.mark.parametrize('global_only',[False,True])
def test_reward_formula_and_accumulation(small_field, global_only):
    env=environment(small_field,global_only,metrics_mode='full')
    _,info=env.reset(options={'start':[0,0]}); reset_trace=info['trace_current']; total=0.
    for action in [0,1,0,1,0,1,2,3,5,6]:
        next_position=env.position+ACTIONS[action]
        points=legal_neighbors(next_position,16)[1]*env.scale
        before=float(env.gp.predict(points)[1].sum())
        _,reward,_,_,info=env.step(action)
        after=float(env.gp.predict(points)[1].sum())
        if info['sampled']:
            assert info['r_local']==pytest.approx((before-after)/(before+1e-12))
        assert reward==pytest.approx((0 if global_only else .25)*info['r_local']+.25*(info['r_global']+info['r_rep']+info['r_out']))
        total += info['r_global']
    assert total == pytest.approx((reset_trace-info['trace_current'])/env.c_global,rel=2e-7,abs=1e-7)


@pytest.mark.parametrize('scene',['zhurong','perseverance'])
def test_real_scene_analytic_and_state(scene):
    c=cfg(); fields=prepare_fields(c); field=fields[scene]
    gp=OnlineGP(field.parameters,1e-6,True); grid=evaluation_grid()
    for position in ([0.,0.],[30.,30.],[10.,14.],[12.,14.],[2.,2.]):
        before=gp.predict(grid)[1].sum()
        analytic=expected_trace_reduction(gp,np.array([position]),grid)[0]
        gp.add(position,field.truth([position])[0])
        actual=before-gp.predict(grid)[1].sum()
        assert abs(actual-analytic) <= max(1e-7,1e-7*before)
    env=environment(field); obs,_=env.reset(seed=0)
    for _ in range(70):
        actions,_=legal_neighbors(env.position,16)
        obs,_,_,_,_=env.step(int(actions[0]))
        assert np.isfinite(obs).all() and (obs[5:]>=0).all() and (obs[5:]<=1).all()
    reference=expected_trace_reduction(OnlineGP(field.parameters,1e-6,True),env.sampling_points,grid).max()
    assert env.c_global==pytest.approx(reference)


def test_policy_checkpoint_variants(small_field,tmp_path):
    c=cfg(); env=environment(small_field); obs,_=env.reset(seed=0)
    agent=PPO(c['ppo'],env.scaler)
    action,lp,v,state=agent.act(obs); _,r,_,_,_=env.step(action)
    agent.buffer.add(state,action,lp,v,r,0.,True,True)
    assert np.isfinite(agent.update()['loss'])
    path=tmp_path/'v2.pt'; save_checkpoint(path,agent,c,{'episode':40,'completed':True})
    payload=read_checkpoint(path); assert payload['model_version']=='ppo-22d-v2'
    restore_checkpoint(payload,PPO(c['ppo'],env.scaler))
    with pytest.raises(ValueError,match='variant'):
        restore_checkpoint(payload,PPO(cfg(True)['ppo'],env.scaler))
    bad=deepcopy(payload); bad['model_version']='ppo-22d-v1'
    with pytest.raises(ValueError,match='version'): restore_checkpoint(bad,agent)
    v1=load_config(ROOT/'configs/main/bnorm_no_mh_uncertain.yaml'); v1['ppo']['device']='cpu'
    with pytest.raises(ValueError): restore_checkpoint(payload,PPO(v1['ppo'],make_scaler({'zhurong':small_field,'perseverance':small_field},v1)))


def test_actual_v1_checkpoint_compatibility():
    p=read_checkpoint(ROOT/'results/main/bnorm_no_mh_uncertain_rbf/seed_0/checkpoint.pt')
    c=deepcopy(p['config']); c['ppo']['device']='cpu'
    scaler=make_scaler(prepare_fields(c),c); agent=PPO(c['ppo'],scaler)
    restore_checkpoint(p,agent)
    assert agent.input_dim==13
    c['evaluation']['seeds']=[1000]
    rows,_,_=evaluate(prepare_fields(c)['perseverance'],scaler,c,agent,metrics_mode='terminal')
    assert len(rows)==1 and np.isfinite(rows[0]['rmse'])


def test_config_weights(tmp_path):
    assert cfg()['seeds']==cfg(True)['seeds']==[0,1,2]
    assert cfg()['ppo']['update_epochs']==cfg(True)['ppo']['update_epochs']==10
    assert cfg()['training']['monitor_seeds']==list(range(1000,1005))
    path=tmp_path/'bad.yaml'
    path.write_text(f'extends: {ROOT}/configs/main/bnorm_no_mh_local_global_3x3_rbf.yaml\nreward:\n  weights: [0, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333]\n')
    with pytest.raises(ValueError):load_config(path)


def test_training_monitor_seeds_and_final_buffer(small_field,tmp_path,monkeypatch):
    import mars_ipp.training.trainer as trainer
    c=cfg(); c['training'].update(episodes=1,eval_every=1,checkpoint_every=1)
    c['env']['max_steps']=2
    monkeypatch.setattr(trainer,'prepare_fields',lambda _: {'zhurong':small_field,'perseverance':small_field})
    called=[]
    real=trainer.evaluate
    def monitor(field,scaler,config,*args,**kwargs):
        called.append(config['evaluation']['seeds'])
        return real(field,scaler,config,*args,**kwargs)
    monkeypatch.setattr(trainer,'evaluate',monitor)
    trainer.train(c,tmp_path/'train')
    assert called==[list(range(1000,1005))]
    assert c['evaluation']['seeds']==list(range(1000,1050))
    p=read_checkpoint(tmp_path/'train/checkpoint.pt')
    assert p['progress']['completed'] and p['progress']['steps']==2
    assert p['config']['ppo']['update_epochs']==10


def test_formal_checkpoint_rejects_wrong_version_and_incomplete(small_field,tmp_path,monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    cli=importlib.import_module('run_comparison')
    base=load_config(ROOT/'configs/comparison/bnorm_baselines.yaml')
    c=cfg(); fields=prepare_fields(c); agent=PPO(c['ppo'],make_scaler(fields,c))
    path=tmp_path/'policy.pt'
    save_checkpoint(path,agent,c,{'episode':40,'completed':True})
    with pytest.raises(ValueError,match='10000'):cli.checkpoint_config(path,base,'ours')
    save_checkpoint(path,agent,c,{'episode':10000,'completed':True})
    assert cli.checkpoint_config(path,base,'ours')['ppo']['observation_mode']==LOCAL_GLOBAL_MODE
    payload=read_checkpoint(path);payload['model_version']='ppo-22d-v1'
    import torch
    torch.save(payload,path)
    with pytest.raises(ValueError,match='version'):cli.checkpoint_config(path,base,'ours')


def test_streaming_diagnostics_and_population_sd():
    from mars_ipp.training.diagnostics import RewardDiagnostics
    from mars_ipp.evaluation.local_global import endpoint_stats
    d=RewardDiagnostics()
    for value in (0.,.5,1.5):d.add({'sampled':value!=0,'r_local':value,'r_global':value/2})
    result=d.report()
    assert result['all']['local']['std']==pytest.approx(np.std([0,.5,1.5],ddof=0))
    assert result['new_samples']['local']['above_one']==1
    rows=[{'stage':'step','step':70,**{k:float(seed) for k in ('rmse','trr','coverage','unique_positions','repeated_visits','outside_actions')}} for seed in (0,1,2) for _ in range(50)]
    result=endpoint_stats(rows)
    assert result['runs']==150 and result['rmse_mean']==1
    assert result['rmse_std']==pytest.approx(np.std([0,1,2],ddof=0))


def test_six_run_summary_and_duplicate_rejection(tmp_path):
    import json
    import yaml
    from mars_ipp.evaluation.local_global import METHODS, summarize_scene, validate_run
    from mars_ipp.evaluation.statistics import ALGORITHMS, load_runs
    from mars_ipp.utils.logger import write_json, write_csv
    from mars_ipp.visualization.learning_curves import read_csv
    baseline=ROOT/'results/comparison/bnorm_no_mh_uncertain_rbf'
    scene='perseverance'
    source=baseline/scene/'ours'
    base_rows=read_csv(source/'eval_metrics.csv'); paths=read_csv(source/'trajectories.csv')
    metadata=json.loads((source/'metadata.json').read_text())
    for method in METHODS:
        for seed in (0,1,2):
            directory=tmp_path/scene/method/f'seed_{seed}'
            c=cfg(method=='v2_global_only');c['seed']=seed;c['field']['test_scenario']=scene
            directory.mkdir(parents=True)
            (directory/'config.yaml').write_text(yaml.safe_dump(c))
            write_json(directory/'metadata.json',metadata)
            rows=[{**r,'rmse':float(r['rmse'])+seed} for r in base_rows]
            write_csv(directory/'eval_metrics.csv',rows);write_csv(directory/'trajectories.csv',paths)
    table=summarize_scene(tmp_path,baseline,scene,formats=('png',))
    assert [r['runs'] for r in table]==[50,50,50,50,150,150]
    assert table[-1]['rmse_training_seed_std']==pytest.approx(np.std([0,1,2],ddof=0))
    assert len(read_csv(tmp_path/scene/'summary/per_training_seed.csv'))==6
    directory=tmp_path/scene/'v2/seed_0'
    rows=read_csv(directory/'eval_metrics.csv');rows.append(rows[0])
    write_csv(directory/'eval_metrics.csv',rows)
    reference=load_runs([baseline/scene/a for a in ALGORITHMS])[ALGORITHMS[0]]
    with pytest.raises(ValueError,match='duplicate'):validate_run(directory,reference,0)
