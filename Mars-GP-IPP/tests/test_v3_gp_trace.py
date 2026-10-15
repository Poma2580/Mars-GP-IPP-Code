from copy import deepcopy
from unittest.mock import patch
import importlib
import numpy as np
import pytest
from mars_ipp.utils.config import ROOT, load_config, merge
from mars_ipp.envs.observations import make_scaler, GPTraceScaler
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.envs.actions import legal_neighbors, ACTIONS
from mars_ipp.gp.uncertainty import expected_trace_reduction
from mars_ipp.agents.baselines.gp_trace import GPTrace
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.training.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.evaluation.evaluator import env_config, evaluate


def cfg():
    c=load_config(ROOT/'configs/main/v3_gp_trace_train1.yaml')
    c['ppo']['device']='cpu'
    return c


def environment(field, mode='terminal'):
    c=cfg()
    return GridEnv(field,make_scaler({'train1':field},c),env_config(c),metrics_mode=mode)


@pytest.mark.parametrize('start,count',[([0,0],3),([15,15],3),([8,8],8)])
def test_state_fixed_slots_no_field_features(small_field,start,count):
    with patch.object(small_field,'truth',side_effect=AssertionError('scaler must not query truth')):
        scaler=make_scaler({'train1':small_field},cfg())
    env=environment(small_field)
    obs,_=env.reset(options={'start':start})
    actions,positions=legal_neighbors(start,16)
    assert len(actions)==count and obs.shape==(11,)
    points=np.vstack((start,positions))*2
    values=env.gp.predict(points)[1][:,0]/env.gp.kernel.Kdiag(points)
    expected=np.zeros(8);expected[actions]=values[1:]
    np.testing.assert_allclose(obs[3:],expected)
    assert obs[2]==pytest.approx(values[0])
    state=scaler.transform(obs)
    np.testing.assert_allclose(state[:2],np.array(start)/15)
    assert state.shape==(11,) and state.dtype==np.float32
    np.testing.assert_allclose(state[2:],obs[2:],rtol=1e-6)
    assert env.reward_names==('gp',)
    with pytest.raises(ValueError): scaler.transform(np.zeros(13))
    bad=obs.copy();bad[2]=np.nan
    with pytest.raises(ValueError): scaler.transform(bad)


@pytest.mark.parametrize('start',[[0,0],[8,8]])
def test_reward_fixed_pre_move_neighborhood_and_argmax(small_field,start):
    template=environment(small_field);template.reset(options={'start':start})
    context=template.planning_context()
    baseline_scores=GPTrace().scores(context)
    assert expected_trace_reduction(template.gp,context.candidate_positions,context.candidate_positions)==pytest.approx(baseline_scores)
    rewards=[]
    for action,score in zip(context.actions,baseline_scores):
        env=environment(small_field);env.reset(options={'start':start})
        points=context.candidate_positions.copy()
        before=float(env.gp.predict(points)[1].sum())
        with patch('mars_ipp.envs.grid_env.expected_trace_reduction',wraps=expected_trace_reduction) as scoring:
            _,reward,_,_,info=env.step(int(action))
        scoring.assert_called_once()
        np.testing.assert_array_equal(scoring.call_args.args[1],points)
        np.testing.assert_array_equal(scoring.call_args.args[2],points)
        actual=before-float(env.gp.predict(points)[1].sum())
        assert actual==pytest.approx(float(score),rel=2e-7,abs=1e-7*max(1,before))
        assert reward==pytest.approx(actual)
        assert info['gp_selected_score']==pytest.approx(score)
        assert info['gp_max_score']==pytest.approx(baseline_scores.max())
        assert info['r_gp']==reward and info['sampled']
        rewards.append(reward)
    assert int(np.argmax(rewards))==int(np.argmax(baseline_scores))


def test_local_raw_reward_not_capped_and_legacy_relative_kept(small_field):
    raw=environment(small_field);raw.reset(options={'start':[8,8]})
    actions,points=legal_neighbors(raw.position,16)
    scores=expected_trace_reduction(raw.gp,points*raw.scale,points*raw.scale)
    action=int(actions[np.argmax(scores)])
    _,reward,_,_,info=raw.step(action)
    assert reward==pytest.approx(info['gp_actual_delta'])
    assert reward==pytest.approx(scores.max())
    relative_cfg=cfg();relative_cfg['reward']['mode']='gp_trace_relative'
    relative=GridEnv(small_field,raw.scaler,env_config(relative_cfg),metrics_mode='terminal')
    relative.reset(options={'start':[8,8]})
    _,old_reward,_,_,_=relative.step(action)
    assert old_reward==pytest.approx(1.)


def test_repeated_and_outside_no_sampling(small_field):
    env=environment(small_field);env.reset(options={'start':[0,0]})
    env.step(0)
    actions,positions=legal_neighbors(env.position,16)
    assert any(tuple(p)==(0,0) for p in positions)
    x,y=env.gp.X.copy(),env.gp.Y.copy(); revision=env.gp.revision
    with patch.object(small_field,'measure',side_effect=AssertionError('no sampling')):
        for action in (2,2):
            _,reward,_,_,info=env.step(action)
            assert reward==0 and not info['sampled']
    np.testing.assert_array_equal(env.gp.X,x);np.testing.assert_array_equal(env.gp.Y,y)
    assert env.gp.revision==revision and env.count==3
    assert info['repeated_visits']==1 and info['outside_actions']==1
    np.testing.assert_array_equal(env.position,[0,0])


@pytest.mark.parametrize('value',[0.,1e-15])
def test_near_zero_maximum(small_field,value):
    env=environment(small_field);env.reset(options={'start':[0,0]})
    with patch('mars_ipp.envs.grid_env.expected_trace_reduction',return_value=np.full(3,value)),patch.object(env.gp,'predict',side_effect=lambda points: (np.zeros((len(points),3)),np.full((len(points),1),1e-15))):
        _,reward,_,_,info=env.step(0)
    assert reward==0 and info['sampled']


@pytest.mark.parametrize('scores',[[np.nan,1,2],[np.inf,1,2],[-1.,1,2]])
def test_bad_scores_fail_before_update(small_field,scores):
    env=environment(small_field);env.reset(options={'start':[0,0]})
    with patch('mars_ipp.envs.grid_env.expected_trace_reduction',return_value=np.array(scores)),pytest.raises(FloatingPointError):
        env.step(0)
    assert len(env.gp.X)==1


def test_reward_mismatch_is_not_clipped(small_field):
    env=environment(small_field);env.reset(options={'start':[0,0]})
    with patch('mars_ipp.envs.grid_env.expected_trace_reduction',return_value=np.zeros(3)),pytest.raises(FloatingPointError,match='mismatch'):
        env.step(0)


def test_already_sampled_max_and_ties(small_field):
    env=environment(small_field);env.reset(options={'start':[0,0]});env.step(0)
    context=env.planning_context()
    repeated_index=np.flatnonzero(np.all(context.candidate_positions==[0,0],axis=1))[0]
    scores=np.zeros(len(context.actions));scores[repeated_index]=1.
    with patch('mars_ipp.envs.grid_env.expected_trace_reduction',return_value=scores):
        _,reward,_,_,info=env.step(2)
    assert reward==0 and info['gp_max_score']==1 and info['gp_selected_score']==1
    context=env.planning_context(); policy=GPTrace(seed=1)
    with patch.object(policy,'scores',return_value=np.ones(len(context.actions))):
        assert policy.select_action(None,context) in context.actions


def test_v3_checkpoint_and_legacy_restore(small_field,tmp_path):
    c=cfg();env=environment(small_field);obs,_=env.reset(seed=0);agent=PPO(c['ppo'],env.scaler)
    action,lp,v,state=agent.act(obs);_,reward,_,_,_=env.step(action)
    agent.buffer.add(state,action,lp,v,reward,0.,True,True)
    assert np.isfinite(agent.update()['loss'])
    path=tmp_path/'v3.pt';save_checkpoint(path,agent,c,{'episode':40,'completed':True})
    payload=read_checkpoint(path)
    assert payload['model_version']=='ppo-11d-v3' and payload['input_dim']==11
    restored=PPO(c['ppo'],env.scaler);restore_checkpoint(payload,restored)
    assert restored.act(obs,generator=__import__('torch').Generator().manual_seed(7))[0]==agent.act(obs,generator=__import__('torch').Generator().manual_seed(7))[0]
    legacy_relative=deepcopy(payload)
    legacy_relative['config']['reward']['mode']='gp_trace_relative'
    restore_checkpoint(legacy_relative,PPO(c['ppo'],env.scaler))
    for key,value in [('model_version','ppo-11d-v1'),('input_dim',13)]:
        bad=deepcopy(payload);bad[key]=value
        with pytest.raises(ValueError):restore_checkpoint(bad,restored)
    bad=deepcopy(payload);bad['config']['reward']['weights']=[.5]
    with pytest.raises(ValueError):restore_checkpoint(bad,restored)
    bad=deepcopy(payload);bad['scaler']['coordinate_scale']=30
    with pytest.raises(ValueError):restore_checkpoint(bad,restored)
    for relative in ('bnorm_no_mh_uncertain_rbf/seed_0','bnorm_no_mh_local_global_3x3_rbf/seed_0'):
        old=ROOT/'results/main'/relative/'checkpoint.pt'
        if old.exists():
            legacy=read_checkpoint(old);oldc=deepcopy(legacy['config']);oldc['ppo']['device']='cpu'
            restore_checkpoint(legacy,PPO(oldc['ppo'],make_scaler(prepare_fields(oldc),oldc)))


def test_evaluator_measurement_and_same_starts(small_field):
    c=cfg();c['evaluation']['seeds']=[1000];c['env']['max_steps']=3
    scaler=make_scaler({'train1':small_field},c)
    for algorithm,agent in [('gp_trace',None),('ppo',PPO(c['ppo'],scaler))]:
        rows,paths,snapshot=evaluate(small_field,scaler,c,agent,algorithm)
        start=[r for r in rows if r['stage']=='reset'][0]
        np.testing.assert_array_equal([start['start_i'],start['start_j']],np.random.default_rng(1000).integers(0,16,2))
        for row in paths:
            truth=small_field.truth([[row['i']*2,row['j']*2]])[0]
            np.testing.assert_allclose([row[k] for k in ('Br','B2','B3')],truth)
        assert all(set(r)==set(rows[0]) for r in rows)


def test_real_train1_identity_over_70_steps():
    c=cfg();fields=prepare_fields(c)
    assert set(fields)=={'train1'} and fields['train1'].sha256==c['v3']['sha256']
    env=environment(fields['train1']);obs,_=env.reset(seed=1000)
    policy=GPTrace(seed=1000)
    rewards=[]
    for _ in range(70):
        obs,reward,_,_,info=env.step(policy.select_action(obs,env.planning_context()))
        assert obs.shape==(11,) and np.isfinite(reward) and reward>=0
        rewards.append(reward)
    assert max(rewards)>1
    assert env.finished and info['trr']>=0


def test_config_rejects_invalid_v3(tmp_path):
    for body in ('reward:\n  weights: [0.5]\n','ppo:\n  observation_mode: full_uncertainty\n','env:\n  unique_samples: false\n'):
        path=tmp_path/'bad.yaml';path.write_text(f'extends: {ROOT}/configs/main/v3_gp_trace_train1.yaml\n'+body)
        with pytest.raises(ValueError):load_config(path)


def test_train_terminal_metrics_and_logging(small_field,tmp_path,monkeypatch):
    import mars_ipp.training.trainer as trainer
    c=cfg();c['training'].update(episodes=2,eval_every=2,checkpoint_every=2);c['env']['max_steps']=3
    monkeypatch.setattr(trainer,'prepare_fields',lambda _: {'train1':small_field})
    directory=trainer.train(c,tmp_path/'training')
    from mars_ipp.visualization.learning_curves import read_csv
    rows=read_csv(directory/'train_metrics.csv')
    assert len(rows)==2
    assert all(k in rows[0] for k in ('r_gp','mean_reward','unique_positions','repeated_visits','outside_actions'))
    payload=read_checkpoint(directory/'checkpoint.pt')
    assert payload['model_version']=='ppo-11d-v3' and payload['progress']['completed']
    assert len(read_csv(directory/'eval_endpoints.csv'))==5


def test_runner_rejects_partial_or_modified_artifacts(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    cli=importlib.import_module('run_v3_gp_trace')
    directory=tmp_path/'run';directory.mkdir();(directory/'x.csv').write_text('a\n1\n')
    with pytest.raises(ValueError,match='partial'):cli.validate_artifacts(directory)
    cli.seal_artifacts(directory);cli.validate_artifacts(directory)
    (directory/'x.csv').write_text('a\n2\n')
    with pytest.raises(ValueError,match='changed'):cli.validate_artifacts(directory)


def test_selected_evaluation_never_loads_other_training_seeds(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    cli=importlib.import_module('run_v3_gp_trace')
    calls=[]
    def checkpoint(config,directory):
        calls.append(('checkpoint',config['seed'],directory.name))
        return tmp_path/'seed0.pt'
    def evaluate_run(config,directory,checkpoint,algorithm,*args,**kwargs):
        calls.append(('evaluate',algorithm,config['seed']))
        directory.mkdir(parents=True)
    monkeypatch.setattr(cli,'checkpoint_for',checkpoint)
    monkeypatch.setattr(cli,'evaluate_run',evaluate_run)
    monkeypatch.setattr(cli,'seal_artifacts',lambda _: None)
    monkeypatch.setattr(cli,'validate_evaluation',lambda *args: None)
    monkeypatch.setattr(cli,'validate_paired_starts',lambda output,seeds,methods: calls.append(('methods',methods)))
    cli.ensure_evaluation(cfg(),tmp_path/'training',tmp_path/'output',methods=('gp_trace','ppo_seed_0'))
    assert [x for x in calls if x[0]=='checkpoint']==[('checkpoint',0,'seed_0')]
    assert [x[1] for x in calls if x[0]=='evaluate']==['gp_trace','ppo']
    assert calls[-1]==('methods',('gp_trace','ppo_seed_0'))
