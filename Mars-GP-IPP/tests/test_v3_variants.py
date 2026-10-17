from copy import deepcopy
from unittest.mock import patch
import numpy as np
import pytest
import torch
from mars_ipp.experiments.v3_variants import VariantEnv, scaler_for, validate_config, VERSIONS
from mars_ipp.experiments.v3_pipeline import scene_report, historical_policy, prepare_data, train_variant
from mars_ipp.utils.config import ROOT, load_config
from mars_ipp.gp.uncertainty import trace
from mars_ipp.envs.actions import ACTIONS
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.training.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from mars_ipp.envs.observations import scaler_from_dict
from mars_ipp.visualization.learning_curves import read_csv


def config(v):
    return load_config(ROOT/f'configs/main/{v}.yaml')


@pytest.mark.parametrize('v', VERSIONS)
def test_numeric_rewards_and_observations(small_field,v):
    c=config(v);env=VariantEnv(small_field,scaler_for(c),c)
    obs,_=env.reset(options={'start':[2,2]})
    assert obs.shape==((14,) if v=='v3_2' else (11,))
    assert scaler_for(c).transform(obs).shape==obs.shape
    if v=='v3_2':
        np.testing.assert_allclose(obs[11:],env.measurement/env.bmax)
    for displacement in ([1,0],[1,1]):
        action=next(i for i,p in enumerate(ACTIONS) if list(p)==displacement)
        before=trace(env.gp.predict(env.evaluation_points)[1]);prior=env.trace_prior
        old=env.measurement.copy()
        _,reward,_,_,info=env.step(action)
        delta=before-trace(env.gp.predict(env.evaluation_points)[1])
        assert info['raw_trace_delta']==pytest.approx(delta,rel=2e-7,abs=1e-5)
        gp=delta if v=='v3_2' else delta/prior
        extra=np.linalg.norm(env.measurement-old)/(np.linalg.norm(displacement)*env.bref) if v=='v3_4' else 0
        assert reward==pytest.approx(gp+extra,rel=2e-7,abs=1e-7)
        assert env.trace_prior==prior
    # Return to an already visited point with no field access or GP update.
    reverse=next(i for i,p in enumerate(ACTIONS) if list(p)==[-1,-1])
    revision=env.gp.revision
    with patch.object(small_field,'measure',side_effect=AssertionError('repeat measurement')):
        _,reward,_,_,info=env.step(reverse)
    assert reward==(-1 if v=='v3_3' else 0)
    assert env.gp.revision==revision and not info['sampled']
    env.reset(options={'start':[0,0]})
    outside=next(i for i,p in enumerate(ACTIONS) if list(p)==[-1,0])
    revision=env.gp.revision
    with patch.object(small_field,'measure',side_effect=AssertionError('outside measurement')):
        assert env.step(outside)[1]==0
    assert env.gp.revision==revision
    np.testing.assert_array_equal(env.position,[0,0])


@pytest.mark.parametrize('v',VERSIONS)
def test_full_episode_identity_and_zero_field(small_field,v):
    c=config(v)
    with patch.object(small_field,'truth',side_effect=lambda p: np.zeros((len(p),3))),patch.object(small_field,'measure',return_value=np.zeros(3)):
        env=VariantEnv(small_field,scaler_for(c),c)
        obs,_=env.reset(seed=1000)
        rng=np.random.default_rng(9);gp=0
        for _ in range(70):
            _,reward,_,_,info=env.step(int(rng.integers(8)));gp+=info['r_gp']
            assert np.isfinite(reward)
            if v=='v3_4': assert info['r_mh']==0
        if v=='v3_2': assert np.all(obs[11:]==0)
        expected=env.trace_reset-info['trace_current']
        assert gp==pytest.approx(expected if v=='v3_2' else expected/env.trace_prior,rel=2e-7,abs=1e-5)


@pytest.mark.parametrize('v',VERSIONS)
def test_checkpoint_roundtrip(v,tmp_path):
    c=config(v);c['ppo']['device']='cpu';agent=PPO(c['ppo'],scaler_for(c))
    path=tmp_path/'checkpoint.pt';save_checkpoint(path,agent,c,{'completed':False})
    p=read_checkpoint(path);other=PPO(c['ppo'],scaler_from_dict(p['scaler']));restore_checkpoint(p,other)
    assert other.input_dim==(14 if v=='v3_2' else 11)
    assert all(torch.equal(a,b) for a,b in zip(agent.network.parameters(),other.network.parameters()))
    bad=deepcopy(p);bad['config']['reward']['weights']=[.5,.5]
    with pytest.raises(ValueError):restore_checkpoint(bad,other)


def test_old_checkpoint_and_data():
    _,p=historical_policy()
    assert p['progress']['completed']
    assert len(prepare_data())==10


def test_rotation_gp_reset_buffer_and_monitor(small_field,tmp_path,monkeypatch):
    import mars_ipp.experiments.v3_pipeline as pipeline
    c=config('v3_3');c['ppo']['device']='cpu'
    c['training'].update(episodes=5,eval_every=5,checkpoint_every=5,monitor_seeds=[1000])
    loaded=[]
    def load(path):
        loaded.append(str(path));return small_field
    monkeypatch.setattr(pipeline.FieldModel,'load',load)
    resets=[];original=VariantEnv.reset
    def reset(self,*args,**kwargs):
        old=getattr(self,'gp',None);out=original(self,*args,**kwargs)
        assert self.gp is not old and len(self.gp.X)==1
        resets.append(self);return out
    monkeypatch.setattr(VariantEnv,'reset',reset)
    batches=[];update=PPO.update
    def checked(self):
        batches.append(len(self.buffer))
        for i, transition in enumerate(self.buffer.transitions):
            assert transition[-2] == transition[-1] == ((i+1) % 70 == 0)
            if transition[-2]:
                assert transition[-3] == 0.
        return update(self)
    monkeypatch.setattr(PPO,'update',checked)
    train_variant(c,tmp_path/'train')
    rows=read_csv(tmp_path/'train/others/train_metrics.csv')
    assert [r['map'] for r in rows]==[f'train{i}' for i in range(1,6)]
    assert batches==[350]  # buffer survives all five maps until final flush
    assert len(resets)==10
    assert len(loaded)==5 and all('train' in p for p in loaded)
    monitors=read_csv(tmp_path/'train/others/eval_endpoints.csv')
    assert {r['map'] for r in monitors}=={f'train{i}' for i in range(1,6)}


def test_budget_reports_population_sd_and_pairing(small_field,tmp_path):
    from mars_ipp.evaluation.evaluator import evaluate
    from mars_ipp.envs.observations import GPTraceScaler
    from mars_ipp.experiments.v3_pipeline import METHODS, validate_scene
    c=load_config(ROOT/'configs/main/v3_1_gp_trace_global_raw_train1.yaml')
    c['evaluation']['seeds']=[1000,1001]
    rows,paths,_=evaluate(small_field,GPTraceScaler(),c,algorithm='gp_trace')
    # A two-run distribution with known population SD, independent of GP accuracy.
    for row in rows:
        row['rmse']=2. if row['seed']==1000 else 4.
    runs={m:deepcopy(rows) for m in METHODS}
    trajectories={m:deepcopy(paths) for m in METHODS}
    summary=scene_report('synthetic',runs,trajectories,tmp_path,[1000,1001],True)
    assert len(summary)==15
    assert {s['step'] for s in summary}=={30,50,70}
    assert all(s['rmse_mean']==3 and s['rmse_std']==1 and s['runs']==2 for s in summary)
    table=(tmp_path/'table/comparison.md').read_text()
    assert table.count('3.0000 ± 1.0000')==15 and '冒烟验收' in table
    bad=deepcopy(runs)
    bad['v3_4'][1]['start_i']+=1
    with pytest.raises(AssertionError):validate_scene(bad,trajectories,[1000,1001])
    bad=deepcopy(runs);bad['v3_3'].pop()
    with pytest.raises(ValueError,match='Incomplete'):validate_scene(bad,trajectories,[1000,1001])
