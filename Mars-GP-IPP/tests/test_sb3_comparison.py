import numpy as np
import torch
import pytest
pytest.importorskip('stable_baselines3')
from gymnasium.utils.env_checker import check_env as gym_check
from stable_baselines3.common.env_checker import check_env as sb3_check
from stable_baselines3.common.buffers import RolloutBuffer
from mars_ipp.envs.observations import GPTraceScaler
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.evaluation.evaluator import env_config
from mars_ipp.agents.ppo.buffer import compute_gae
from mars_ipp.training.sb3_comparison import GymGridEnv, isolated_rng, make_model, SB3Inference
from mars_ipp.utils.config import ROOT
import yaml


def cfg():
    c=yaml.safe_load((ROOT/'configs/main/v3_gp_trace_train1.yaml').read_text())
    c['ppo'].update(device='cpu',rollout_steps=70,update_epochs=2)
    c['sb3']={'batch_size':35}
    return c


def test_adapter_parity_and_checks(small_field):
    c=cfg(); scaler=GPTraceScaler()
    adapter=GymGridEnv(small_field,scaler,c)
    gym_check(adapter,skip_render_check=True); sb3_check(adapter)
    native=GridEnv(small_field,scaler,env_config(c),metrics_mode='terminal')
    a,ai=adapter.reset(seed=1000); b,bi=native.reset(seed=1000)
    np.testing.assert_array_equal(a,scaler.transform(b))
    for action in ([0,1,2,3,4,5,6,7]*9)[:70]:
        a,ar,at,ax,ai=adapter.step(action); b,br,bt,bx,bi=native.step(action)
        np.testing.assert_array_equal(a,scaler.transform(b))
        assert ar==br and at==bt and ax==bx==False
        assert ai['sampled']==bi['sampled'] and ai['unique_positions']==bi['unique_positions']
    assert at


def test_fixed_rollout_gae_matches_sb3():
    import gymnasium as gym
    rewards=np.array([.2,.5,.3,.1]); values=np.array([.1,.2,.4,.3])
    # Episode ends after transition 1; last transition bootstraps from .7.
    nextvalues=np.array([.2,0,.3,.7]); terminal=np.array([False,True,False,False])
    adv,ret=compute_gae(rewards,values,nextvalues,terminal,terminal,.99,.9)
    b=RolloutBuffer(4,gym.spaces.Box(0,1,(11,)),gym.spaces.Discrete(8),device='cpu',gae_lambda=.9,gamma=.99)
    for i in range(4):
        b.add(np.zeros(11),np.array([0]),np.array([rewards[i]]),np.array([i in (0,2)]),torch.tensor([values[i]]),torch.tensor([0.]))
    b.compute_returns_and_advantage(torch.tensor([.7]),np.array([False]))
    np.testing.assert_allclose(b.advantages[:,0],adv,atol=1e-7)
    np.testing.assert_allclose(b.returns[:,0],ret,atol=1e-7)


def test_saved_policy_and_isolated_sampling(small_field,tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    c=cfg(); scaler=GPTraceScaler(); env=DummyVecEnv([lambda:GymGridEnv(small_field,scaler,c)])
    model=make_model(env,c)
    before={k:v.clone() for k,v in model.policy.state_dict().items()}
    model.learn(140)
    assert model._n_updates==4
    assert any(not torch.equal(v,before[k]) for k,v in model.policy.state_dict().items())
    model.save(tmp_path/'model.zip'); restored=PPO.load(tmp_path/'model.zip',device='cpu')
    obs,_=env.envs[0].core.reset(seed=1000)
    p=SB3Inference(model,scaler); q=SB3Inference(restored,scaler)
    assert torch.equal(p.probabilities(obs),q.probabilities(obs))
    state=torch.get_rng_state().clone()
    generator=torch.Generator().manual_seed(12)
    p.sample(obs,generator)
    assert torch.equal(state,torch.get_rng_state())
    with isolated_rng(): torch.rand(100)
    assert torch.equal(state,torch.get_rng_state())


def test_native_readonly_diagnostics_do_not_change_weights():
    from mars_ipp.agents.ppo.ppo import PPO
    c=cfg()['ppo']; scaler=GPTraceScaler()
    torch.manual_seed(7); a=PPO(c,scaler)
    torch.manual_seed(7); b=PPO({**c,'readonly_diagnostics':True},scaler)
    for agent in (a,b):
        for i in range(8):
            state=np.full(11,.1*i,dtype=np.float32)
            with torch.no_grad():
                probs=agent.network.actor(torch.tensor(state))
                value=agent.network.critic(torch.tensor(state)).item()
            agent.buffer.add(state,i,float(torch.log(probs[i])),value,.5,0.,True,True)
    a.update(); report=b.update()
    assert 'approx_kl' in report
    for k,v in a.network.state_dict().items(): assert torch.equal(v,b.network.state_dict()[k])


def test_final_callback_saves_after_last_optimization(small_field,tmp_path,monkeypatch):
    import mars_ipp.training.sb3_comparison as module
    import json
    from stable_baselines3 import PPO
    c=cfg(); c['training'].update(episodes=2,eval_every=1,monitor_seeds=[1000])
    c['evaluation']['record_episode_reward']=True
    monkeypatch.setattr(module,'prepare_fields',lambda config:{'train1':small_field})
    monkeypatch.setattr(module,'make_scaler',lambda fields,config:GPTraceScaler())
    # The fixture is synthetic, so give its metadata explicit local provenance.
    small_field.source='synthetic'; small_field.sha256='synthetic'
    d=tmp_path/'run'; module.train_sb3(c,d)
    status=json.loads((d/'run_status.json').read_text())
    assert status['completed'] and status['steps']==140 and status['gradient_updates']==8
    model=PPO.load(d/'checkpoint.zip',device='cpu')
    assert model._n_updates==4 and model.num_timesteps==140
