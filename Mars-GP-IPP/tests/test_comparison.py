from copy import deepcopy
import numpy as np
import pytest
from mars_ipp.utils.config import load_config, ROOT
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.envs.observations import ObservationScaler
from mars_ipp.envs.actions import legal_neighbors
from mars_ipp.evaluation.evaluator import env_config, evaluate
from mars_ipp.agents.baselines.coverage import Coverage
from mars_ipp.agents.baselines.gp_trace import GPTrace
from mars_ipp.agents.baselines.msr import Msr
from mars_ipp.gp.estimator import OnlineGP
from mars_ipp.training.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from mars_ipp.agents.ppo.ppo import PPO


def test_grid16_geometry_and_scaler(config, small_field):
    assert config['env']['grid_size'] == 16
    scaler = ObservationScaler.from_training_field(small_field, 16, 2)
    assert scaler.coordinate_scale == 15
    env = GridEnv(small_field, scaler, env_config(config))
    assert len(env.evaluation_points) == 961
    points = []
    for x in range(16):
        for y in range(16):
            obs, info = env.reset(options={'start': [x,y]})
            np.testing.assert_allclose(obs[2:], small_field.truth([[x*2,y*2]])[0])
            points.append(env.gp.X[0])
            assert info['trr'] == pytest.approx(info['trace_current']/info['trace_prior']*100)
    assert len(points) == 256
    np.testing.assert_array_equal(np.max(points, axis=0), [30,30])
    for p in ([0,0], [15,15], [0,15], [15,0]):
        assert len(legal_neighbors(p,16)[0]) == 3
    np.testing.assert_allclose(scaler.transform([15,15,*scaler.mean])[:2], [1,1])


def test_cover_full_path_and_single_run(config, small_field):
    policy = Coverage()
    path = policy.path()
    for step, node in ((0,(14,14)), (13,(1,14)), (14,(0,13)), (15,(1,12)), (28,(14,12)), (29,(15,11)), (30,(14,10))):
        np.testing.assert_array_equal(path[step], node)
    assert path.shape == (71,2) and ((path >= 0)&(path <= 15)).all()
    scaler = ObservationScaler.from_training_field(small_field,16,2)
    rows, trajectories, snapshot = evaluate(small_field, scaler, config, algorithm='cover')
    assert len(rows) == 72 and len(trajectories) == 70
    assert rows[-1]['samples'] == 71
    np.testing.assert_array_equal(snapshot['trajectory'], path*2)


def test_local_trace_explicit_update(config, small_field):
    scaler = ObservationScaler.from_training_field(small_field,16,2)
    env = GridEnv(small_field,scaler,env_config(config))
    for start in ([0,0],[8,8],[15,15]):
        env.reset(options={'start': start})
        context = env.planning_context()
        scores = GPTrace().scores(context)
        before = context.gp.predict(context.candidate_positions)[1].sum()
        for i, point in enumerate(context.candidate_positions):
            gp = OnlineGP(small_field.parameters)
            for x,y in zip(context.gp.X, context.gp.Y): gp.add(x,y)
            gp.add(point,[1,2,3])
            actual = before-gp.predict(context.candidate_positions)[1].sum()
            assert scores[i] == pytest.approx(actual, abs=1e-7)


def test_msr_rewards_dimensions_and_checkpoint(small_field, tmp_path):
    cfg = load_config(ROOT/'configs/main/msr.yaml')
    cfg['ppo']['device'] = 'cpu'
    scaler = ObservationScaler.from_training_field(small_field,16,2)
    env = GridEnv(small_field,scaler,env_config(cfg))
    points = np.array([(i*2,j*2) for i in range(16) for j in range(16)])
    assert env.m_max == pytest.approx(np.abs(small_field.truth(points)).sum(axis=1).max())
    obs,_ = env.reset(options={'start':[0,0]})
    _,reward,_,_,info = env.step(3)
    assert reward == pytest.approx(-1/3) and info['samples'] == 1
    obs,reward,_,_,info = env.step(1)
    assert reward == pytest.approx(info['r_intensity']/3) and info['samples'] == 2
    obs,reward,_,_,info = env.step(3)
    assert reward == pytest.approx((info['r_intensity']-1)/3) and info['samples'] == 3
    assert 'r_mh' not in info and 'r_tr' not in info
    agent = Msr(cfg['ppo'],scaler)
    assert agent.network.actor[0].in_features == agent.network.critic[0].in_features == 2
    action,logprob,value,state = agent.act(obs)
    assert state.shape == (2,)
    agent.buffer.add(state,action,logprob,value,reward,0,True,True)
    assert agent.buffer.transitions[0][0].shape == (2,)
    assert np.isfinite(agent.update()['loss'])
    save_checkpoint(tmp_path/'msr.pt',agent,cfg,{'episode':1})
    payload = read_checkpoint(tmp_path/'msr.pt')
    assert payload['input_dim'] == 2 and payload['algorithm'] == 'msr'
    restore_checkpoint(payload,Msr(cfg['ppo'],scaler))
    full = deepcopy(cfg); full['ppo']['observation_mode'] = 'full'
    with pytest.raises(ValueError): restore_checkpoint(payload,PPO(full['ppo'],scaler))


def test_msr_zero_field_and_scene_specific_max(small_field):
    cfg = load_config(ROOT/'configs/main/msr.yaml')
    scaler = ObservationScaler.from_training_field(small_field,16,2)
    class ConstantField:
        parameters = small_field.parameters
        def __init__(self,value): self.value=value
        def truth(self,points): return np.full((len(points),3),self.value)
        def measure(self,point,rng,noise_rate): return np.full(3,self.value)
    for value in (0,2,8):
        env = GridEnv(ConstantField(value),scaler,env_config(cfg))
        assert env.m_max == 3*value
        env.reset(options={'start':[0,0]})
        _,reward,_,_,info = env.step(1)
        assert info['r_intensity'] == (1 if value else 0)
        assert reward == pytest.approx((1 if value else 0)/3)


def saved_runs(tmp_path):
    from mars_ipp.utils.logger import write_csv, write_json
    import yaml
    directories = []
    for name in ('gp_greedy','gp_trace','msr','cover'):
        directory = tmp_path/name; directory.mkdir()
        cfg = load_config(ROOT/'configs/comparison/baselines.yaml')
        cfg['evaluation']['seeds'] = [1000,1001]
        (directory/'config.yaml').write_text(yaml.safe_dump(cfg))
        write_json(directory/'metadata.json', {'fields': {'same': 'model'}})
        rows, trajectories = [], []
        for seed in ([1000] if name == 'cover' else [1000,1001]):
            for stage,step in [('prior',0),('reset',0)]+[('step',i) for i in range(1,71)]:
                trr = 100 if stage == 'prior' else 80-step
                rows.append({'algorithm':name,'seed':seed,'episode':10000 if name=='msr' else 0,
                             'stage':stage,'step':step,'rmse':1000 if stage=='prior' else 10+seed-1000,
                             'trace_prior':100,'trace_current':trr,'trr':trr})
            trajectories.extend({'seed':seed,'step':i,'i':1,'j':1} for i in range(1,71))
        write_csv(directory/'eval_metrics.csv',rows)
        write_csv(directory/'trajectories.csv',trajectories)
        np.savez(directory/'field_snapshot.npz',trajectory=np.ones((71,2))*2)
        directories.append(directory)
    return directories


def test_table_per_algorithm_episode_cover_std_and_prior_exclusion(tmp_path, monkeypatch):
    from mars_ipp.evaluation.statistics import load_runs, summarize_runs
    from mars_ipp.visualization.comparison import plot_comparison
    import mars_ipp.visualization.comparison as module
    import matplotlib.pyplot as plt
    directories = saved_runs(tmp_path)
    runs = load_runs(list(reversed(directories)))
    table = summarize_runs(runs,tmp_path/'table')
    assert len(table) == 4 and table[2]['runs'] == 2
    assert table[0]['rmse_avg'] == 10.5 and table[0]['rmse_std'] == .5
    assert table[0]['trr_avg'] == 10
    assert table[-1]['trr_std'] == '—' and table[-1]['rmse_std'] == '—'
    assert table[-1]['rmse_avg'] == table[-1]['rmse_min'] == table[-1]['rmse_max']
    def checked_save(fig,prefix,formats):
        if prefix.name in ('trr','rmse'):
            ax = fig.axes[0]
            assert len(ax.lines) == 4 and len(ax.collections) == 3
            assert len(ax.lines[0].get_xdata()) == 71
            assert ax.lines[0].get_ydata()[0] == (80 if prefix.name=='trr' else 10.5)
        plt.close(fig)
    monkeypatch.setattr(module,'save_figure',checked_save)
    plot_comparison(runs,tmp_path/'figures')
    with pytest.raises(ValueError,match='Duplicate'): load_runs(directories[:1]*4)
    from mars_ipp.visualization.learning_curves import read_csv
    from mars_ipp.utils.logger import write_csv
    damaged = read_csv(directories[0]/'eval_metrics.csv')
    damaged[-1]['step'] = 69
    write_csv(directories[0]/'eval_metrics.csv', damaged)
    with pytest.raises(ValueError,match='Incomplete'): load_runs(directories)


def test_legacy_grid15_five_dimensional_checkpoint(small_field,tmp_path):
    cfg = load_config(ROOT/'configs/comparison/baselines.yaml')
    cfg['env']['grid_size'] = 15
    cfg['ppo']['device'] = 'cpu'
    scaler = ObservationScaler.from_training_field(small_field,15,2)
    agent = PPO(cfg['ppo'],scaler)
    save_checkpoint(tmp_path/'legacy.pt',agent,cfg,{'episode':40})
    payload = read_checkpoint(tmp_path/'legacy.pt')
    payload.pop('input_dim'); payload.pop('algorithm'); payload['format'] = 1
    restored = PPO(payload['config']['ppo'],scaler)
    restore_checkpoint(payload,restored)
    assert restored.network.actor[0].in_features == 5
    assert restored.scaler.coordinate_scale == 14
    env = GridEnv(small_field,restored.scaler,env_config(payload['config']))
    obs,_ = env.reset(options={'start':[14,14]})
    assert restored.prepare_state(obs)[0] == 1
    assert env.size == 15



def test_optional_ours_saved_results_and_labels(tmp_path, monkeypatch):
    import shutil
    from mars_ipp.evaluation.statistics import load_runs, summarize_runs, LABELS
    from mars_ipp.utils.logger import write_csv
    from mars_ipp.visualization.learning_curves import read_csv
    from mars_ipp.visualization.comparison import plot_comparison
    import mars_ipp.visualization.comparison as plotting
    import matplotlib.pyplot as plt
    directories = saved_runs(tmp_path)
    ours = tmp_path/'ours'; shutil.copytree(directories[0],ours)
    for filename in ('eval_metrics.csv','trajectories.csv'):
        rows = read_csv(ours/filename)
        for row in rows: row['algorithm'] = 'ours'
        write_csv(ours/filename,rows)
    runs = load_runs([*directories,ours])
    table = summarize_runs(runs,tmp_path/'summary')
    assert len(table) == 5 and table[-1]['method'] == 'Ours'
    def check(fig,prefix,formats):
        ax = fig.axes[0]
        assert [line.get_label() for line in ax.lines] == list(LABELS.values())
        assert len(ax.collections) == 4
        plt.close(fig)
    monkeypatch.setattr(plotting,'save_figure',check)
    plot_comparison(runs,tmp_path/'figures')


def test_checkpoint_protocol_and_explicit_ours_interface(config, scaler, small_field, tmp_path, monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    cli = importlib.import_module('run_comparison')
    base = load_config(ROOT/'configs/comparison/baselines.yaml')
    full = deepcopy(base); full['ppo']['device']='cpu'; full['evaluation']['seeds']=[1000]
    agent = PPO(full['ppo'],scaler)
    checkpoint = tmp_path/'ours.pt'
    save_checkpoint(checkpoint,agent,full,{'episode':10000,'completed':True})
    monkeypatch.setattr(cli,'prepare_fields',lambda cfg:{'same':small_field})
    accepted = cli.checkpoint_config(checkpoint,base,'ours')
    assert accepted['evaluation']['seeds'] == list(range(1000,1050))
    assert full['evaluation']['seeds'] == [1000]
    rejected = deepcopy(base); rejected['env']['grid_size']=15
    with pytest.raises(ValueError,match='environment'): cli.checkpoint_config(checkpoint,rejected,'ours')
    rejected = deepcopy(base); rejected['field']['test_scenario']='zhurong'
    with pytest.raises(ValueError,match='scenario'): cli.checkpoint_config(checkpoint,rejected,'ours')
    monkeypatch.setattr(cli,'prepare_fields',lambda cfg:{'same':small_field} if cfg is base else {})
    with pytest.raises(ValueError,match='GP parameters'): cli.checkpoint_config(checkpoint,base,'ours')
    monkeypatch.setattr(cli,'prepare_fields',lambda cfg:{'same':small_field})
    msr = load_config(ROOT/'configs/main/msr.yaml'); msr['ppo']['device']='cpu'
    msr_path=tmp_path/'msr.pt'
    save_checkpoint(msr_path,Msr(msr['ppo'],scaler),msr,{'episode':10000,'completed':True})
    assert cli.checkpoint_config(msr_path,base,'msr')['evaluation']['seeds'] == list(range(1000,1050))
    with pytest.raises(ValueError,match='not Msr'): cli.checkpoint_config(msr_path,base,'ours')
    save_checkpoint(msr_path,Msr(msr['ppo'],scaler),msr,{'episode':40,'completed':True})
    with pytest.raises(ValueError,match='10000'): cli.checkpoint_config(msr_path,base,'msr')


def test_comparison_runner_adds_ours_only_when_requested(tmp_path, monkeypatch):
    import importlib
    import yaml
    from mars_ipp.utils.logger import write_csv,write_json
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    cli=importlib.import_module('run_comparison')
    base=load_config(ROOT/'configs/comparison/baselines.yaml')
    calls=[]
    monkeypatch.setattr(cli,'checkpoint_config',lambda path,cfg,algorithm:cfg)
    def fake_evaluate(cfg,output,checkpoint,algorithm,formats,plot):
        assert not plot
        output.mkdir()
        (output/'config.yaml').write_text(yaml.safe_dump(cfg))
        write_json(output/'metadata.json',{'fields':{'same':'model'}})
        calls.append((algorithm,checkpoint))
        rows=[]; trajectories=[]
        for seed in ([1000] if algorithm=='cover' else cfg['evaluation']['seeds']):
            for stage,step in [('prior',0),('reset',0)]+[('step',i) for i in range(1,71)]:
                rows.append(dict(algorithm=algorithm,seed=seed,episode=10000 if checkpoint else 0,
                                 stage=stage,step=step,rmse=2.,trace_prior=100.,trace_current=100.-step,trr=100.-step))
            trajectories.extend(dict(seed=seed,step=i,i=1,j=1) for i in range(1,71))
        write_csv(output/'eval_metrics.csv',rows); write_csv(output/'trajectories.csv',trajectories)
    monkeypatch.setattr(cli,'evaluate_run',fake_evaluate)
    monkeypatch.setattr(cli,'plot_comparison',lambda *args:None)
    cli.run_comparison(base,tmp_path/'four','msr.pt')
    assert [a for a,_ in calls] == ['gp_greedy','gp_trace','msr','cover']
    calls.clear()
    cli.run_comparison(base,tmp_path/'five','msr.pt','ours.pt')
    assert calls[-1] == ('ours','ours.pt')
    calls.clear()
    (tmp_path/'five/comparison.json').unlink()
    cli.run_comparison(base,tmp_path/'five','unused',stage='summarize')
    cli.run_comparison(base,tmp_path/'five','unused',stage='plot')
    assert calls == []


def test_retired_global_figures_output_is_rejected(tmp_path):
    from mars_ipp.utils.config import validate_output_path
    from mars_ipp.utils.logger import write_json
    from mars_ipp.visualization.comparison import plot_results
    from mars_ipp.visualization.aggregate import aggregate_results
    forbidden=ROOT/'results/figures'
    for path in (forbidden,forbidden/'nested/file.png'):
        with pytest.raises(ValueError,match='retired'): validate_output_path(path)
    with pytest.raises(ValueError,match='retired'): write_json(forbidden/'report.json',{})
    with pytest.raises(ValueError,match='retired'): plot_results(tmp_path,forbidden)
    assert validate_output_path(tmp_path/'experiment/figures') == tmp_path/'experiment/figures'


def test_ours_policy_evaluation_uses_own_checkpoint_mode(config, small_field, scaler):
    cfg=deepcopy(config); cfg['env']['max_steps']=2; cfg['evaluation']['seeds']=[1000]; cfg['ppo']['device']='cpu'
    policy=PPO(cfg['ppo'],scaler)
    rows,trajectories,snapshot=evaluate(small_field,scaler,cfg,policy,algorithm='ours',episode=10000)
    assert len(rows)==4 and len(trajectories)==2
    assert {row['algorithm'] for row in rows} == {'ours'}
    assert rows[-1]['episode']==10000 and policy.network.actor[0].in_features==5
