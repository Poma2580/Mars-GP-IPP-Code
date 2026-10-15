"""Run only the approved seed-0, 2000-episode native/SB3 comparison."""
import _bootstrap
import argparse
from copy import deepcopy
from pathlib import Path
import json
import shutil
import time
import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits
from stable_baselines3 import PPO as SB3PPO
from evaluate import run as evaluate_native
from plot_trajectories import run as trajectories
from run_v3_gp_trace import (sha256, seal_artifacts, validate_artifacts, validate_evaluation,
                             validate_paired_starts, preflight)
from mars_ipp.utils.config import ROOT
from mars_ipp.utils.logger import write_json, write_csv
from mars_ipp.utils.seed import seed_all
from mars_ipp.training.trainer import train
from mars_ipp.training.sb3_comparison import train_sb3, SB3Inference, sample_sb3
from mars_ipp.training.checkpoint import read_checkpoint, restore_checkpoint
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.envs.observations import make_scaler
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.evaluation.evaluator import evaluate, env_config
from mars_ipp.visualization.learning_curves import read_csv, save_figure
from mars_ipp.visualization.comparison import plot_results, plot_comparison

METHODS = ('gp_trace','native','sb3')
METRICS = ('rmse','trr','unique_positions','repeated_visits','outside_actions')


def config(smoke=False):
    c = yaml.safe_load((ROOT/'configs/main/v3_gp_trace_train1.yaml').read_text())
    c.update(seed=0,seeds=[0],experiment='v3_sb3_short_train1'+('_smoke' if smoke else ''))
    c['training'].update(episodes=4 if smoke else 2000,eval_every=2 if smoke else 200,
                         checkpoint_every=2 if smoke else 200,monitor_seeds=[1000] if smoke else list(range(1000,1005)))
    c['ppo'].update(rollout_steps=140 if smoke else 2000,update_epochs=2 if smoke else 10,readonly_diagnostics=True)
    c['evaluation']['seeds']=[1000] if smoke else list(range(1000,1050))
    c['evaluation']['record_episode_reward']=True
    c['sb3']={'batch_size':70 if smoke else 250,'learning_rate':3e-4,'max_grad_norm':.5,
              'orthogonal_init':True,'clip_range_vf':None,'target_kl':None}
    return c


def protect():
    manifest=json.loads((ROOT/'docs/sb3_short_protected_artifacts.json').read_text())
    for name,digest in manifest.items():
        if sha256(ROOT/name)!=digest: raise ValueError('Historical artifact changed: '+name)
    return len(manifest)


def roots(c, tag):
    name=c['experiment']+('_'+tag if tag else '')
    return ROOT/'results/main'/name,ROOT/'results/comparison'/name


def ensure_train(c, root, allow_train=True):
    for method in ('native','sb3'):
        d=root/method
        if not d.exists():
            if not allow_train: raise FileNotFoundError('Train both methods before evaluation: '+str(d))
            print('Training '+method,flush=True)
            (train if method=='native' else train_sb3)(c,d)
            plot_results(d,formats=('png',))
            seal_artifacts(d)
        validate_artifacts(d)
        if yaml.safe_load((d/'config.yaml').read_text())!=c: raise ValueError('Training config mismatch')
        s=json.loads((d/'run_status.json').read_text())
        if not s['completed'] or s['steps']!=c['training']['episodes']*70 or s['pending_rollout']:
            raise ValueError('Incomplete training; use a new --run-tag')
        rows=read_csv(d/'train_metrics.csv')
        if [int(r['episode']) for r in rows]!=list(range(1,c['training']['episodes']+1)):
            raise ValueError('Training episode log is incomplete')
        updates=read_csv(d/'updates.csv')
        expected=s['steps']//c['ppo']['rollout_steps']
        if len(updates)!=expected or int(updates[-1]['steps'])!=s['steps']:
            raise ValueError('Missing final rollout update')
        if method=='sb3':
            final=SB3PPO.load(d/'checkpoint.zip',device=c['ppo']['device'])
            if final.num_timesteps!=s['steps'] or final._n_updates!=expected*c['ppo']['update_epochs']:
                raise ValueError('SB3 final optimization is missing')
        else:
            payload=read_checkpoint(d/'checkpoint.pt')
            if payload['progress']!={'episode':c['training']['episodes'],'steps':s['steps'],'completed':True}:
                raise ValueError('Native final checkpoint progress mismatch')


def load_agent(c,root,method):
    fields=prepare_fields(c); scaler=make_scaler(fields,c)
    if method=='sb3':
        meta=json.loads((root/method/'metadata.json').read_text())
        if meta['scaler']!=scaler.to_dict() or meta['config']!=c: raise ValueError('SB3 metadata mismatch')
        agent=SB3Inference(SB3PPO.load(root/method/'checkpoint.zip',device=c['ppo']['device']),scaler)
    else:
        agent=PPO(c['ppo'],scaler); restore_checkpoint(read_checkpoint(root/method/'checkpoint.pt'),agent)
    return fields['train1'],scaler,agent


def ensure_eval(c,root,out):
    out.mkdir(parents=True,exist_ok=True)
    baseline=ROOT/'results/comparison/v3_gp_trace_train1/gp_trace'
    old=yaml.safe_load((baseline/'config.yaml').read_text())
    validate_evaluation(baseline,old,None,'gp_trace')
    for key in ('env','reward','field','v3'):
        if old[key]!=c[key]: raise ValueError('Baseline protocol mismatch: '+key)
    if old['evaluation']['seeds']!=list(range(1000,1050)): raise ValueError('Baseline seeds mismatch')
    if not (out/'gp_trace').exists():
        shutil.copytree(baseline,out/'gp_trace')
        write_json(out/'baseline_reuse.json',{'source':str(baseline),'sha256':sha256(baseline/'eval_metrics.csv')})
    validate_artifacts(out/'gp_trace')
    for method in ('native','sb3'):
        d=out/method; checkpoint=root/method/('checkpoint.pt' if method=='native' else 'checkpoint.zip')
        if not d.exists():
            print('Evaluating '+method,flush=True)
            if method=='native':
                evaluate_native(c,d,checkpoint,formats=('png',))
            else:
                field,scaler,agent=load_agent(c,root,method)
                d.mkdir()
                rows,paths,snapshot=evaluate(field,scaler,c,agent,episode=c['training']['episodes'],action_sampler=sample_sb3)
                write_csv(d/'eval_metrics.csv',rows); write_csv(d/'trajectories.csv',paths)
                np.savez_compressed(d/'field_snapshot.npz',**snapshot)
                (d/'config.yaml').write_text(yaml.safe_dump(c))
                write_json(d/'metadata.json',{'algorithm':'ppo','checkpoint_sha256':sha256(checkpoint),
                    'fields':{'train1':{'sha256':field.sha256}},'scaler':scaler.to_dict(), 'config':c})
                plot_results(d,formats=('png',))
            seal_artifacts(d)
        validate_evaluation(d,c,checkpoint,'ppo')
    for seed in c['evaluation']['seeds']:
        starts=[]
        for method in METHODS:
            rows=[r for r in read_csv(out/method/'eval_metrics.csv') if int(r['seed'])==seed and r['stage']=='reset']
            if len(rows)!=1: raise ValueError('Missing/duplicate reset')
            starts.append((int(rows[0]['start_i']),int(rows[0]['start_j'])))
        expected=tuple(np.random.default_rng(seed).integers(0,c['env']['grid_size'],size=2))
        if any(start!=expected for start in starts): raise ValueError('Paired start mismatch')


def diagnose(c,root,out):
    for method in ('native','sb3'):
        field,scaler,agent=load_agent(c,root,method)
        paths=read_csv(out/method/'trajectories.csv'); details=[]; episodes=[]
        for seed in c['evaluation']['seeds']:
            env=GridEnv(field,scaler,env_config(c),metrics_mode='terminal'); obs,_=env.reset(seed=seed)
            group=[]
            for row in [r for r in paths if int(r['seed'])==seed]:
                value=agent.value(obs)
                obs,reward,done,_,info=env.step(int(row['action']))
                if not np.isclose(reward,float(row['reward']),atol=1e-7): raise ValueError('Replay mismatch')
                group.append({'seed':seed,'step':env.count,'value':value,'reward':reward,
                    'max_trace_action':int(np.isclose(info['gp_selected_score'],info['gp_max_score'],rtol=2e-7,atol=1e-12)),
                    'sampled':int(info['sampled'])})
            ret=0.
            for row in reversed(group):
                ret=row['reward']+.99*ret; row['discounted_return']=ret
                row['value_error']=row['value']-ret
                row['maximum_possible_return']=(1-.99**(71-row['step']))/.01
            details.extend(group)
            invalid=int(info['outside_actions'])+int(info['repeated_visits'])
            episodes.append({'seed':seed,'total_reward':sum(r['reward'] for r in group),
                'max_trace_action_fraction':np.mean([r['max_trace_action'] for r in group]),
                'invalid_action_reward_loss':invalid,
                'sampled_action_reward_loss':sum(1-r['reward'] for r in group if r['sampled']),
                'late_value_mean':np.mean([r['value'] for r in group[-10:]]),
                'late_return_mean':np.mean([r['discounted_return'] for r in group[-10:]]),
                'late_upper_bound_violation_fraction':np.mean([r['value']>r['maximum_possible_return'] for r in group[-10:]])})
        write_csv(out/method/'critic_replay.csv',details); write_csv(out/method/'episode_diagnostics.csv',episodes)
        seal_artifacts(out/method)


def plot_report(c,root,out):
    import matplotlib.pyplot as plt
    runs={m:{'rows':read_csv(out/m/'eval_metrics.csv'),'algorithm':'gp_trace' if m=='gp_trace' else 'ppo','config':c} for m in METHODS}
    for run in runs.values():
        run['rows']=[r for r in run['rows'] if int(r['seed']) in c['evaluation']['seeds']]
    summary=[]; endpoints={}
    for method in METHODS:
        endpoints[method]={int(r['seed']):r for r in runs[method]['rows'] if int(r['step'])==70 and int(r['seed']) in c['evaluation']['seeds']}
        row={'method':method,'training_seed':0 if method!='gp_trace' else '', 'tests':len(endpoints[method])}
        for k in METRICS:
            values=[float(r[k]) for r in endpoints[method].values()]
            row[k+'_mean']=np.mean(values); row[k+'_std']=np.std(values,ddof=0)
        summary.append(row)
    write_csv(out/'summary.csv',summary)
    paired=[{'seed':s,**{k:float(endpoints['sb3'][s][k])-float(endpoints['native'][s][k]) for k in METRICS}} for s in c['evaluation']['seeds']]
    write_csv(out/'paired_sb3_minus_native.csv',paired)
    plot_comparison(runs,out/'figures',formats=('png',),methods=METHODS)
    for seed in c['evaluation']['seeds'][:3]: trajectories(out,seed,formats=('png',),methods=METHODS)
    for source,keys in [('train_metrics.csv',('reward','rmse','trr','repeated_visits','outside_actions')),('eval_endpoints.csv',('episode_reward','rmse','trr','trace_current','repeated_visits','outside_actions'))]:
        fig,axes=plt.subplots(len(keys),1,figsize=(9,3*len(keys)))
        for method in ('native','sb3'):
            rows=read_csv(root/method/source)
            for ax,key in zip(axes,keys):
                episodes=sorted({int(r['episode']) for r in rows})
                if source=='train_metrics.csv':
                    episodes=sorted({(int(r['episode'])-1)//100 for r in rows})
                    vals=[np.mean([float(r[key]) for r in rows if (int(r['episode'])-1)//100==e]) for e in episodes]
                    x=[min((e+1)*100,c['training']['episodes']) for e in episodes]
                else:
                    vals=[np.mean([float(r[key]) for r in rows if int(r['episode'])==e]) for e in episodes]; x=episodes
                ax.plot(x,vals,label=method); ax.set_ylabel(key); ax.legend()
        axes[-1].set_xlabel('Episode'); save_figure(fig,out/'figures'/source.replace('.csv',''),('png',))
    fig,axes=plt.subplots(5,1,figsize=(9,14))
    for method in ('native','sb3'):
        rows=read_csv(root/method/'updates.csv')
        for ax,key in zip(axes,('approx_kl','clip_fraction','entropy','value_loss','explained_variance')):
            column='entropy_loss' if method=='sb3' and key=='entropy' else key
            selected=[r for r in rows if r[column]!='']
            values=[float(r[column])*(-1 if column=='entropy_loss' else 1) for r in selected]
            ax.plot([int(r['steps']) for r in selected],values,label=method)
            ax.set_ylabel(key); ax.legend()
    axes[-1].set_xlabel('Environment steps')
    save_figure(fig,out/'figures'/'optimization_diagnostics',('png',))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method in ('native','sb3'):
        rows=read_csv(out/method/'critic_replay.csv')
        steps=range(1,71)
        for key,style in (('value','-'),('discounted_return','--')):
            means=[np.mean([float(r[key]) for r in rows if int(r['step'])==s]) for s in steps]
            axes[0].plot(list(steps),means,style,label=method+' '+key)
        diag=read_csv(out/method/'episode_diagnostics.csv')
        invalid=np.mean([float(r['invalid_action_reward_loss']) for r in diag])
        sampled=np.mean([float(r['sampled_action_reward_loss']) for r in diag])
        axes[1].bar(method,invalid,color='tab:blue',label='Invalid actions' if method=='native' else None)
        axes[1].bar(method,sampled,bottom=invalid,color='tab:orange',label='Sampled non-max actions' if method=='native' else None)
    axes[0].set(xlabel='Step',ylabel='Discounted return / critic prediction'); axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Episode reward loss relative to 70'); axes[1].legend()
    save_figure(fig,out/'figures'/'behavior_diagnostics',('png',))
    lines=['# SB3 / native short comparison','',f"Completed: seed 0, {c['training']['episodes']} episodes each; final checkpoints; {len(paired)} paired tests; std uses ddof=0.",'', '| Method | RMSE | Trr | Unique | Repeat | Outside |','|---|---|---|---|---|---|']
    for r in summary:
        lines.append('| '+r['method']+' | '+' | '.join(f"{r[k+'_mean']:.4f} ± {r[k+'_std']:.4f}" for k in METRICS)+' |')
    a,b=summary[1:]
    improved=b['rmse_mean']<a['rmse_mean'] and b['trr_mean']<a['trr_mean']
    for key in ('rmse','trr'):
        delta=np.array([r[key] for r in paired])
        lines += ['',f"SB3 − native {key}: mean {delta.mean():.6f}, std {delta.std(ddof=0):.6f}; SB3 lower in {int((delta<0).sum())}/{len(delta)} paired tests."]
    lines+=['', '本次短对照改善。' if improved else '本次短对照未同时改善 RMSE 与 Trr；请分别看两项指标。',
        'Only one training seed: this does not establish general algorithm superiority.',
        'Native: actor/critic LR 3e-4/1e-3, full-batch updates, PyTorch default initialization, no gradient clipping, Adam eps=1e-8, population-std advantage normalization.',
        'SB3: both LR 3e-4, minibatches, orthogonal initialization, grad norm 0.5, Adam eps=1e-5, minibatch sample-std advantage normalization.',
        'Both: 64×64 Tanh, gamma=.99, lambda=.9, entropy=.01, value=.5, no value clipping or KL stop. Same environment steps are not the same optimization process.',
        'Diagnostic timing differs: native KL/clip fraction/explained variance are measured after optimization; SB3 uses its stock training logger. Native value loss/entropy come from the last full-batch epoch; SB3 averages minibatches. Interpret these as within-method trends.',
        '11D variance state omits cross-covariance and remaining time; no action mask; repeat/outside actions get zero reward. Trr is residual Trace percentage (lower is better).']
    for method in ('native','sb3'):
        status=json.loads((root/method/'run_status.json').read_text())
        gradients=status['updates']*c['ppo']['update_epochs']*(c['ppo']['rollout_steps']//c['sb3']['batch_size'] if method=='sb3' else 1)
        diag=read_csv(out/method/'episode_diagnostics.csv')
        lines += ['',f"{method}: {status['steps']} steps, {gradients} gradient updates, {status['invocation_seconds']:.1f} seconds."]
        lines += [f"{k}: {np.mean([float(r[k]) for r in diag]):.6f}" for k in diag[0] if k!='seed']
    lines+=['',f'Historical artifacts verified: {protect()} files.',f'Training: {root}',f'Evaluation: {out}',
            'Old native-10000 is background only; it is excluded from this budget comparison.']
    if c['training']['episodes']==2000:
        lines += ['', '## 结果解释',
            '本次替换 SB3 训练流程未解决性能落后问题，不能据此把原因归结为原 PPO 更新代码。固定 rollout 的 GAE/回报目标已与 SB3 核对，原实现只读诊断不改变更新后的权重。',
            '两种方法的末期 critic 预测均明显高于实际剩余折扣回报；这提示需要优先检查有限时域的价值估计与状态信息。当前 11 维状态没有剩余步数，也没有完整 GP 交叉协方差或访问历史。',
            '两种策略仍损失大量奖励于重复/越界和非最大 Trace 的新采样动作。后续宜逐项做状态/critic 与奖励目标对齐的消融，再考虑大范围调参；这些是诊断方向，尚未证明单一因果。',
            '本次已停止训练，未运行其他训练 seed 或更长预算。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    latex=[r'\begin{tabular}{lrrrrr}',r'Method & RMSE & Trr & Unique & Repeat & Outside \\']
    for row in summary:
        label=row['method'].replace('_',r'\_')
        latex.append(label+' & '+' & '.join(f"{row[k+'_mean']:.4f} $\\pm$ {row[k+'_std']:.4f}" for k in METRICS)+r' \\')
    (out/'summary.tex').write_text('\n'.join(latex+[r'\end{tabular}'])+'\n')


def execute(stage,c,tag):
    root,out=roots(c,tag); preflight(c); protect()
    out.mkdir(parents=True,exist_ok=True)
    status_path=out/'pipeline_status.json'
    started=time.monotonic()
    write_json(status_path,{'stage':stage,'state':'running'})
    try:
        if stage in ('train','all'): ensure_train(c,root)
        if stage in ('evaluate','all'):
            ensure_train(c,root,allow_train=False)
            ensure_eval(c,root,out); diagnose(c,root,out)
        if stage in ('plot','all'): plot_report(c,root,out)
        import importlib.metadata
        write_json(out/'provenance.json',{'entrypoint_sha256':sha256(Path(__file__)),
            'requirements_sha256':sha256(ROOT/'requirements-sb3.txt'),
            'versions':{n:importlib.metadata.version(n) for n in ('stable-baselines3','gymnasium','cloudpickle','farama-notifications','torch','numpy','scipy','GPy')},
            'training_metadata':[str(root/m/'metadata.json') for m in ('native','sb3')]})
        write_json(status_path,{'stage':stage,'state':'completed','elapsed_seconds':time.monotonic()-started,'protected_files':protect()})
    except Exception as exc:
        write_json(status_path,{'stage':stage,'state':'error','error':repr(exc),'elapsed_seconds':time.monotonic()-started})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('smoke','train','evaluate','plot','all'),default='all')
    p.add_argument('--run-tag',default=''); p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.run_tag and (not a.run_tag.replace('-','').replace('_','').isalnum()): p.error('Invalid run tag')
    if a.dry_run:
        print(yaml.safe_dump(config(a.stage=='smoke'))); return
    with threadpool_limits(limits=1):
        seed_all(0)
        if a.stage in ('smoke','all'): execute('all',config(True),a.run_tag)
        if a.stage!='smoke': execute(a.stage,config(),a.run_tag)

if __name__=='__main__': main()
