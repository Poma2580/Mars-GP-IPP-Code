"""Audit saved final V3 trajectories; extra variance-greedy evaluation, no training."""
import _bootstrap
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from evaluate import run as evaluate_run
from run_v3_gp_trace import seal_artifacts, sha256, protect_history, validate_artifacts
from mars_ipp.utils.config import ROOT
from mars_ipp.utils.logger import write_csv, write_json
from mars_ipp.visualization.learning_curves import read_csv, save_figure
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.envs.observations import make_scaler
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.envs.actions import ACTIONS
from mars_ipp.evaluation.evaluator import env_config, evaluate
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.training.checkpoint import read_checkpoint, restore_checkpoint
from mars_ipp.gp.uncertainty import expected_trace_reduction
import matplotlib.pyplot as plt


def stats(values):
    values=np.asarray(values,dtype=float)
    return {'mean':float(values.mean()),'std':float(values.std(ddof=0)),
            'min':float(values.min()),'max':float(values.max())}


def analyze(output, seed):
    checkpoint=ROOT/'results/main'/output.name/f'seed_{seed}'/'checkpoint.pt'
    payload=read_checkpoint(checkpoint);cfg=payload['config']
    field=prepare_fields(cfg)['train1'];scaler=make_scaler({'train1':field},cfg)
    agent=PPO(cfg['ppo'],scaler);restore_checkpoint(payload,agent)
    diagnostic=output/'diagnostics';destination=diagnostic/'variance_greedy'
    if not destination.exists():
        evaluate_run(cfg,destination,algorithm='gp_greedy',formats=('png','pdf'))
        seal_artifacts(destination)
    validate_artifacts(destination)
    # Diagnostic only: same checkpoint, unmasked argmax actions, no policy update.
    argmax_directory=diagnostic/'ppo_argmax'
    if not argmax_directory.exists():
        class ArgmaxPPO(PPO):
            @torch.no_grad()
            def act(self, observation, generator=None):
                state=self.prepare_state(observation)
                tensor=torch.as_tensor(state,device=self.device)
                probabilities=self.network.actor(tensor)
                action=int(probabilities.argmax().item())
                return action,float(torch.log(probabilities[action]).item()),float(self.network.critic(tensor).item()),state
        greedy_agent=ArgmaxPPO(cfg['ppo'],scaler);restore_checkpoint(payload,greedy_agent)
        rows,paths,snapshot=evaluate(field,scaler,cfg,greedy_agent,algorithm='ppo')
        write_csv(argmax_directory/'eval_metrics.csv',rows);write_csv(argmax_directory/'trajectories.csv',paths)
        np.savez_compressed(argmax_directory/'field_snapshot.npz',**snapshot)
        import yaml
        (argmax_directory/'config.yaml').write_text(yaml.safe_dump(cfg))
        write_json(argmax_directory/'metadata.json',{'checkpoint_sha256':sha256(checkpoint),
            'training_seed':seed,'eval_policy':'unmasked argmax diagnostic; official evaluation remains stochastic'})
        seal_artifacts(argmax_directory)
    validate_artifacts(argmax_directory)
    methods={'gp_trace':output/'gp_trace',f'ppo_seed_{seed}':output/f'ppo_seed_{seed}',
             'variance_greedy':destination,'ppo_argmax':argmax_directory}
    steps=[];episodes=[];summary={};policy_rows=[]
    for method,directory in methods.items():
        trajectories=read_csv(directory/'trajectories.csv');metrics=read_csv(directory/'eval_metrics.csv')
        for test_seed in sorted({int(r['seed']) for r in trajectories}):
            path=[r for r in trajectories if int(r['seed'])==test_seed]
            by_step={int(r['step']):r for r in metrics if int(r['seed'])==test_seed and r['stage']!='prior'}
            reset=by_step[0];points=[[int(reset['start_i']),int(reset['start_j'])]]
            rewards=[];global_deltas=[]
            # Replay only the five monitor seeds to audit policy probabilities and score ranking.
            replay=test_seed in range(1000,1005)
            if replay:
                env=GridEnv(field,scaler,env_config(cfg),metrics_mode='terminal');obs,_=env.reset(seed=test_seed)
            for row in path:
                k=int(row['step']);action=int(row['action']);info=by_step[k]
                global_delta=float(by_step[k-1]['trace_current'])-float(info['trace_current'])
                best=float(info['gp_max_score']);selected=float(info['gp_selected_score']);reward=float(row['reward'])
                steps.append({'method':method,'test_seed':test_seed,'step':k,'action':action,
                    'sampled':info['sampled']=='True','reward':reward,'selected_local_score':selected,
                    'max_local_score':best,'local_score_ratio':selected/best if best>0 else 0.,
                    'selected_is_trace_max':bool(np.isclose(selected,best,rtol=1e-10,atol=1e-12)),
                    'global_trace_delta':global_delta,'identity_error':float(info['gp_identity_error'])})
                points.append([int(row['i']),int(row['j'])]);rewards.append(reward);global_deltas.append(global_delta)
                if replay:
                    context=env.planning_context();values=expected_trace_reduction(env.gp,context.candidate_positions,context.candidate_positions)
                    ties=np.isclose(values,values.max(),rtol=1e-10,atol=1e-12)
                    variance=env.gp.predict(context.candidate_positions)[1][:,0]
                    with torch.no_grad():
                        probabilities=agent.network.actor(torch.as_tensor(agent.prepare_state(obs),device=agent.device)).cpu().numpy().astype(float)
                    policy_rows.append({'method_state_distribution':method,'test_seed':test_seed,'step':k,
                        'variance_greedy_is_trace_max':bool(ties[int(np.argmax(variance))]),
                        'policy_probability_of_trace_max':float(probabilities[context.actions[ties]].sum()),
                        'policy_entropy':float(-np.sum(probabilities*np.log(np.maximum(probabilities,1e-30)))),
                        'policy_diagonal_probability':float(probabilities[4:].sum())})
                    obs,replay_reward,_,_,_=env.step(action)
                    if tuple(env.position)!=tuple(points[-1]) or not np.isclose(replay_reward,reward,rtol=1e-6,atol=1e-7):
                        raise ValueError('Replay differs from formal trajectory/reward')
            points=np.asarray(points);bbox=points.max(axis=0)-points.min(axis=0);last=by_step[70]
            episodes.append({'method':method,'test_seed':test_seed,'reward_sum':sum(rewards),
                'global_trace_reduction':sum(global_deltas),
                **{k:float(last[k]) for k in ('rmse','trr','unique_positions','repeated_visits','outside_actions')},
                'span_i':int(bbox[0]),'span_j':int(bbox[1]),'rows_used':len(set(points[:,0])),
                'columns_used':len(set(points[:,1]))})
        ss=[r for r in steps if r['method']==method];ee=[r for r in episodes if r['method']==method]
        action_counts=Counter(r['action'] for r in ss)
        summary[method]={k:stats([r[k] for r in ee]) for k in
            ('rmse','trr','reward_sum','global_trace_reduction','unique_positions','repeated_visits',
             'outside_actions','span_i','span_j','rows_used','columns_used')}
        summary[method].update({'action_counts':{str(k):action_counts[k] for k in range(8)},
            'trace_max_action_fraction':float(np.mean([r['selected_is_trace_max'] for r in ss])),
            'mean_local_score_ratio':float(np.mean([r['local_score_ratio'] for r in ss])),
            'max_identity_error':max(r['identity_error'] for r in ss),
            'reward_global_gain_correlation':float(np.corrcoef([r['reward'] for r in ss],[r['global_trace_delta'] for r in ss])[0,1])})
        pp=[r for r in policy_rows if r['method_state_distribution']==method]
        summary[method]['policy_replay_monitor_seeds']={k:stats([r[k] for r in pp]) for k in
            ('variance_greedy_is_trace_max','policy_probability_of_trace_max','policy_entropy','policy_diagonal_probability')}
    write_csv(diagnostic/'step_diagnostics.csv',steps);write_csv(diagnostic/'episode_diagnostics.csv',episodes)
    write_csv(diagnostic/'policy_replay.csv',policy_rows)
    training=read_csv(checkpoint.parent/'train_metrics.csv');updates=read_csv(checkpoint.parent/'updates.csv');blocks=[]
    for start in range(0,len(training),1000):
        rows=training[start:start+1000]
        blocks.append({'episode_start':start+1,'episode_end':start+len(rows),
            **{k+'_mean':float(np.mean([float(r[k]) for r in rows])) for k in
               ('reward','rmse','trr','unique_positions','repeated_visits','outside_actions')}})
    write_csv(diagnostic/'training_blocks.csv',blocks)
    audit={'training_seed':seed,'training_episodes':payload['progress']['episode'],
        'checkpoint_sha256':sha256(checkpoint),'methods':summary,'action_order':ACTIONS.tolist(),
        'update_entropy_first_20':float(np.mean([float(r['entropy']) for r in updates[:20]])),
        'update_entropy_last_20':float(np.mean([float(r['entropy']) for r in updates[-20:]])),
        'scope':'One completed training seed; variance_greedy is extra diagnostic, no training/checkpoint selection',
        'replay_seeds':list(range(1000,1005)),'protected_artifacts':protect_history()}
    write_json(diagnostic/'audit.json',audit)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for index,method in enumerate(methods):
        ss=[r for r in steps if r['method']==method];ee=[r for r in episodes if r['method']==method]
        axes[0].bar(np.arange(8)+(index-1.5)*.18,[sum(r['action']==a for r in ss)/len(ss) for a in range(8)],width=.18,label=method)
        axes[1].scatter([r['reward_sum'] for r in ee],[r['trr'] for r in ee],s=18,alpha=.6,label=method)
        axes[2].plot(range(1,71),[np.mean([r['global_trace_delta'] for r in ss if r['step']==k]) for k in range(1,71)],label=method)
    axes[0].set(xlabel='Action index',ylabel='Frequency');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Episode normalized reward',ylabel='Final Trr (%)')
    axes[2].set(xlabel='Step',ylabel='Global Trace reduction')
    save_figure(fig,diagnostic/'figures/behavior_diagnosis',('png','pdf'))
    lines=[f'# V3 seed {seed} 问题诊断','',
        '只完成一个训练seed（10000回合）；其余正式训练seed取消。最终checkpoint未按测试结果选择。',
        '主对照各50个相同测试seed；额外用原GP-Greedy诊断最大边际方差策略。',
        'GP-Greedy使用的信息可从11维状态得到，但具有合法动作约束；不属于PPO正式结果。',
        'ppo_argmax仅把同一最终checkpoint的测试改为无mask argmax，用来区分随机采样影响；不替代正式stochastic结果。',
        '', '| 方法 | RMSE | Trr (%) | 奖励总和 | 唯一位置 | 重复 | 越界 | Trace最大动作比例 |',
        '|---|---|---|---|---|---|---|---|']
    for method,v in summary.items():
        cells=[method]+[f"{v[k]['mean']:.4f} ± {v[k]['std']:.4f}" for k in
            ('rmse','trr','reward_sum','unique_positions','repeated_visits','outside_actions')]+[f"{v['trace_max_action_fraction']:.2%}"]
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['','实现核查：5个监控seed轨迹/奖励逐步复现；全部正式步的解析与实际下降误差见audit.json。',
        '动作顺序：'+str(ACTIONS.tolist()),
        f"PPO更新熵：前20次平均 {audit['update_entropy_first_20']:.4f}，后20次平均 {audit['update_entropy_last_20']:.4f} nats。",
        '证据：audit.json、step_diagnostics.csv、episode_diagnostics.csv、policy_replay.csv、training_blocks.csv。',
        '单训练seed不能证明所有seed都会失败，边际方差不足以保证完整Trace决策等价。']
    (diagnostic/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/comparison/v3_gp_trace_train1')
    parser.add_argument('--training-seed',type=int,default=0)
    args=parser.parse_args()
    with threadpool_limits(limits=1): analyze(args.output,args.training_seed)
