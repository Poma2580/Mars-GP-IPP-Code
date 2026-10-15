"""Isolated train1 GP-Trace/PPO diagnostic; never select checkpoints by test results."""
import _bootstrap
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import yaml
from threadpoolctl import threadpool_limits
from evaluate import run as evaluate_run
from plot_trajectories import run as plot_trajectories, load_trajectory
from mars_ipp.utils.config import ROOT, load_config, merge
from mars_ipp.utils.logger import write_csv, write_json
from mars_ipp.training.trainer import train
from mars_ipp.training.checkpoint import read_checkpoint
from mars_ipp.visualization.learning_curves import read_csv
from mars_ipp.visualization.comparison import plot_results, plot_comparison
from mars_ipp.field.field_model import FieldModel
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.envs.observations import make_scaler
from mars_ipp.envs.grid_env import GridEnv
from mars_ipp.evaluation.evaluator import env_config

METHODS = ('gp_trace', 'ppo_seed_0', 'ppo_seed_1', 'ppo_seed_2')
METRICS = ('rmse', 'trr', 'unique_positions', 'repeated_visits', 'outside_actions')
CONFIG = ROOT / 'configs/main/v3_gp_trace_train1.yaml'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protect_history():
    source = ROOT / 'docs/v3_protected_artifacts.json'
    if not source.exists():
        source = ROOT / 'docs/v3_non_v3_protected_artifacts.json'
    manifest = json.loads(source.read_text())
    for name, digest in manifest.items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Protected historical artifact changed: {name}')
    return len(manifest)


def preflight(cfg):
    fields = prepare_fields(cfg)
    if set(fields) != {'train1'}:
        raise ValueError('V3 must load only train1')
    field = fields['train1']
    if sha256(cfg['v3']['original_source']) != cfg['v3']['sha256'] or field.sha256 != cfg['v3']['sha256']:
        raise ValueError('train1 provenance mismatch')
    if field.parameters['type'] != 'rbf':
        raise ValueError('train1 must use original RBF kernel')
    original = FieldModel.load(Path(cfg['v3']['original_source']))
    if field.parameters != original.parameters:
        raise ValueError('train1 kernel changed')
    import torch
    if cfg['ppo']['device'] == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; V3 formal config requires CUDA')
        torch.ones(1, device='cuda').sum().item()
    env = GridEnv(field, make_scaler(fields, cfg), env_config(cfg), metrics_mode='terminal')
    obs, _ = env.reset(seed=1000)
    if obs.shape != (11,):
        raise ValueError('Expected 11D state')
    return {'train1_sha256': field.sha256, 'parameters': field.parameters,
            'origin_degrees': [95, 10], 'device': cfg['ppo']['device'],
            'protected_artifacts': protect_history()}


def smoke_config(cfg):
    return merge(cfg, {'training': {'episodes': 40, 'eval_every': 10, 'checkpoint_every': 10}})


def checkpoint_for(cfg, directory):
    path = directory / 'checkpoint.pt'
    payload = read_checkpoint(path)
    if (payload['config'] != cfg or payload.get('model_version') != 'ppo-11d-v3'
            or payload.get('input_dim') != 11 or not payload['progress'].get('completed')
            or payload['progress']['episode'] != cfg['training']['episodes']):
        raise ValueError(f'Checkpoint incomplete/incompatible; use a new --run-tag: {directory}')
    for name in ('train_metrics.csv', 'eval_endpoints.csv', 'updates.csv', 'metadata.json', 'config.yaml', 'diagnostics.json'):
        if not (directory / name).is_file():
            raise ValueError(f'Missing training artifact: {directory / name}')
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata['fields']['train1']['sha256'] != cfg['v3']['sha256']:
        raise ValueError('Checkpoint field provenance mismatch')
    if yaml.safe_load((directory / 'config.yaml').read_text()) != cfg:
        raise ValueError('Saved training config mismatch')
    rows = read_csv(directory / 'train_metrics.csv')
    if [int(r['episode']) for r in rows] != list(range(1, cfg['training']['episodes']+1)):
        raise ValueError('Incomplete training CSV')
    status = json.loads((directory / 'run_status.json').read_text())
    if not status['completed'] or status['pending_rollout']:
        raise ValueError('Unfinished PPO rollout')
    validate_artifacts(directory)
    return path


def seal_artifacts(directory):
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.name != 'artifact_hashes.json')
    write_json(directory / 'artifact_hashes.json', {p.name: sha256(p) for p in files})


def validate_artifacts(directory):
    manifest = directory / 'artifact_hashes.json'
    if not manifest.exists():
        raise ValueError(f'No completion manifest; do not reuse partial run: {directory}')
    for name, digest in json.loads(manifest.read_text()).items():
        if not (directory/name).is_file() or sha256(directory/name) != digest:
            raise ValueError(f'Run artifact changed: {directory/name}')


def ensure_training(cfg, directory):
    if directory.exists():
        return checkpoint_for(cfg, directory)
    train(cfg, directory)
    plot_results(directory, formats=('png', 'pdf'))
    seal_artifacts(directory)
    return checkpoint_for(cfg, directory)


def validate_evaluation(directory, cfg, checkpoint, algorithm):
    validate_artifacts(directory)
    if yaml.safe_load((directory/'config.yaml').read_text()) != cfg:
        raise ValueError(f'Existing evaluation config differs: {directory}')
    meta = json.loads((directory/'metadata.json').read_text())
    if meta['algorithm'] != algorithm or meta['checkpoint_sha256'] != (sha256(checkpoint) if checkpoint else None):
        raise ValueError('Evaluation method/checkpoint differs')
    if meta['fields']['train1']['sha256'] != cfg['v3']['sha256']:
        raise ValueError('Evaluation field differs')
    rows = read_csv(directory/'eval_metrics.csv')
    seeds = cfg['evaluation']['seeds']
    if {int(r['seed']) for r in rows} != set(seeds):
        raise ValueError('Evaluation seed set differs')
    for seed in seeds:
        selected = [r for r in rows if int(r['seed']) == seed]
        if [(r['stage'], int(r['step'])) for r in selected] != [('prior',0), ('reset',0)] + [('step',i) for i in range(1,71)]:
            raise ValueError('Missing/duplicate evaluation stages')
        for r in selected:
            values = [float(r[k]) for k in METRICS]
            if not np.isfinite(values).all() or min(values) < 0:
                raise ValueError('Invalid evaluation metrics')
            if not np.isclose(float(r['trr']), 100*float(r['trace_current'])/float(r['trace_prior'])):
                raise ValueError('Inconsistent Trr')
            reward = float(r['r_gp'])
            if not np.isfinite(reward) or reward < -1e-7 or (cfg['reward']['mode'] == 'gp_trace_relative' and reward > 1 + 1e-6):
                raise ValueError('Invalid GP reward')
        load_trajectory(directory, seed)
    with np.load(directory/'field_snapshot.npz') as snapshot:
        if snapshot['points'].shape != (961,2) or snapshot['mean'].shape != (961,3):
            raise ValueError('Unexpected evaluation snapshot')
    return rows


def ensure_evaluation(cfg, training_root, output, smoke=False, methods=METHODS):
    seeds = [1000] if smoke else list(range(1000,1050))
    for method in methods:
        own = merge(cfg, {'evaluation': {'seeds': seeds}})
        checkpoint = None
        algorithm = 'gp_trace'
        if method != 'gp_trace':
            own['seed'] = int(method[-1])
            checkpoint = checkpoint_for(merge(cfg, {'seed': own['seed']}), training_root/f"seed_{own['seed']}")
            algorithm = 'ppo'
        directory = output/method
        if not directory.exists():
            print(f'Evaluate {method}: {len(seeds)} test seeds', flush=True)
            evaluate_run(own, directory, checkpoint, algorithm, ('png','pdf'), plot=True)
            seal_artifacts(directory)
        validate_evaluation(directory, own, checkpoint, algorithm)
    validate_paired_starts(output, seeds, methods)


def validate_paired_starts(output, seeds, methods=METHODS):
    starts = {}
    for method in methods:
        rows = read_csv(output/method/'eval_metrics.csv')
        starts[method] = {int(r['seed']):(int(r['start_i']),int(r['start_j'])) for r in rows if r['stage']=='reset'}
    expected = {s:tuple(np.random.default_rng(s).integers(0,16,2)) for s in seeds}
    if any(value != expected for value in starts.values()):
        raise ValueError('Methods have different actual start coordinates')
    return starts


def summarize(output, methods=METHODS):
    table=[]; runs={}
    for method in methods:
        directory=output/method
        rows=read_csv(directory/'eval_metrics.csv')
        endpoint=[r for r in rows if r['stage']=='step' and int(r['step'])==70]
        seeds=sorted(int(r['seed']) for r in endpoint)
        if len(seeds)!=len(set(seeds)) or not seeds:
            raise ValueError('Duplicate/missing endpoints')
        entry={'method':method,'training_seed':int(method[-1]) if method.startswith('ppo') else '', 'test_runs':len(endpoint)}
        for key in METRICS:
            values=np.array([float(r[key]) for r in endpoint])
            entry[key+'_mean']=float(values.mean()); entry[key+'_std']=float(values.std(ddof=0))
        table.append(entry)
        runs[method]={'rows':rows,'algorithm':rows[0]['algorithm'],'config':yaml.safe_load((directory/'config.yaml').read_text())}
    summary=output/'summary'
    write_csv(summary/'comparison.csv',table)
    lines=[r'\begin{tabular}{lrrrrr}',r'Method & RMSE (nT) & Trr (\%) & Unique & Repeated & Outside \\',r'\hline']
    for row in table:
        cells=[row['method'].replace('_',r'\_')]+[f"${row[k+'_mean']:.4f} \\pm {row[k+'_std']:.4f}$" for k in METRICS]
        lines.append(' & '.join(cells)+r' \\')
    lines.append(r'\end{tabular}')
    (summary/'comparison.tex').write_text('\n'.join(lines)+'\n')
    plot_comparison(runs, summary/'figures', methods=methods)
    seeds=sorted({int(r['seed']) for r in runs['gp_trace']['rows']})
    validate_paired_starts(output,seeds,methods)
    for seed in (1000,1001,1002):
        if seed in seeds:
            plot_trajectories(output,seed,methods=methods)
    write_json(summary/'validation.json',{'completed':True,'methods':methods,'training_seeds':[int(m[-1]) for m in methods if m.startswith('ppo')],
        'test_seeds':seeds,'std_ddof':0,'checkpoint_selection':'final completed checkpoint only',
        'reward_delta':'raw actual on fixed pre-move legal neighborhood' if runs['gp_trace']['config']['reward']['mode'] == 'gp_trace_local_raw' else 'actual on fixed pre-move legal neighborhood',
        'reward_units':'nT^2' if runs['gp_trace']['config']['reward']['mode'] == 'gp_trace_local_raw' else 'dimensionless',
        'protected_artifacts':protect_history()})
    report=['# V3 GP-Trace train1 实验报告','',f'完成正式评估：每方法 {len(seeds)} 个固定测试 seed。',
        '训练 seeds：'+', '.join(m[-1] for m in methods if m.startswith('ppo'))+'；测试 seed 单独记录。没有按正式测试挑选 checkpoint。',
        'Trr = 当前共享 latent Trace / 先验 Trace × 100；std 使用 ddof=0。',
        'train1 与祝融来源于相同原始数据，不属于独立泛化场景。',
        '11维状态不含交叉协方差；PPO 不启用动作 mask，重复及越界动作奖励为零。',
        '奖励使用移动前合法八邻域上的实际局部 Trace 下降（nT²），不按候选最大下降量归一化；评价集合包含已访问点。',
        '最大评分为已访问点时，重复动作仍为零奖励；并列最大评分遵循原 baseline 的随机选择。',
        '', '| 方法 | RMSE | Trr (%) | 唯一位置 | 重复 | 越界 |','|---|---|---|---|---|---|']
    for row in table:
        report.append('| '+row['method']+' | '+' | '.join(f"{row[k+'_mean']:.4f} ± {row[k+'_std']:.4f}" for k in METRICS)+' |')
    report += ['', 'CSV：`comparison.csv`；可复制 LaTeX：`comparison.tex`；图像：`figures/`。']
    (summary/'report.md').write_text('\n'.join(report)+'\n')
    return table


def smoke(cfg, training_root, output, training_seeds=(0,1,2)):
    smoke_cfg=smoke_config(cfg)
    tick=time.monotonic()
    for seed in training_seeds:
        ensure_training(merge(smoke_cfg,{'seed':seed}),training_root/f'seed_{seed}')
    methods=('gp_trace', *(f'ppo_seed_{seed}' for seed in training_seeds))
    ensure_evaluation(smoke_cfg,training_root,output,smoke=True,methods=methods)
    summarize(output,methods)
    timing=[json.loads((training_root/f'seed_{s}'/'diagnostics.json').read_text()) for s in training_seeds]
    estimated=sum((d['total_seconds']-d['monitor_seconds'])*250+d['monitor_seconds']*12.5 for d in timing)
    report={'completed':True,'training_seeds':list(training_seeds),'episodes_per_training_seed':40,'test_seeds':[1000],
        'smoke_seconds':time.monotonic()-tick,'estimated_training_hours':estimated/3600,
        'note':'40-episode extrapolation, not a promised runtime; no formal tests used for checkpoint selection',
        'training_diagnostics':timing}
    write_json(output/'smoke_report.json',report)
    print(f"Smoke passed; estimated training hours: {estimated/3600:.2f}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('smoke','train','evaluate','plot','all'),default='all')
    parser.add_argument('--training-seeds', nargs='+', type=int, choices=(0,1,2), default=[0], help='Train/evaluate only these training seeds; default 0 per revised protocol')
    parser.add_argument('--run-tag',default='')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.run_tag and not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_tag):
        parser.error('--run-tag must contain only letters, digits, underscore or hyphen')
    selected=sorted(set(args.training_seeds))
    methods=('gp_trace', *(f'ppo_seed_{seed}' for seed in selected))
    cfg=load_config(CONFIG)
    name=cfg['experiment']+('_'+args.run_tag if args.run_tag else '')
    training_root=ROOT/'results/main'/name
    output=ROOT/'results/comparison'/name
    smoke_training=ROOT/'results/main'/(name+'_smoke')
    smoke_output=ROOT/'results/comparison'/(name+'_smoke')
    if args.dry_run:
        print(json.dumps({'stage':args.stage,'training_root':str(training_root),'evaluation_root':str(output),
            'training_seeds':selected,'episodes':10000,'test_seeds':list(range(1000,1050)),
            'runtime':'mars_gp_ipp','resume':'inference only; unfinished training requires new run-tag'},indent=2))
        return
    output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    def status(stage,completed=False,**extra):
        write_json(output/'pipeline_status.json',{'stage':stage,'completed':completed,
            'invocation_seconds':time.monotonic()-started,**extra})
    with threadpool_limits(limits=1):
        try:
            status('preflight'); report=preflight(cfg);write_json(output/'preflight.json',report)
            if args.stage in ('all','smoke'):
                status('smoke');smoke(cfg,smoke_training,smoke_output,selected)
            if args.stage in ('all','train'):
                report_path=smoke_output/'smoke_report.json'
                if not report_path.exists() or not json.loads(report_path.read_text())['completed']:
                    raise ValueError('Run smoke successfully before formal training')
                for seed in selected:
                    status('training',training_seed=seed)
                    print(f'Train V3 seed={seed}',flush=True)
                    ensure_training(merge(cfg,{'seed':seed}),training_root/f'seed_{seed}')
            if args.stage in ('all','evaluate'):
                status('formal_evaluation');ensure_evaluation(cfg,training_root,output,methods=methods)
            if args.stage in ('all','plot'):
                status('plot');summarize(output,methods)
            status('complete' if args.stage=='all' else args.stage+'_complete',True,
                   protected_artifacts=protect_history(), training_seeds=selected)
        except Exception as exc:
            status('failed',error=f'{type(exc).__name__}: {exc}')
            raise


if __name__=='__main__':
    main()
