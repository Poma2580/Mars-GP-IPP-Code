"""Reproducible multimap training and paired evaluation for independent V3 variants."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import logging
import shutil
import time
import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.envs.observations import scaler_from_dict
from mars_ipp.evaluation.evaluator import evaluate
from mars_ipp.field.field_model import FieldModel
from mars_ipp.training.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from mars_ipp.utils.config import ROOT, load_config
from mars_ipp.utils.logger import append_csv, write_csv, write_json
from mars_ipp.utils.seed import seed_all
from mars_ipp.visualization.learning_curves import read_csv, save_figure, plot_training_rewards
from .v3_variants import VERSIONS, TRAIN_MAPS, VariantEnv, scaler_for, validate_config

LOG = logging.getLogger(__name__)
HISTORY = ROOT / 'results/main/v3_1_gp_trace_global_raw_train1_recovery/seed_0'
SOURCE = ROOT.parent / 'Mars_Env_GP/output/10datasets'
METHODS = ('v3_1', *VERSIONS, 'gp_trace')
METRICS = ('rmse', 'trace_current', 'trr', 'unique_positions', 'repeated_visits', 'outside_actions')
NOTE = ('Known per-map kernel and Bmax priors. Trr=Trace_current/Trace_prior*100%, lower is better. '
        'Population SD ddof=0. V3.1 differs in training maps and gamma; differences cannot be attributed to B alone.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protect_history():
    files = [p for p in (ROOT / 'results').rglob('*') if p.is_file()]
    files += list((ROOT / 'src/mars_ipp/agents/baselines').glob('*.py'))
    files += list((ROOT / 'configs').rglob('*.yaml'))
    return {str(p): sha(p) for p in files}


def verify_hashes(manifest):
    for path, digest in manifest.items():
        if sha(path) != digest:
            raise ValueError(f'Protected input changed: {path}')


def empty_output(directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Refuse to overwrite {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def prepare_data(copy=False):
    manifest = {}
    pairs = [(SOURCE / f'{kind}{i}.pkl', ROOT / f'data/models/{kind}{i}.pkl')
             for kind in ('train', 'test') for i in range(1, 6)]
    # Check every conflict before copying anything.
    for source, target in pairs:
        digest = sha(source)
        if target.exists() and sha(target) != digest:
            raise ValueError(f'Data conflict: {target}')
        manifest[source.stem] = {'source': str(source), 'target': str(target), 'sha256': digest}
    if copy:
        for source, target in pairs:
            if not target.exists():
                with source.open('rb') as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
            if sha(target) != manifest[source.stem]['sha256']:
                raise ValueError(f'Copy hash mismatch: {target}')
    return manifest


def historical_policy():
    path = HISTORY / 'checkpoint.pt'
    payload = read_checkpoint(path)
    c = payload['config']
    reference = load_config(ROOT / 'configs/main/v3_1_gp_trace_global_raw_train1.yaml')
    if (payload['progress'] != {'episode': 10000, 'steps': 700000, 'completed': True}
            or payload.get('input_dim') != 11 or c != reference):
        raise ValueError('Require frozen completed original V3.1 seed-0 configuration')
    hashes_path = HISTORY / 'artifact_hashes.json'
    if hashes_path.exists():
        import json
        # Historical manifest format is retained as provenance; verify checkpoint below when present.
        manifest = json.loads(hashes_path.read_text())
        expected = manifest.get('checkpoint.pt')
        if isinstance(expected, str) and expected != sha(path):
            raise ValueError('V3.1 checkpoint hash mismatch')
    agent = PPO(c['ppo'], scaler_from_dict(payload['scaler']))
    restore_checkpoint(payload, agent)
    return agent, payload


def preflight():
    data = prepare_data()
    _, history = historical_policy()
    configs = {v: load_config(ROOT / f'configs/main/{v}.yaml') for v in VERSIONS}
    return {'data': data, 'configs': configs, 'historical_checkpoint': str(HISTORY / 'checkpoint.pt'),
            'historical_sha256': sha(HISTORY / 'checkpoint.pt'), 'historical_progress': history['progress'],
            'cuda_available': torch.cuda.is_available(), 'note': NOTE}


def training_reward_data(directory):
    """Read only training CSV/config; V3.1 resolves its immutable source record."""
    directory = Path(directory)
    aux = directory / 'others'
    if (aux / 'provenance.json').exists():
        source = Path(json.loads((aux / 'provenance.json').read_text())['source'])
        rows = read_csv(source / 'train_metrics.csv')
        config = yaml.safe_load((source / 'config.yaml').read_text())
    else:
        rows = read_csv(aux / 'train_metrics.csv')
        config = yaml.safe_load((aux / 'config.yaml').read_text())
    return rows, config['reward']['mode']


def plot_version_rewards(directory, output=None, label=None):
    rows, mode = training_reward_data(directory)
    # Keep smoke identification without adding a title to the formal reference layout.
    title = label if label and label.startswith('SMOKE') else None
    plot_training_rewards(rows, Path(output or directory) / 'figures', mode, title=title)


def replot_training_rewards(train_root):
    """Replace only the four reward PNGs; no evaluation CSV or model is needed."""
    directories = [Path(train_root) / version for version in ('v3_1', *VERSIONS)]
    inputs = [training_reward_data(directory) for directory in directories]
    for directory, (rows, mode) in zip(directories, inputs):
        smoke = len(rows) < 10000
        plot_training_rewards(rows, directory / 'figures', mode,
                              title='SMOKE — non-formal results' if smoke else None)
        LOG.info('Redrew reward figure %s', directory / 'figures/training.png')


def plot_training(directory, label, output=None):
    import matplotlib.pyplot as plt
    directory = Path(directory)
    figure_root = Path(output) if output else directory
    plot_version_rewards(directory, output=figure_root, label=label)
    monitor = read_csv(directory / 'others/eval_endpoints.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = sorted({int(r['episode']) for r in monitor})
    for ax, metric in zip(axes, ('rmse', 'trace_current')):
        for scene in TRAIN_MAPS:
            values = [np.mean([float(r[metric]) for r in monitor
                       if int(r['episode']) == ep and r['map'] == scene]) for ep in epochs]
            ax.plot(epochs, values, alpha=.55, label=scene)
        values = [np.mean([float(r[metric]) for r in monitor if int(r['episode']) == ep]) for ep in epochs]
        ax.plot(epochs, values, color='black', linewidth=2, label='equal-map mean')
        ax.set(xlabel='Episode (training maps only)', ylabel=metric, title=label); ax.legend(fontsize=7)
    save_figure(fig, figure_root / 'figures/training_evaluation')


def train_variant(config, output, smoke=False, plotter=None):
    validate_config(config)
    c = deepcopy(config)
    if smoke:
        c['training'].update(episodes=30, checkpoint_every=10, eval_every=10)
    directory = empty_output(output)
    aux = directory / 'others'; aux.mkdir()
    (aux / 'config.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    seed_all(c['seed'])
    fields = {scene: FieldModel.load(ROOT / c['field']['models'][scene]) for scene in TRAIN_MAPS}
    scaler = scaler_for(c)
    agent = PPO(c['ppo'], scaler)
    envs = {scene: VariantEnv(f, scaler, c, 'terminal') for scene, f in fields.items()}
    rng = np.random.default_rng(c['seed'])
    for env in envs.values():
        env.rng = rng
    write_json(aux / 'metadata.json', {'smoke': smoke, 'note': NOTE,
        'fields': {s: {'sha256': f.sha256, 'parameters': f.parameters,
                       'bmax': envs[s].bmax, 'b_ref': envs[s].bref} for s, f in fields.items()},
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'src').rglob('*.py')},
        'monitor_seeds': c['training']['monitor_seeds'], 'scaler': scaler.to_dict()})
    counts = Counter(); steps = 0; start = time.monotonic()
    pending = {'train': [], 'updates': []}
    try:
        for episode in range(1, c['training']['episodes'] + 1):
            scene = TRAIN_MAPS[(episode - 1) % 5]; env = envs[scene]
            obs, reset_info = env.reset(); total = 0.; terms = dict.fromkeys(env.reward_names, 0.)
            prior = reset_info['trace_prior']
            for _ in range(70):
                action, logprob, value, state = agent.act(obs)
                nxt, reward, done, truncated, info = env.step(action)
                if env.weight_experiment:
                    expected = sum(info[f'weighted_r_{name}'] for name in terms)
                    if not np.isclose(reward, expected, rtol=1e-10, atol=1e-12) or info['trace_prior'] != prior:
                        raise AssertionError('Weighted reward or fixed prior identity failed')
                next_value = 0. if done else agent.value(nxt)
                agent.buffer.add(state, action, logprob, value, reward, next_value, done, done or truncated)
                steps += 1; total += reward
                for name in terms:
                    terms[name] += info[f'r_{name}']
                if len(agent.buffer) >= c['ppo']['rollout_steps']:
                    pending['updates'].append({'steps': steps, **agent.update()})
                obs = nxt
            if not np.isfinite([total, *terms.values(), *[info[k] for k in METRICS]]).all():
                raise FloatingPointError('Nonfinite training result')
            counts[scene] += 1
            weighted = ({f'weighted_r_{name}': weight * terms[name]
                         for name, weight in zip(env.reward_names, env.weights)} if env.weight_experiment else {})
            if env.weight_experiment and not np.isclose(total, sum(weighted.values()), rtol=1e-10, atol=1e-12):
                raise AssertionError('Episode weighted reward identity failed')
            pending['train'].append({'episode': episode, 'map': scene, 'steps': steps, 'reward': total,
                **{f'r_{k}': v for k, v in terms.items()}, **{k: info[k] for k in METRICS},
                **weighted,
                'trace_prior': info['trace_prior'], 'raw_trace_sum': env.trace_reset-info['trace_current']})
            completed = episode == c['training']['episodes']
            if completed and len(agent.buffer):
                pending['updates'].append({'steps': steps, **agent.update()})
            if (episode % c['training']['checkpoint_every'] == 0
                    or episode % c['training']['eval_every'] == 0 or completed):
                monitor = []
                for name, field in fields.items():
                    mc = deepcopy(c); mc['field']['test_scenario'] = name
                    mc['evaluation']['seeds'] = c['training']['monitor_seeds']
                    rows, _, _ = evaluate(field, scaler, mc, agent, episode=episode, metrics_mode='terminal')
                    monitor.extend(dict(row, map=name) for row in rows)
                append_csv(aux / 'eval_endpoints.csv', monitor)
                append_csv(aux / 'train_metrics.csv', pending['train'])
                append_csv(aux / 'updates.csv', pending['updates'])
                pending = {'train': [], 'updates': []}
                save_checkpoint(directory / 'checkpoint.pt', agent, c,
                                {'episode': episode, 'steps': steps, 'completed': completed})
                write_json(aux / 'run_status.json', {'completed': completed, 'episode': episode, 'steps': steps,
                    'map_counts': dict(counts), 'updates': agent.updates, 'pending_rollout': len(agent.buffer),
                    'seconds': time.monotonic()-start, 'smoke': smoke})
                LOG.info('%s episode=%d steps=%d counts=%s', c['experiment'], episode, steps, dict(counts))
        if not all(torch.isfinite(p).all() for p in agent.network.parameters()):
            raise FloatingPointError('Nonfinite policy parameters')
        if set(counts.values()) != {c['training']['episodes'] // 5}:
            raise AssertionError('Unbalanced map rotation')
        (plotter or plot_training)(directory, 'SMOKE — non-formal results' if smoke else c['experiment'])
    except Exception as exc:
        write_json(aux / 'failure.json', {'error': repr(exc), 'steps': steps})
        raise
    return directory


def copy_history(output):
    directory = empty_output(output)
    shutil.copy2(HISTORY / 'checkpoint.pt', directory / 'checkpoint.pt')
    for name in ('training_evaluation.png',):
        target = directory / 'figures' / name; target.parent.mkdir(exist_ok=True)
        shutil.copy2(HISTORY / 'figures' / name, target)
    write_json(directory / 'others/provenance.json', {'source': str(HISTORY),
               'sha256': sha(HISTORY / 'checkpoint.pt'), 'historical_training': True, 'note': NOTE})
    plot_version_rewards(directory)


def load_policy(path):
    payload = read_checkpoint(path)
    agent = PPO(payload['config']['ppo'], scaler_from_dict(payload['scaler']))
    restore_checkpoint(payload, agent)
    return agent, payload


def validate_scene(runs, paths, seeds, methods=METHODS):
    reference = None
    for method in methods:
        rows = runs[method]; starts = {}
        for seed in seeds:
            selected = [r for r in rows if int(r['seed']) == seed]
            if [(r['stage'], int(r['step'])) for r in selected] != [('prior', 0), ('reset', 0)] + [('step', i) for i in range(1, 71)]:
                raise ValueError('Incomplete episode')
            reset = selected[1]
            starts[seed] = [float(reset[k]) for k in ('start_i','start_j','rmse','trace_prior','trace_current')]
            trajectory = [r for r in paths[method] if int(r['seed']) == seed]
            if [int(r['step']) for r in trajectory] != list(range(1,71)):
                raise ValueError('Incomplete trajectory')
            for r in selected:
                values = [float(r[k]) for k in METRICS]
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite result')
                if not np.isclose(float(r['trr']), float(r['trace_current']) / float(r['trace_prior']) * 100):
                    raise ValueError('Inconsistent Trr')
                if int(r['unique_positions']) > int(r['step']) + 1:
                    raise ValueError('Impossible unique sample count')
        if reference is not None:
            np.testing.assert_allclose(list(starts.values()), list(reference.values()), rtol=1e-10, atol=1e-8)
        reference = starts


def scene_report(scene, runs, paths, output, seeds, smoke, methods=METHODS,
                 note=NOTE, aux_output=None, weight_report=False):
    import matplotlib.pyplot as plt
    validate_scene(runs, paths, seeds, methods)
    output = Path(output)
    label = 'SMOKE — non-formal results' if smoke else 'Formal evaluation'
    summary = []
    for budget in (30, 50, 70):
        for method in methods:
            rows = [r for r in runs[method] if r['stage'] == 'step' and int(r['step']) == budget]
            item = dict(step=budget, method=method, runs=len(rows))
            for key in METRICS:
                values = np.array([float(r[key]) for r in rows])
                item[key+'_mean'] = float(values.mean()); item[key+'_std'] = float(values.std(ddof=0))
            summary.append(item)
    write_csv(Path(aux_output or output / 'others') / 'summary.csv', summary)
    lines = [f'# {scene} — ' + ('冒烟验收，非正式性能结果' if smoke else '正式测试'), '', note, '']
    headings = (('RMSE (nT)', '最终 Trace (nT²)', 'Trr (%)', '唯一点', '重复', '越界') if weight_report
                else ('RMSE(nT)', 'Trace(nT²)', 'Trr(%)', '唯一样本位置数', '重复访问数', '越界动作数'))
    for budget in (30,50,70):
        heading = f'预算{budget}步表现' if weight_report else f'{budget}步'
        lines += [f'## {heading}', '', '| 方法 | ' + ' | '.join(headings) + ' |', '|---|' + '---|'*6]
        for item in [s for s in summary if s['step'] == budget]:
            cells = [f"{item[k+'_mean']:.4f} ± {item[k+'_std']:.4f}" for k in METRICS]
            lines.append('| '+item['method']+' | '+' | '.join(cells)+' |')
        lines.append('')
    table = output / 'table/comparison.md'; table.parent.mkdir(parents=True,exist_ok=True)
    table.write_text('\n'.join(lines))
    for key, ylabel in (('rmse','RMSE (nT)'),('trr','Trr (%) — lower is better')):
        fig, ax = plt.subplots(figsize=(9,5))
        for method in methods:
            groups = [[float(r[key]) for r in runs[method] if r['stage'] != 'prior' and int(r['step']) == step] for step in range(71)]
            means = np.array([np.mean(g) for g in groups]); stds = np.array([np.std(g,ddof=0) for g in groups])
            line, = ax.plot(range(71),means,label=method)
            ax.fill_between(range(71),means-stds,means+stds,alpha=.12,color=line.get_color())
            final = next(s for s in summary if s['step']==70 and s['method']==method)
            np.testing.assert_allclose([means[-1],stds[-1]],[final[key+'_mean'],final[key+'_std']])
        ax.set(xlabel='Action step',ylabel=ylabel,title=f'{scene}: {label}',xlim=(0,70));ax.legend();ax.grid(alpha=.3)
        save_figure(fig,output / f'figures/{key}')
    fig, axes = plt.subplots(1,len(methods),figsize=(3.8*len(methods),4), squeeze=False)
    for ax, method in zip(axes.flat,methods):
        reset = next(r for r in runs[method] if int(r['seed'])==1000 and r['stage']=='reset')
        selected = [r for r in paths[method] if int(r['seed'])==1000]
        x = [int(reset['start_i'])]+[int(r['i']) for r in selected]
        y = [int(reset['start_j'])]+[int(r['j']) for r in selected]
        ax.plot(x,y,'.-',markersize=3);ax.scatter(x[0],y[0],marker='o',color='green');ax.scatter(x[-1],y[-1],marker='s',color='red')
        ax.set(title=method,xlim=(-.5,15.5),ylim=(-.5,15.5),xlabel='Grid i',ylabel='Grid j');ax.set_aspect('equal');ax.grid(alpha=.3)
    fig.suptitle(f'{scene}, seed1000: {label}')
    save_figure(fig,output / 'figures/trajectories_seed_1000')
    return summary


def test_policies(train_root, output, smoke=False):
    output = empty_output(output)
    seeds = [1000,1001] if smoke else list(range(1000,1050))
    agents = {v: load_policy(Path(train_root)/v/'checkpoint.pt') for v in ('v3_1',*VERSIONS)}
    if not smoke:
        for v, (_,p) in agents.items():
            if p['progress'] != {'episode':10000,'steps':700000,'completed':True}:
                raise ValueError(f'Formal evaluation requires completed training: {v}')
    provenance = {'smoke':smoke,'seeds':seeds,'note':NOTE,
        'checkpoints':{v:{'path':str(Path(train_root)/v/'checkpoint.pt'), 'sha256':sha(Path(train_root)/v/'checkpoint.pt')} for v in agents},'fields':{}}
    for i in range(1,6):
        scene = f'test{i}'; field = FieldModel.load(ROOT/f'data/models/{scene}.pkl')
        runs = {}; paths = {}
        provenance['fields'][scene] = {'sha256':field.sha256,'parameters':field.parameters}
        for method in METHODS:
            agent,payload = agents['v3_1' if method=='gp_trace' else method]
            cfg = deepcopy(payload['config']);cfg['field']['test_scenario'] = scene;cfg['evaluation']['seeds'] = seeds
            weights = {k:v.detach().clone() for k,v in agent.network.state_dict().items()}
            scaler = deepcopy(agent.scaler.to_dict())
            rows, trajectory, _ = evaluate(field,agent.scaler,cfg,None if method=='gp_trace' else agent,
                                            'gp_trace' if method=='gp_trace' else 'ppo',episode=payload['progress']['episode'])
            if scaler != agent.scaler.to_dict() or any(not weights[k].equal(v) for k,v in agent.network.state_dict().items()):
                raise ValueError('Evaluation changed policy')
            for row in (*rows,*trajectory):
                row['algorithm'] = method
            runs[method]=rows;paths[method]=trajectory
            write_csv(output/scene/f'others/{method}/eval_metrics.csv',rows)
            write_csv(output/scene/f'others/{method}/trajectories.csv',trajectory)
        scene_report(scene,runs,paths,output/scene,seeds,smoke)
        LOG.info('Completed paired evaluation %s; seeds=%s',scene,seeds)
    write_json(output/'others/metadata.json',provenance)
    return output


def replot(train_root, test_root, output):
    import json
    output = empty_output(output)
    metadata = json.loads((Path(test_root)/'others/metadata.json').read_text())
    for version in VERSIONS:
        plot_training(Path(train_root)/version, 'SMOKE — non-formal results' if metadata['smoke'] else version, output/'training'/version)
    plot_version_rewards(Path(train_root)/'v3_1', output/'training/v3_1',
                         'SMOKE — non-formal results' if metadata['smoke'] else 'v3_1')
    target = output/'training/v3_1/figures/training_evaluation.png'
    shutil.copy2(Path(train_root)/'v3_1/figures/training_evaluation.png', target)
    for i in range(1,6):
        scene=f'test{i}'; base=Path(test_root)/scene/'others'
        runs={m:read_csv(base/m/'eval_metrics.csv') for m in METHODS}
        paths={m:read_csv(base/m/'trajectories.csv') for m in METHODS}
        scene_report(scene,runs,paths,output/scene,metadata['seeds'],metadata['smoke'])
    write_json(output/'others/metadata.json',metadata)
