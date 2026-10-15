"""Audit saved V2 artifacts without training or rerunning any evaluation."""
import _bootstrap
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np
import yaml
from mars_ipp.utils.logger import write_json
from mars_ipp.utils.config import ROOT, load_config, merge
from mars_ipp.training.checkpoint import read_checkpoint, restore_checkpoint
from mars_ipp.envs.observations import scaler_from_dict
from mars_ipp.envs.actions import ACTIONS
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.evaluation.statistics import ALGORITHMS, load_runs
from mars_ipp.evaluation.local_global import METHODS, METRICS, validate_run, endpoint_stats
from mars_ipp.visualization.learning_curves import read_csv

EXPERIMENTS = ('bnorm_no_mh_local_global_3x3_rbf', 'bnorm_no_mh_local_global_3x3_rbf_no_local')
OUTPUT = ROOT / 'results/comparison' / EXPERIMENTS[0]
BASELINES = ROOT / 'results/comparison/bnorm_no_mh_uncertain_rbf'


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def same(a, b, message, rtol=1e-10, atol=1e-10):
    check(np.allclose(a, b, rtol=rtol, atol=atol), message)


def figure(path):
    check(path.exists() and path.stat().st_size > 1000, f'Missing/empty figure {path}')
    if path.suffix == '.png':
        check(path.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n', f'Invalid PNG {path}')
    else:
        check(path.read_bytes()[:4] == b'%PDF', f'Invalid PDF {path}')


def audit():
    check(json.loads((OUTPUT/'pipeline_status.json').read_text()) == {'stage':'complete','completed':True}, 'Pipeline incomplete')
    preflight = json.loads((OUTPUT/'preflight.json').read_text())
    old_validation = json.loads((BASELINES/'validation.json').read_text())
    for name, experiment in (('ours', 'bnorm_no_mh_uncertain_rbf'), ('msr', 'msr')):
        checkpoint = ROOT/'results/main'/experiment/'seed_0/checkpoint.pt'
        check(hashlib.sha256(checkpoint.read_bytes()).hexdigest() == old_validation[f'{name}_checkpoint_sha256'], 'Original V1/Msr checkpoint modified')
    master = read_csv(OUTPUT/'cross_scene_summary.csv')
    check(len(master) == 12, 'Wrong cross-scene summary row count')
    training, evaluations, paired, main_tables = [], [], {}, {}
    total_rows = total_actions = 0
    source_sets=[]
    for method, experiment in zip(METHODS, EXPERIMENTS):
        for seed in (0,1,2):
            directory=ROOT/'results/main'/experiment/f'seed_{seed}'
            cfg=merge(load_config(ROOT/'configs/main'/f'{experiment}.yaml'),{'seed':seed})
            payload=read_checkpoint(directory/'checkpoint.pt')
            status=json.loads((directory/'run_status.json').read_text())
            check(payload['config']==cfg, f'Checkpoint config mismatch {directory}')
            check(payload['progress']=={'episode':10000,'steps':700000,'completed':True}, 'Not final checkpoint')
            check(status['completed'] and status['updates']==350 and status['pending_rollout']==0, 'Incorrect update/buffer status')
            check(payload['input_dim']==22 and payload['model_version']=='ppo-22d-v2', 'Wrong model dimensions/version')
            scaler=scaler_from_dict(payload['scaler']); policy=merge(cfg['ppo'],{'device':'cpu'})
            restore_checkpoint(payload,PPO(policy,scaler))
            meta=json.loads((directory/'metadata.json').read_text())
            source_sets.append(meta['source_sha256'])
            for relative, digest in meta['source_sha256'].items():
                check(hashlib.sha256((ROOT/'src/mars_ipp'/relative).read_bytes()).hexdigest()==digest, f'Source changed: {relative}')
            for scene in ('zhurong','perseverance'):
                same(meta['c_global'][scene],preflight['scenes'][scene]['c_global'],'C_global changed')
            train_rows=read_csv(directory/'train_metrics.csv'); monitor=read_csv(directory/'eval_endpoints.csv'); updates=read_csv(directory/'updates.csv')
            check([int(r['episode']) for r in train_rows]==list(range(1,10001)), 'Missing/duplicate training episodes')
            check([int(r['steps']) for r in train_rows]==list(range(70,700001,70)), 'Wrong training steps')
            check(len(updates)==350 and [int(r['samples']) for r in updates]==[2000]*350, 'Wrong PPO rollout sizes')
            check(all(np.isfinite(float(r[k])) for r in updates for k in ('loss','policy_loss','value_loss','entropy')), 'Nonfinite PPO loss')
            check([(int(r['episode']),int(r['seed'])) for r in monitor]==[(e,s) for e in range(200,10001,200) for s in range(1000,1005)], 'Monitoring protocol differs')
            check(all(r['stage']=='step' and int(r['step'])==70 for r in monitor),'Nonterminal monitoring rows')
            weights=cfg['reward']['weights']
            for row in train_rows:
                terms=[float(row[f'r_{k}']) for k in ('local','global','rep','out')]
                check(np.isfinite(terms).all(), 'Nonfinite training rewards')
                check(all(np.isfinite(float(row[k])) for k in ('reward', 'rmse', 'trr', 'trace_current', 'trace_prior')), 'Nonfinite training metrics')
                same(float(row['reward']),np.dot(weights,terms),'Training weighted reward mismatch',atol=1e-9)
            diagnostics=json.loads((directory/'diagnostics.json').read_text())
            check(diagnostics['steps']==700000 and diagnostics['rewards']['all']['global']['count']==700000,'Wrong reward diagnostic counts')
            check(diagnostics['rewards']['all']['global']['above_one']==0,'Global reward above one')
            for suffix in ('png','pdf'):
                for name in ('training','training_evaluation'):figure(directory/'figures'/f'{name}.{suffix}')
            training.append({'method':method,'training_seed':seed,'episodes':10000,'steps':700000,'monitor_episodes':len(monitor),
                             'seconds':status['invocation_seconds'],'environment_seconds_per_step':diagnostics['environment_seconds_per_step'],
                             'reward_new_samples':diagnostics['rewards']['new_samples'],
                             'checkpoint_sha256':hashlib.sha256((directory/'checkpoint.pt').read_bytes()).hexdigest()})
    check(all(s==source_sets[0] for s in source_sets),'Training source snapshots differ')
    for scene in ('zhurong','perseverance'):
        baselines=load_runs([BASELINES/scene/name for name in (*ALGORITHMS,'ours')])
        reference=baselines[ALGORITHMS[0]]
        for name, original in preflight['scenes'][scene]['baseline_sources'].items():
            d=Path(original['directory'])
            check(hashlib.sha256((d/'eval_metrics.csv').read_bytes()).hexdigest()==original['metrics_sha256'],'Baseline metrics modified')
            check(hashlib.sha256((d/'trajectories.csv').read_bytes()).hexdigest()==original['trajectory_sha256'],'Baseline trajectories modified')
        table={r['algorithm']:r for r in read_csv(OUTPUT/scene/'summary/table_iv.csv')}
        for name, row in table.items():
            check(next(r for r in master if r['scenario'] == scene and r['algorithm'] == name) == row, 'Cross-scene summary differs')
        per_seed=read_csv(OUTPUT/scene/'summary/per_training_seed.csv')
        combined=read_csv(OUTPUT/scene/'summary/v2_eval_metrics.csv')
        check(len(combined)==21600,'Wrong aggregate row count')
        check(len({(r['algorithm'],r['training_seed'],r['seed'],r['stage'],r['step']) for r in combined})==21600,'Aggregate duplicate identities')
        main_tables[scene]=[]
        final_groups={}
        for method, experiment in zip(METHODS,EXPERIMENTS):
            final_groups[method]={}
            all_rows=[]; means={k:[] for k in METRICS}
            for train_seed in (0,1,2):
                directory=OUTPUT/scene/method/f'seed_{train_seed}'
                rows=validate_run(directory,reference,train_seed); all_rows.extend(rows)
                paths=read_csv(directory/'trajectories.csv')
                meta=json.loads((directory/'metadata.json').read_text())
                train_dir=ROOT/'results/main'/experiment/f'seed_{train_seed}'
                check(meta['checkpoint_sha256']==hashlib.sha256((train_dir/'checkpoint.pt').read_bytes()).hexdigest(),'Evaluated wrong checkpoint')
                check(meta['model_version']=='ppo-22d-v2','Wrong evaluated model version')
                config=yaml.safe_load((directory/'config.yaml').read_text())
                same(config['reward']['weights'],[.25]*4 if method=='v2' else [0.,.25,.25,.25],'Wrong reward variant')
                check(config['ppo']['update_epochs']==10,'PPO update epochs changed')
                grouped=defaultdict(list); trajectories=defaultdict(list)
                for r in rows:grouped[int(r['seed'])].append(r)
                for r in paths:trajectories[int(r['seed'])].append(r)
                max_step_error=max_episode_error=0.
                for test_seed in range(1000,1050):
                    run=grouped[test_seed]; path=trajectories[test_seed]
                    start=np.random.default_rng(test_seed).integers(0,16,2)
                    same(start,[int(run[1]['start_i']),int(run[1]['start_j'])],'Start RNG differs')
                    position=start.copy(); visited={tuple(position)}; repetitions=outside=0; reward_sum=0.
                    c=float(run[1]['c_global']); reset_trace=float(run[1]['trace_current'])
                    for r,t,prev in zip(run[2:],path,run[1:-1]):
                        next_position=position+ACTIONS[int(t['action'])]
                        out=bool((next_position<0).any() or (next_position>=16).any())
                        repeated=False
                        if out:outside+=1
                        else:
                            repeated=tuple(next_position) in visited
                            repetitions+=int(repeated);position=next_position;visited.add(tuple(position))
                        same(position,[int(t['i']),int(t['j'])],'Action/position mismatch')
                        check(int(r['unique_positions'])==len(visited)==int(r['samples']),'Unique sample/position mismatch')
                        check(int(r['repeated_visits'])==repetitions and int(r['outside_actions'])==outside,'Incorrect action counters')
                        same(float(r['coverage']),len(visited)/256,'Coverage mismatch')
                        check((r['sampled']=='True') == (not out and not repeated),'Incorrect sampled flag')
                        same(float(r['r_rep']),-1. if repeated else 0.,'Repeat penalty mismatch')
                        same(float(r['r_out']),-1. if out else 0.,'Outside penalty mismatch')
                        terms=np.array([float(r[f'r_{k}']) for k in ('local','global','rep','out')])
                        check(np.isfinite(terms).all() and terms[0]>=0 and terms[1]>=0,'Invalid formal reward')
                        same(float(t['reward']),np.dot(config['reward']['weights'],terms),'Formal weighted reward mismatch')
                        if out or repeated:check(terms[0]==terms[1]==0.,'Information reward on invalid/repeated sample')
                        actual=(float(prev['trace_current'])-float(r['trace_current']))/c
                        same(terms[1],actual,'Per-step global Trace reward mismatch',rtol=2e-7,atol=1e-7)
                        max_step_error=max(max_step_error,abs(terms[1]-actual));reward_sum+=terms[1]
                        same(float(r['online_noise_variance']),1e-6,'Noise changed')
                    error=abs(reward_sum-(reset_trace-float(run[-1]['trace_current']))/c)
                    same(reward_sum,(reset_trace-float(run[-1]['trace_current']))/c,'Episode global Trace identity mismatch',rtol=2e-7,atol=1e-7)
                    max_episode_error=max(max_episode_error,error)
                    final_groups[method][(train_seed,test_seed)]=run[-1]
                stats=endpoint_stats(rows)
                saved=next(r for r in per_seed if r['method']==table[method]['method'] and int(r['training_seed'])==train_seed)
                for key,value in stats.items():same(float(saved[key]),value,'Per-seed statistics differ')
                for key in METRICS:means[key].append(stats[f'{key}_mean'])
                for suffix in ('png','pdf'):
                    for name in ('evaluation','component_rmse','field_map','error_trajectory'):figure(directory/'figures'/f'{name}.{suffix}')
                snapshot=np.load(directory/'field_snapshot.npz')
                check(snapshot['points'].shape==(961,2) and snapshot['trajectory'].shape==(71,2),'Invalid representative snapshot')
                same(snapshot['trajectory'][0],np.random.default_rng(1000).integers(0,16,2)*2,'Representative seed differs')
                total_rows+=len(rows);total_actions+=len(paths)
                evaluations.append({'scene':scene,'method':method,'training_seed':train_seed,'episodes':50,'metric_rows':len(rows),'actions':len(paths),
                    'max_global_step_identity_error':max_step_error,'max_global_episode_identity_error':max_episode_error, **stats})
                final_monitor=[r for r in read_csv(train_dir/'eval_endpoints.csv') if int(r['episode'])==10000]
                if scene=='perseverance':
                    for monitor in final_monitor:
                        final=grouped[int(monitor['seed'])][-1]
                        same([float(monitor[k]) for k in ('rmse','trr')],[float(final[k]) for k in ('rmse','trr')],'Final monitoring/formal results differ')
            stats=endpoint_stats(all_rows)
            for key,value in stats.items():same(float(table[method][key]),value,'Pooled statistics differ')
            for key in METRICS:same(float(table[method][f'{key}_training_seed_std']),np.std(means[key],ddof=0),'Between-training-seed SD differs')
        for method in ALGORITHMS:
            for key,value in endpoint_stats(baselines[method]['rows']).items():same(float(table[method][key]),value,'Baseline statistics differ')
        for suffix in ('png','pdf'):
            for name in ('rmse','trr'):figure(OUTPUT/scene/'figures'/f'{name}.{suffix}')
        for method in (*ALGORITHMS,*METHODS):main_tables[scene].append(table[method])
        paired[scene]={}
        for key in ('rmse','trr'):
            differences=np.array([float(final_groups['v2_global_only'][identity][key])-float(value[key]) for identity,value in final_groups['v2'].items()])
            by_seed=[float(np.mean([float(final_groups['v2_global_only'][(seed,s)][key])-float(final_groups['v2'][(seed,s)][key]) for s in range(1000,1050)])) for seed in (0,1,2)]
            paired[scene][key]={'global_only_minus_v2_mean':float(differences.mean()),'per_training_seed_mean_differences':by_seed,
                                'fraction_global_only_lower':float(np.mean(differences<0)), 'note':'Descriptive paired differences; not an independent 150-sample significance test'}
    check(total_rows==43200 and total_actions==42000,'Formal experiment counts differ')
    report={'passed':True,'training_runs':6,'training_episodes':60000,'training_steps':4200000,'training_seconds':sum(r['seconds'] for r in training),
            'formal_runs':12,'formal_episodes':600,'formal_metric_rows':total_rows,'formal_actions':total_actions,'monitor_episodes':1500,
            'baseline_hashes_unchanged':True,'identical_training_source_snapshots':True,'checkpoint_and_policy_loads_passed':True,
            'original_v1_msr_checkpoint_hashes_unchanged':True,'cross_scene_master_table_matches':True,
            'reward_and_trace_accounting_passed':True,'statistics_independently_recomputed':True,'artifact_headers_checked':True,
            'training':training,'evaluations':evaluations,'paired_global_only_minus_v2':paired,'main_tables':main_tables}
    write_json(OUTPUT/'acceptance_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('training','evaluations','main_tables')},indent=2))
    return report


if __name__=='__main__':
    audit()
