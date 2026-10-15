"""Validate one completed V3 local-raw replacement, then replace only canonical V3 result roots."""
import _bootstrap
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import yaml
from run_v3_gp_trace import (preflight, checkpoint_for, validate_evaluation,
                             validate_paired_starts, validate_artifacts,
                             protect_history, seal_artifacts)
from mars_ipp.utils.config import ROOT, load_config, merge
from mars_ipp.utils.logger import write_json

CONFIG = ROOT / 'configs/main/v3_gp_trace_train1.yaml'
METHODS = ('gp_trace', 'ppo_seed_0')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def verify_mirror(snapshot):
    mirror = ROOT / '代码上传专用/Mars_GP_IPP'
    expected = json.loads(Path(snapshot).read_text())
    current = {str(p.relative_to(mirror)): sha256(p) for p in mirror.rglob('*') if p.is_file()}
    if current != expected:
        raise ValueError('代码上传专用 changed')
    return len(current)


def verify_stage(tag, mirror_snapshot):
    cfg = load_config(CONFIG)
    if cfg['reward'] != {'mode': 'gp_trace_local_raw', 'weights': [1.], 'disabled': []} or cfg['ppo']['gamma'] != .99:
        raise ValueError('Unexpected V3 reward/protocol')
    preflight(cfg)
    name = cfg['experiment'] + '_' + tag
    main = ROOT / 'results/main' / name
    comparison = ROOT / 'results/comparison' / name
    smoke_main = ROOT / 'results/main' / (name + '_smoke')
    smoke_comparison = ROOT / 'results/comparison' / (name + '_smoke')
    status = json.loads((comparison/'pipeline_status.json').read_text())
    if status['stage'] != 'complete' or not status['completed'] or status['training_seeds'] != [0]:
        raise ValueError('Staged experiment is incomplete')
    checkpoint = checkpoint_for(cfg, main/'seed_0')
    progress = json.loads((main/'seed_0'/'run_status.json').read_text())
    if progress['episode'] != 10000 or progress['steps'] != 700000 or progress['updates'] != 350:
        raise ValueError('Training budget or update count differs')
    smoke = json.loads((smoke_comparison/'smoke_report.json').read_text())
    if not smoke['completed'] or smoke['training_seeds'] != [0] or smoke['episodes_per_training_seed'] != 40:
        raise ValueError('Smoke did not complete')
    smoke_cfg = merge(cfg, {'training': {'episodes':40,'eval_every':10,'checkpoint_every':10}})
    smoke_checkpoint = checkpoint_for(smoke_cfg, smoke_main/'seed_0')
    for directory, own, ckpt, seeds in ((comparison/'gp_trace',cfg,None,list(range(1000,1050))),
                                         (comparison/'ppo_seed_0',cfg,checkpoint,list(range(1000,1050))),
                                         (smoke_comparison/'gp_trace',merge(smoke_cfg,{'evaluation':{'seeds':[1000]}}),None,[1000]),
                                         (smoke_comparison/'ppo_seed_0',merge(smoke_cfg,{'evaluation':{'seeds':[1000]}}),smoke_checkpoint,[1000])):
        rows = validate_evaluation(directory,own,ckpt,'gp_trace' if ckpt is None else 'ppo')
        if {int(r['seed']) for r in rows} != set(seeds):
            raise ValueError('Evaluation seeds differ')
        for row in rows:
            if row['stage'] == 'step':
                reward=float(row['r_gp'])
                if reward < 0 or not __import__('math').isfinite(reward):
                    raise ValueError('Invalid local Trace reward')
                if row['sampled'] == 'True' and abs(reward-float(row['gp_actual_delta'])) > 1e-7:
                    raise ValueError('Reward differs from actual local Trace decrease')
                if row['sampled'] == 'False' and reward != 0:
                    raise ValueError('Unsampled action received a reward')
    validate_paired_starts(comparison,list(range(1000,1050)),METHODS)
    validate_paired_starts(smoke_comparison,[1000],METHODS)
    table=read_rows(comparison/'summary/comparison.csv')
    if [r['method'] for r in table] != list(METHODS) or any(int(r['test_runs']) != 50 for r in table):
        raise ValueError('Formal summary incomplete')
    for folder in (main/'seed_0',smoke_main/'seed_0',comparison/'gp_trace',comparison/'ppo_seed_0',
                   smoke_comparison/'gp_trace',smoke_comparison/'ppo_seed_0'):
        validate_artifacts(folder)
    mirror_files = verify_mirror(mirror_snapshot)
    protect_history()
    return cfg, (main,comparison,smoke_main,smoke_comparison), mirror_files


def promote(tag, mirror_snapshot):
    cfg, (main,comparison,smoke_main,smoke_comparison), mirror_files = verify_stage(tag,mirror_snapshot)
    old = [ROOT/category/name for category in ('results/main','results/comparison')
           for name in ('v3_gp_trace_train1','v3_gp_trace_train1_smoke',
                        'v3_sb3_short_train1','v3_sb3_short_train1_smoke',
                        'v3_1_gp_trace_global_raw_train1','v3_1_gp_trace_global_raw_train1_smoke')]
    destinations = [ROOT/'results/main/v3_gp_trace_train1',ROOT/'results/comparison/v3_gp_trace_train1',
                    ROOT/'results/main/v3_gp_trace_train1_smoke',ROOT/'results/comparison/v3_gp_trace_train1_smoke']
    for directory in old:
        if directory.exists():
            if not directory.is_dir() or directory.is_symlink():
                raise ValueError(f'Unexpected result path: {directory}')
            shutil.rmtree(directory)
    for source,destination in zip((main,comparison,smoke_main,smoke_comparison),destinations):
        source.rename(destination)
    for evaluation, checkpoint in ((destinations[1]/'ppo_seed_0',destinations[0]/'seed_0/checkpoint.pt'),
                                   (destinations[3]/'ppo_seed_0',destinations[2]/'seed_0/checkpoint.pt')):
        metadata_path=evaluation/'metadata.json'
        metadata=json.loads(metadata_path.read_text())
        if metadata['checkpoint_sha256'] != sha256(checkpoint):
            raise ValueError('Promoted checkpoint SHA mismatch')
        metadata['checkpoint']=str(checkpoint)
        write_json(metadata_path,metadata)
        seal_artifacts(evaluation)
    # Both evaluation protocols must still validate at their canonical locations.
    validate_evaluation(destinations[1]/'gp_trace',cfg,None,'gp_trace')
    validate_evaluation(destinations[1]/'ppo_seed_0',cfg,destinations[0]/'seed_0/checkpoint.pt','ppo')
    validate_paired_starts(destinations[1],list(range(1000,1050)),METHODS)
    protected=protect_history()
    if verify_mirror(mirror_snapshot) != mirror_files:
        raise ValueError('Export mirror changed during promotion')
    receipt={'completed':True,'reward_mode':'gp_trace_local_raw','training_seed':0,
             'training_episodes':10000,'training_steps':700000,'test_seeds':list(range(1000,1050)),
             'checkpoint_sha256':sha256(destinations[0]/'seed_0/checkpoint.pt'),
             'source_tag':tag,'non_v3_artifacts_verified':protected,'mirror_files_verified_unchanged':mirror_files,
             'removed_old_series':['V3 relative','V3.1 global raw','SB3 short']}
    write_json(destinations[1]/'replacement_validation.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tag',required=True)
    parser.add_argument('--mirror-snapshot',default='/tmp/v3_mirror_before.json')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if not args.run_tag.replace('_','').replace('-','').isalnum():
        parser.error('Invalid run tag')
    if args.verify_only:
        cfg, paths, mirror_files = verify_stage(args.run_tag,args.mirror_snapshot)
        print(json.dumps({'ready':True,'staged_paths':[str(p) for p in paths],
                          'reward_mode':cfg['reward']['mode'],'mirror_files_unchanged':mirror_files},indent=2))
    else:
        promote(args.run_tag,args.mirror_snapshot)
