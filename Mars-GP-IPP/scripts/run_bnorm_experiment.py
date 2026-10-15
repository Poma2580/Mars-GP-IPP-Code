"""Run the approved B-Norm protocol: baselines, training, then paired evaluation."""
import _bootstrap
import argparse
import hashlib
import json
from pathlib import Path
from evaluate import run as evaluate_run
from run_comparison import run_comparison, checkpoint_config
from mars_ipp.utils.config import ROOT, load_config, merge, validate_output_path
from mars_ipp.utils.logger import write_json, write_csv
from mars_ipp.training.checkpoint import read_checkpoint
from mars_ipp.training.trainer import train
from mars_ipp.visualization.comparison import plot_results
from mars_ipp.visualization.learning_curves import read_csv


def run():
    config = load_config(ROOT/'configs/main/bnorm_no_mh_uncertain.yaml')
    base = load_config(ROOT/'configs/comparison/bnorm_baselines.yaml')
    comparison_root = validate_output_path(ROOT/'results/comparison'/config['experiment'])
    comparison_root.mkdir(parents=True, exist_ok=True)
    msr_checkpoint = ROOT/'results/main/msr/seed_0/checkpoint.pt'
    training_dir = ROOT/'results/main'/config['experiment']/'seed_0'
    checkpoint = training_dir/'checkpoint.pt'
    status_path = comparison_root/'pipeline_status.json'

    def status(stage, completed=False, **extra):
        write_json(status_path, {'stage':stage, 'completed':completed, **extra})

    try:
        status('baselines')
        for scene in ('zhurong','perseverance'):
            effective = merge(base, {'field': {'test_scenario': scene}})
            output = comparison_root/scene
            if output.exists():
                manifest = json.loads((output/'comparison.json').read_text())
                if not manifest['completed']:
                    raise ValueError(f'Incomplete comparison cannot be silently reused: {output}')
            else:
                run_comparison(effective, output, msr_checkpoint)
        status('training')
        if training_dir.exists():
            payload = read_checkpoint(checkpoint)
            if not payload['progress'].get('completed') or payload['progress']['episode'] != 10000:
                raise ValueError('Incomplete training cannot be resumed; investigate before a new run')
            if payload['config'] != config:
                raise ValueError('Existing training configuration differs from approved protocol')
        else:
            train(config, training_dir)
            plot_results(training_dir, formats=('png','pdf'))
        status('formal_evaluation')
        for scene in ('zhurong','perseverance'):
            effective = merge(base, {'field': {'test_scenario': scene}})
            output = comparison_root/scene
            ours = output/'ours'
            manifest_path = output/'comparison.json'
            manifest = json.loads(manifest_path.read_text())
            if not ours.exists():
                own_config = checkpoint_config(checkpoint, effective, 'ours')
                evaluate_run(own_config, ours, checkpoint, 'ours', ('png','pdf'), plot=True)
            if 'ours' not in manifest['algorithms']:
                manifest['algorithms'].append('ours')
                manifest['labels']['ours'] = 'Ours'
                manifest['ours_checkpoint'] = str(checkpoint)
                write_json(manifest_path, manifest)
            run_comparison(effective, output, msr_checkpoint, checkpoint, stage='summarize')
            run_comparison(effective, output, msr_checkpoint, checkpoint, stage='plot')
        cross_scene = []
        for scene in ('zhurong','perseverance'):
            for row in read_csv(comparison_root/scene/'summary/table_iv.csv'):
                cross_scene.append({'scenario':scene, **row})
        write_csv(comparison_root/'cross_scene_summary.csv', cross_scene)
        source_paths = [*sorted((ROOT/'src/mars_ipp').rglob('*.py')),
                        *sorted((ROOT/'scripts').glob('*.py')),
                        ROOT/'configs/main/bnorm_no_mh_uncertain.yaml',
                        ROOT/'configs/comparison/bnorm_baselines.yaml']
        write_json(comparison_root/'validation.json', {
            'completed': True, 'training_episodes':10000, 'training_seed':0,
            'evaluation_seeds':list(range(1000,1050)), 'steps':70,
            'methods':['gp_greedy','gp_trace','msr','cover','ours'],
            'msr_note':'Policy reused from old-protocol 10000 Episode training; online GP protocol changed only for this evaluation.',
            'msr_checkpoint_sha256':hashlib.sha256(msr_checkpoint.read_bytes()).hexdigest(),
            'ours_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}})
        status('complete', True, training=str(training_dir), results=str(comparison_root))
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
    return comparison_root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(run())
