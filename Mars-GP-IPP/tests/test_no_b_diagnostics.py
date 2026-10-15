from copy import deepcopy
import numpy as np
import pytest
from mars_ipp.envs.actions import ACTIONS
from mars_ipp.evaluation.no_b_diagnostics import seed_statistics, summarize_group


def saved_run(actions, seed=1000):
    position = np.random.default_rng(seed).integers(0, 16, 2)
    rows = [dict(stage='prior', step=0, seed=seed, rmse=12., trace_current=120., sampled=False),
            dict(stage='reset', step=0, seed=seed, rmse=10., trace_current=100., sampled=True)]
    paths = []
    for step, action in enumerate(actions, start=1):
        candidate = position + ACTIONS[action]
        outside = bool(np.any(candidate < 0) or np.any(candidate >= 16))
        if not outside:
            position = candidate
        rows.append(dict(stage='step', step=step, seed=seed, rmse=5., trace_current=25., sampled=not outside))
        paths.append(dict(step=step, action=action, i=int(position[0]), j=int(position[1])))
    return rows, paths


def test_revisits_exclude_boundary_no_motion():
    rows, paths = saved_run([0] * 70)
    result = seed_statistics(rows, paths)
    legal_moves = 15 - int(np.random.default_rng(1000).integers(0, 16, 2)[1])
    assert result['unique_cells'] == 1 + legal_moves
    assert result['repeat_visits'] == 0
    assert result['outside_actions'] == 70 - legal_moves
    assert result['rmse_decline_pct'] == 50.
    assert result['trace_decline_pct'] == 75.
    assert result['action_0_pct'] == 100.
    rows, paths = saved_run([0, 2] * 35)
    result = seed_statistics(rows, paths)
    assert result['unique_cells'] == 2
    assert result['repeat_visits'] == 69
    assert result['outside_actions'] == 0
    assert result['action_0_pct'] == result['action_2_pct'] == 50.


def test_missing_duplicate_and_inconsistent_steps_rejected():
    rows, paths = saved_run([0] * 70)
    with pytest.raises(ValueError, match='metric steps'):
        seed_statistics(rows[:-1], paths)
    with pytest.raises(ValueError, match='trajectory steps'):
        seed_statistics(rows, paths[:-1])
    duplicate = deepcopy(rows)
    duplicate[-1] = duplicate[-2]
    with pytest.raises(ValueError, match='metric steps'):
        seed_statistics(duplicate, paths)
    paths[0]['i'] += 1
    with pytest.raises(ValueError, match='mismatch'):
        seed_statistics(rows, paths)


def test_relative_decline_is_averaged_per_seed_with_population_sd():
    group = {'stats': [{'seed': 1000, 'initial_rmse': 10., 'rmse_decline_pct': 50.},
                       {'seed': 1001, 'initial_rmse': 100., 'rmse_decline_pct': 10.}]}
    result = summarize_group(group)
    assert result['rmse_decline_pct_mean'] == 30.
    assert result['rmse_decline_pct_std'] == 20.


def test_diagnostic_report_end_to_end_with_synthetic_runs(scaler, tmp_path):
    """Synthetic fixtures in tmp_path exercise pairing, artifacts and refusal to overwrite."""
    import yaml
    from mars_ipp.agents.ppo.ppo import PPO
    from mars_ipp.training.checkpoint import save_checkpoint
    from mars_ipp.utils.config import ROOT, load_config, merge
    from mars_ipp.utils.logger import write_csv, write_json
    from mars_ipp.evaluation.no_b_diagnostics import run_diagnostics, validate_training

    full = merge(load_config(ROOT / 'configs/main/original.yaml'), {'ppo': {'device': 'cpu'}})
    no_b = merge(full, {'experiment': 'no_b_rbf', 'seed': 0, 'seeds': [0],
                       'ppo': {'observation_mode': 'position_only'}})
    metadata = {'scaler': scaler.to_dict(), 'fields': {
        scenario: {'sha256': 'synthetic-test-fixture', 'parameters': {'type': 'rbf'}}
        for scenario in ('zhurong', 'perseverance')}}
    arguments = {}
    for method, config in (('full', full), ('no_b', no_b)):
        training = tmp_path / method / 'seed_0'
        training.mkdir(parents=True)
        (training / 'config.yaml').write_text(yaml.safe_dump(config))
        write_json(training / 'metadata.json', metadata)
        write_json(training / 'run_status.json', {'episode': 10000, 'steps': 700000, 'completed': True,
                                                 'pending_rollout': 0, 'updates': 350})
        save_checkpoint(training / 'checkpoint.pt', PPO(config['ppo'], scaler), config,
                        {'episode': 10000, 'steps': 700000, 'completed': True})
        write_csv(training / 'eval_endpoints.csv', [
            dict(episode=episode, seed=seed, step=70, stage='step', rmse=5.)
            for episode in range(200, 10001, 200) for seed in range(1000, 1050)])
        arguments[f'{method}_training'] = training
        for scenario in ('zhurong', 'perseverance'):
            directory = tmp_path / f'{method}_{scenario}'
            directory.mkdir()
            (directory / 'config.yaml').write_text(yaml.safe_dump(merge(config, {'field': {'test_scenario': scenario}})))
            write_json(directory / 'metadata.json', {**metadata, 'checkpoint': str(training / 'checkpoint.pt')})
            metrics, trajectories = [], []
            for seed in range(1000, 1050):
                rows, paths = saved_run([0, 2] * 35, seed)
                metrics.extend(dict(row, episode=10000, algorithm='ppo') for row in rows)
                trajectories.extend(dict(row, episode=10000, algorithm='ppo', seed=seed) for row in paths)
            write_csv(directory / 'eval_metrics.csv', metrics)
            write_csv(directory / 'trajectories.csv', trajectories)
            arguments[f'{method}_{scenario}'] = directory
    arguments['output'] = tmp_path / 'diagnostics'
    arguments['report_path'] = tmp_path / 'docs/no_b_diagnostic_report.md'
    summaries = run_diagnostics(**arguments)
    assert len(summaries) == 4
    assert all(row['runs'] == 50 and row['rmse_decline_pct_mean'] == 50. for row in summaries)
    output = arguments['output']
    assert (output / 'diagnostic_summary.csv').exists()
    assert (output / 'diagnostic_per_seed.csv').exists()
    assert 'seed=1000' in arguments['report_path'].read_text()
    assert len(list((output / 'diagnostic_figures').glob('*.png'))) == 4
    assert len(list((output / 'diagnostic_figures').glob('*.pdf'))) == 4
    with pytest.raises(FileExistsError):
        run_diagnostics(**arguments)
    write_json(arguments['no_b_training'] / 'run_status.json', {
        'episode': 200, 'steps': 14000, 'completed': False, 'pending_rollout': 0, 'updates': 7})
    with pytest.raises(ValueError, match='not complete'):
        validate_training(arguments['no_b_training'], 'position_only')
