from copy import deepcopy
import numpy as np
import torch
from mars_ipp.training.trainer import train
from mars_ipp.training.checkpoint import read_checkpoint
from mars_ipp.field.generate_field import prepare_fields
from mars_ipp.field.dataset import load_dataset, evaluation_grid
from mars_ipp.utils.config import ROOT, merge
from mars_ipp.evaluation.evaluator import evaluate
from mars_ipp.envs.observations import ObservationScaler
from mars_ipp.agents.ppo.ppo import PPO
from mars_ipp.utils.seed import seed_all, rng_state


def test_real_field_shapes_and_mapping(config):
    fields = prepare_fields(config)
    for name, field in fields.items():
        x, y = load_dataset(ROOT / config['field']['workbook'], name)
        np.testing.assert_allclose(field.model.X, x, atol=1e-12)
        np.testing.assert_allclose(field.model.Y, y, atol=1e-12)
        mean, variance = field.model.predict_noiseless(evaluation_grid())
        assert mean.shape == (961, 3) and variance.shape == (961, 1)


def test_periodic_append_and_final_checkpoint(config, tmp_path, monkeypatch):
    import csv
    import mars_ipp.training.trainer as trainer
    from mars_ipp.visualization.comparison import plot_results
    config = merge(config, {'training': {'episodes': 7, 'eval_every': 2, 'checkpoint_every': 3},
                           'env': {'max_steps': 3}, 'ppo': {'rollout_steps': 5, 'update_epochs': 2},
                           'evaluation': {'seeds': [1000, 1001]}})
    cycles = []
    original_save = trainer.save_checkpoint

    def checked_save(path, agent, config, progress):
        directory = path.parent
        with (directory / 'train_metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert [int(r['episode']) for r in rows] == list(range(1, progress['episode'] + 1))
        with (directory / 'eval_endpoints.csv').open() as stream:
            endpoints = list(csv.DictReader(stream))
        assert len(endpoints) == (len(cycles) + 1) * 2
        assert {int(r['episode']) for r in endpoints[-2:]} == {progress['episode']}
        if not progress['completed']:
            assert len(agent.buffer) == progress['steps'] % 5
        cycles.append(progress['episode'])
        original_save(path, agent, config, progress)

    monkeypatch.setattr(trainer, 'save_checkpoint', checked_save)
    directory = train(config, tmp_path / 'training')
    assert cycles == [2, 3, 4, 6, 7]
    payload = read_checkpoint(directory / 'checkpoint.pt')
    assert payload['progress'] == {'episode': 7, 'steps': 21, 'completed': True}
    for filename in ('train_metrics.csv', 'updates.csv', 'eval_endpoints.csv'):
        with (directory / filename).open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key, value in row.items():
                if key not in ('stage', 'algorithm', 'sampled'):
                    assert np.isfinite(float(value)), (key, value)
        if filename == 'updates.csv':
            assert [int(r['samples']) for r in rows] == [5, 5, 5, 5, 1]
        if filename == 'eval_endpoints.csv':
            assert len({(r['episode'], r['seed']) for r in rows}) == 10
            assert {int(r['step']) for r in rows} == {3}
    for filename in ('eval_metrics.csv', 'trajectories.csv', 'field_snapshot.npz'):
        assert not (directory / filename).exists()
    plot_results(directory)
    assert (directory / 'figures/training.png').exists()
    assert (directory / 'figures/training_evaluation.png').exists()
    assert not list((directory / 'figures').glob('*.pdf'))
    import pytest
    with pytest.raises(FileExistsError):
        train(config, directory)


def test_evaluation_does_not_consume_training_rng(config):
    config = merge(config, {'env': {'max_steps': 2}, 'evaluation': {'seeds': [1000]}})
    fields = prepare_fields(config)
    scaler = ObservationScaler.from_training_field(fields['zhurong'], config['env']['grid_size'], 2.)
    seed_all(4)
    agent = PPO(config['ppo'], scaler)
    before = rng_state()
    first = evaluate(fields['perseverance'], scaler, config, agent)
    second = evaluate(fields['perseverance'], scaler, config, agent)
    after = rng_state()
    assert before['python'] == after['python']
    np.testing.assert_equal(before['numpy'][1], after['numpy'][1])
    assert torch.equal(before['torch'], after['torch'])
    assert all(torch.equal(a,b) for a,b in zip(before['cuda'], after['cuda']))
    assert first[0] == second[0] and first[1] == second[1]


def test_policy_endpoint_evaluation_matches_full(config, small_field, scaler, monkeypatch):
    config = merge(config, {'env': {'grid_size': 3}, 'evaluation': {'seeds': [1000, 1001]}})
    seed_all(6)
    agent = PPO(config['ppo'], scaler)
    actions = []
    original_act = agent.act

    def act(*args, **kwargs):
        result = original_act(*args, **kwargs)
        actions.append(result[0])
        return result

    monkeypatch.setattr(agent, 'act', act)
    before = rng_state()
    full, trajectories, snapshot = evaluate(small_field, scaler, config, agent, episode=200)
    full_actions = actions.copy()
    actions.clear()
    terminal, paths, image = evaluate(small_field, scaler, config, agent, episode=200, metrics_mode='terminal')
    after = rng_state()
    assert actions == full_actions == [row['action'] for row in trajectories]
    assert len(full) == 144 and len(trajectories) == 140 and snapshot is not None
    assert terminal == [r for r in full if r['stage'] == 'step' and r['step'] == 70]
    assert paths == [] and image is None
    assert before['python'] == after['python']
    np.testing.assert_equal(before['numpy'], after['numpy'])
    assert torch.equal(before['torch'], after['torch'])
    assert all(torch.equal(a, b) for a, b in zip(before['cuda'], after['cuda']))
