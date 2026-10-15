import json
import pytest
import numpy as np
from mars_ipp.evaluation.statistics import summarize
from mars_ipp.visualization.aggregate import aggregate_results
from mars_ipp.utils.logger import write_csv


def rows(algorithm='ppo'):
    return [dict(episode=40, step=70, stage='step', seed=seed, algorithm=algorithm,
                 rmse=value, trace_current=10*value, rmse_Br=value, rmse_B2=value, rmse_B3=value)
            for seed, value in [(1000, 1.), (1001, 3.)]]


def test_population_standard_deviation_and_final_selection():
    data = rows() + [dict(rows()[0], episode=20, rmse=100)]
    summary = summarize(data)[0]
    assert summary['runs'] == 2 and summary['rmse_mean'] == 2
    assert summary['rmse_std'] == 1 and summary['trace_current_std'] == 10


def test_aggregate_preserves_input_and_rejects_duplicate_runs(tmp_path):
    inputs = []
    for seed in (0, 1):
        directory = tmp_path / f'seed_{seed}'
        directory.mkdir()
        (directory / 'config.yaml').write_text(f'experiment: test_fixture\nseed: {seed}\n')
        write_csv(directory / 'eval_metrics.csv', rows())
        inputs.append(directory)
    original = [p.joinpath('eval_metrics.csv').read_bytes() for p in inputs]
    output = tmp_path / 'aggregate'
    aggregate_results(inputs, output)
    assert (output / 'figures/evaluation.png').exists()
    sources = json.loads((output / 'sources.json').read_text())
    assert [s['training_seed'] for s in sources] == [0, 1]
    assert original == [p.joinpath('eval_metrics.csv').read_bytes() for p in inputs]
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate_results([inputs[0], inputs[0]], tmp_path / 'duplicate')


def test_csv_append_schema_validation(tmp_path):
    import csv
    import pytest
    from mars_ipp.utils.logger import append_csv
    path = tmp_path / 'records.csv'
    append_csv(path, [{'episode': 1, 'reward': 2.}])
    append_csv(path, [{'episode': 2, 'reward': 3.}])
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r['episode'] for r in rows] == ['1', '2']
    original = path.read_bytes()
    with pytest.raises(ValueError, match='header'):
        append_csv(path, [{'episode': 3, 'loss': 4.}])
    with pytest.raises(ValueError, match='fields'):
        append_csv(path, [{'episode': 3, 'reward': 4.}, {'episode': 4}])
    assert path.read_bytes() == original


def test_aggregate_requires_formal_evaluation(tmp_path):
    import pytest
    from mars_ipp.visualization.aggregate import aggregate_results
    training = tmp_path / 'training'
    training.mkdir()
    output = tmp_path / 'aggregate'
    with pytest.raises(ValueError, match='formal evaluation'):
        aggregate_results([training], output)
    assert not output.exists()
