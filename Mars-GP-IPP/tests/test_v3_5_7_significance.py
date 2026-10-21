"""Statistics and pairing regressions, independent of PPO and ROS runtimes."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('significance', SCRIPTS / 'analyze_v3_5_7_significance.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def samples():
    return [dict(map_id='test1', seed=seed, method=method, rmse=value, trr=2*value,
                 repeated_visits=0., source_file=f'{method}.csv', source_line=seed-998)
            for seed in (1000, 1001)
            for method, value in [(analysis.OURS, seed-999.),
                                  *[(label, seed-997.) for label in analysis.METHODS.values()]]]


def metric_rows():
    rows = []
    for seed in (1000, 1001):
        start = np.random.default_rng(seed).integers(0, 16, 2)
        for stage, step in [('prior', 0), ('reset', 0)] + [('step', n) for n in range(1, 71)]:
            rows.append(dict(seed=str(seed), stage=stage, step=str(step), algorithm='ppo',
                             rmse='1', trr='50', trace_prior='100', trace_current='50',
                             online_noise_variance='1e-6', samples='1', unique_positions='1', repeated_visits='0',
                             start_i=str(start[0]), start_j=str(start[1]), episode='10000'))
    return rows


def test_pairing_is_keyed_and_preserves_source_rows():
    original = samples()
    shuffled = original.copy()
    np.random.default_rng(7).shuffle(shuffled)
    paired = analysis.pair_samples(shuffled)
    assert paired == analysis.pair_samples(original)
    assert len(paired) == 8
    assert all(row['rmse_difference'] == -2 for row in paired)
    assert all(row['trr_difference'] == -4 for row in paired)
    assert paired[0]['ours_source_file'] == 'V3.5.7.csv'
    assert paired[0]['ours_source_line'] == 2


@pytest.mark.parametrize('change,match', [
    (lambda rows: rows + [rows[0]], 'Duplicate sample'),
    (lambda rows: rows[1:], 'Missing ours'),
    (lambda rows: [r for r in rows if not (r['seed'] == 1000 and r['method'] == 'Msr')], 'Missing baseline'),
])
def test_reject_duplicate_and_missing_samples(change, match):
    with pytest.raises(ValueError, match=match):
        analysis.pair_samples(change(samples()))


def test_metric_index_rejects_duplicate_missing_seed_and_corruption():
    rows = metric_rows()
    indexed = analysis.index_metrics(list(reversed(rows)), 'ppo', (1000, 1001))
    assert len(indexed) == 144
    with pytest.raises(ValueError, match='Duplicate metric'):
        analysis.index_metrics(rows + [rows[0]], 'ppo', (1000, 1001))
    with pytest.raises(ValueError, match='Missing or unexpected'):
        analysis.index_metrics(rows, 'ppo', (1000, 1001, 1002))
    changed = [dict(row) for row in rows]
    changed[-1]['trr'] = '51'
    with pytest.raises(ValueError, match='Trr definition'):
        analysis.index_metrics(changed, 'ppo', (1000, 1001))
    changed[-1]['rmse'] = 'nan'
    with pytest.raises(ValueError, match='Invalid metric'):
        analysis.index_metrics(changed, 'ppo', (1000, 1001))


def test_seeded_start_and_protocol_must_match():
    config = dict(env=dict(analysis.ENV), evaluation={'seeds': list(analysis.SEEDS)})
    analysis.validate_protocol(config)
    config['env']['noise_rate'] = .01
    with pytest.raises(ValueError, match='Environment protocol'):
        analysis.validate_protocol(config)
    rows = metric_rows()
    rows[1]['start_i'] = '16'
    with pytest.raises(ValueError, match='Seed/start'):
        analysis.index_metrics(rows, 'ppo', (1000, 1001))


def test_manual_t_confidence_interval_and_scipy_agree():
    # d = 1..5: mean=3, variance=2.5, SE=sqrt(.5), t=3*sqrt(2).
    result = analysis.paired_statistics(np.arange(1., 6.), np.zeros(5))
    assert result['df'] == 4
    assert result['difference_mean'] == 3
    assert result['difference_sd'] == pytest.approx(np.sqrt(2.5))
    assert result['se'] == pytest.approx(np.sqrt(.5))
    assert result['t'] == pytest.approx(3*np.sqrt(2))
    assert result['p_raw'] == pytest.approx(.013235599563682695)
    assert result['ci95_low'] == pytest.approx(3 - 2.7764451051977987*np.sqrt(.5))
    assert result['ci95_high'] == pytest.approx(3 + 2.7764451051977987*np.sqrt(.5))


def test_holm_four_hypotheses_and_unavailable_comparison():
    rows = [dict(p_raw=p, difference_mean=-1) for p in (.001, .02, .04, .2)]
    analysis.holm(rows)
    assert [r['p_holm'] for r in rows] == pytest.approx([.004, .06, .08, .2])
    assert [r['significant_holm'] for r in rows] == [True, False, False, False]
    assert rows[0]['conclusion'] == 'V3.5.7 显著更低'
    rows[-1]['p_raw'] = None
    analysis.holm(rows)
    assert rows[-1]['p_holm'] is None
    assert rows[0]['p_holm'] == pytest.approx(.004)


def test_wilcoxon_direction_uses_signed_ranks_instead_of_mean():
    # One extreme negative difference reverses the mean but not the rank sum.
    row = analysis.wilcoxon_statistics([-1000., *range(1, 31)])
    assert row['difference_mean'] < 0
    assert row['rank_sum_positive'] == 465
    assert row['rank_sum_negative'] == 31
    rows = [dict(row) for _ in range(4)]
    analysis.holm(rows)
    assert rows[0]['significant_holm']
    assert rows[0]['conclusion'] == '正差值秩占优'
    zeros = analysis.wilcoxon_statistics(np.zeros(50))
    assert zeros['p_raw'] == 1
    assert zeros['rank_biserial'] == 0


@pytest.mark.parametrize('difference', [0., 2., -2.])
def test_constant_difference_is_explicitly_unavailable(difference):
    result = analysis.paired_statistics(np.full(50, difference), np.zeros(50))
    assert result['status'] == 'constant_difference'
    assert result['difference_mean'] == difference
    assert result['t'] is result['p_raw'] is result['ci95_low'] is result['ci95_high'] is None


def test_sources_detect_changes_and_keep_csv_line_numbers(tmp_path):
    path = tmp_path / 'source.csv'
    path.write_text('seed,rmse\n1000,1\n1001,2\n')
    sources = analysis.Sources(tmp_path)
    rows = sources.csv(path)
    assert [r['_source_line'] for r in rows] == [2, 3]
    sources.verify()
    path.write_text('seed,rmse\n1000,999\n')
    with pytest.raises(ValueError, match='Input changed'):
        sources.verify()


def test_failed_audit_writes_explanation_without_significance_tables(tmp_path, monkeypatch):
    def failure(root, sources, audit):
        audit.append(dict(map_id='test1', baseline='GP-Entropy', status='failed'))
        raise ValueError('Missing matched seed 1000')
    monkeypatch.setattr(analysis, 'audit_inputs', failure)
    # Create only the files read before the deliberately failing data audit.
    for name in ('field/field_model.py', 'gp/kernels.py', 'envs/grid_env.py', 'evaluation/evaluator.py'):
        path = tmp_path / 'src/mars_ipp' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture\n')
    (tmp_path / 'scripts').mkdir()
    for name in ('analyze_v3_5_7_significance.py', 'requirements-significance.txt'):
        (tmp_path / 'scripts' / name).write_text((SCRIPTS / name).read_text())
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='Missing matched seed'):
        analysis.run(tmp_path, output)
    assert 'Missing matched seed 1000' in (output / 'significance_report.md').read_text()
    assert (output / 'pairing_audit.csv').exists()
    assert not (output / 'paired_ttest.csv').exists()
    assert not (output / 'supplementary_250').exists()
    with pytest.raises(ValueError, match='Output must be empty'):
        analysis.run(tmp_path, output)


def test_analysis_writes_complete_outputs_and_separate_holm_families(tmp_path):
    raw = []
    for map_number in range(1, 6):
        for seed in range(1000, 1010):
            for method_number, method in enumerate((analysis.OURS, *analysis.METHODS.values())):
                value = 1. + method_number + (seed-1000)*.1 + map_number*.05
                # Vary paired differences across starts and maps.
                value += method_number * (seed-1000)**2 * .01 * map_number
                raw.append(dict(map_id=f'test{map_number}', seed=seed, method=method,
                                rmse=value, trr=value*3, repeated_visits=method_number,
                                source_file='synthetic-test-only.csv', source_line=len(raw)+2))
    output = tmp_path / 'analysis'
    results = analysis.analyze(raw, output, 10, 'python analyze.py --output-dir NEW_EMPTY_DIR')
    assert len(results) == 8
    assert all(row['n'] == 50 and row['df'] == 49 for row in results)
    for metric in analysis.METRICS:
        family = [r for r in results if r['metric'] == metric]
        expected = analysis.multipletests([r['p_raw'] for r in family], method='holm')[1]
        assert [r['p_holm'] for r in family] == pytest.approx(expected)
    assert len(list((output / 'figures').glob('*.png'))) == 8
    assert len(list((output / 'figures').glob('*.pdf'))) == 8
    with (output / 'paired_samples.csv').open() as stream:
        assert len(list(analysis.csv.DictReader(stream))) == 200
    assert '百分点' in (output / 'significance_report.md').read_text()
    assert (output / 'table_significance.tex').read_text().count('Ours lower') == 8
