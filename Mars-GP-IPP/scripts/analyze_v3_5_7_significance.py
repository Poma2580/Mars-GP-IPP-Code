"""Audit frozen evaluation artifacts, then compare paired step-70 outcomes.

No training/evaluation is performed. Only trusted, local project model pickles
are loaded, to verify that the historical and current maps are equivalent.
"""
import argparse
import csv
import importlib.metadata
import json
import math
import shlex
import sys
import warnings
from pathlib import Path

import _bootstrap
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import yaml
from statsmodels.stats.multitest import multipletests

from mars_ipp.field.field_model import FieldModel


ROOT = Path(__file__).resolve().parents[1]
OURS = 'V3.5.7'
METHODS = {'gp_trace': 'GP-Trace', 'gp_greedy': 'GP-Entropy', 'msr': 'Msr', 'cover': 'Cover'}
METRICS = ('rmse', 'trr')
SEEDS = tuple(range(1000, 1050))
RTOL = ATOL = 1e-8
ENV = dict(grid_size=16, coordinate_scale=2., evaluation_size=31,
           evaluation_extent=30., max_steps=70, noise_rate=0.,
           online_noise_variance=1e-6, unique_samples=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    number = float(value)
    require(math.isfinite(number) and number.is_integer(), f'Invalid integer: {value}')
    return int(number)


def equivalent(actual, expected, context):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
            f'{context}: invalid shape or nonfinite values')
    require(np.allclose(a, b, rtol=RTOL, atol=ATOL), f'{context}: numerical mismatch')
    return float(np.max(np.abs(a - b))) if a.size else 0.


class Sources:
    def __init__(self, root):
        self.root = root
        self.hashes = {}

    def path(self, value):
        path = Path(value)
        if path.is_absolute():
            # Historical metadata contains absolute paths from this repository.
            for name, base in (('Mars-GP-IPP', self.root), ('Mars_Env_GP', self.root.parent / 'Mars_Env_GP')):
                if name in path.parts:
                    path = base.joinpath(*path.parts[path.parts.index(name) + 1:])
                    break
        else:
            path = self.root / path
        return path.resolve()

    def track(self, path):
        import hashlib
        path = self.path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key = str(path)
        require(key not in self.hashes or self.hashes[key] == digest, f'Input changed: {path}')
        self.hashes[key] = digest
        return path

    def json(self, path):
        return json.loads(self.track(path).read_text())

    def config(self, path):
        return yaml.safe_load(self.track(path).read_text())

    def csv(self, path):
        path = self.track(path)
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            records = []
            for row in reader:
                row['_source_file'] = str(path)
                row['_source_line'] = reader.line_num
                records.append(row)
            return records

    def verify(self):
        for name in list(self.hashes):
            self.track(name)


def validate_protocol(config):
    require(config['env'] == ENV, 'Environment protocol mismatch')
    require(tuple(config['evaluation']['seeds']) == SEEDS, 'Evaluation seed protocol mismatch')


def index_metrics(rows, algorithm, seeds=SEEDS):
    """Index by seed/stage/step; never use CSV row ordering or episode IDs to pair."""
    indexed = {}
    for row in rows:
        seed, step = integer(row['seed']), integer(row['step'])
        key = seed, row['stage'], step
        require(key not in indexed, f'Duplicate metric key: {key}')
        require(row['algorithm'] == algorithm, f'Algorithm mismatch: {row["algorithm"]}')
        values = [float(row[k]) for k in ('rmse', 'trr', 'trace_prior', 'trace_current')]
        require(np.isfinite(values).all() and min(values) >= 0 and values[2] > 0,
                f'Invalid metric: {key}')
        require(integer(row['repeated_visits']) >= 0, f'Invalid repeated visits: {key}')
        equivalent(values[1], values[3] / values[2] * 100., f'Trr definition {key}')
        equivalent(float(row['online_noise_variance']), ENV['online_noise_variance'], 'GP noise')
        require(integer(row['samples']) == integer(row['unique_positions']), 'Unique-sample protocol mismatch')
        indexed[key] = row
    expected = {(seed, stage, step) for seed in seeds
                for stage, step in [('prior', 0), ('reset', 0)] + [('step', n) for n in range(1, 71)]}
    require(set(indexed) == expected, f'Missing or unexpected metric keys: missing={len(expected - set(indexed))}, extra={len(set(indexed) - expected)}')
    for seed in seeds:
        initial = indexed[seed, 'reset', 0]
        start = tuple(integer(initial[k]) for k in ('start_i', 'start_j'))
        require(start == tuple(np.random.default_rng(seed).integers(0, 16, 2)), f'Seed/start mismatch: {seed}')
        episodes = set()
        for stage, step in [('prior', 0), ('reset', 0)] + [('step', n) for n in range(1, 71)]:
            row = indexed[seed, stage, step]
            episodes.add(integer(row['episode']))
            require(tuple(integer(row[k]) for k in ('start_i', 'start_j')) == start, 'Start changes within episode')
        require(len(episodes) == 1, f'Multiple evaluation episodes for seed {seed}')
    return indexed


def parameters_equal(a, b, context):
    require(a['type'] == b['type'] and a['nu'] == b['nu'], f'{context}: kernel type mismatch')
    return equivalent([a[k] for k in ('variance', 'lengthscale', 'noise')],
                      [b[k] for k in ('variance', 'lengthscale', 'noise')], context)


def validate_trajectories(rows, truth_grid):
    keys = set()
    observed, truth = [], []
    for row in rows:
        key = integer(row['seed']), integer(row['step'])
        require(key not in keys, f'Duplicate trajectory key: {key}')
        keys.add(key)
        i, j = integer(row['i']), integer(row['j'])
        require(0 <= i < 16 and 0 <= j < 16, 'Trajectory position outside map')
        observed.append([float(row[k]) for k in ('Br', 'B2', 'B3')])
        truth.append(truth_grid[i, j])
    require(keys == {(s, n) for s in SEEDS for n in range(1, 71)}, 'Incomplete trajectory')
    return equivalent(observed, truth, 'Noiseless trajectory measurements')


def audit_inputs(root, sources, audit):
    comparison = root / 'results/comparison'
    ours_root = comparison / 'v3.5.5_v.3.5.10_test'
    metadata = sources.json(ours_root / 'others/metadata.json')
    validate_protocol({**metadata['configs']['V3_5_7'], 'evaluation': {'seeds': metadata['seeds']}})
    require(not metadata['smoke'], 'Smoke data are not formal evaluation data')
    entropy_path = sources.track(root / 'src/mars_ipp/agents/baselines/gp_entropy.py')
    sources.track(root / 'src/mars_ipp/agents/baselines/gp_greedy.py')
    require('class GPEntropy(GPGreedy)' in entropy_path.read_text(), 'GP-Entropy alias requires review')
    baseline_sources = sources.json(comparison / 'bnorm_no_mh_local_global_3x3_rbf/perseverance/summary/sources.json')['baseline_sources']
    raw = []
    points = np.array([(i, j) for i in range(31) for j in range(31)], dtype=float)
    grid = np.array([(i, j) for i in range(16) for j in range(16)], dtype=float) * 2
    for number in range(1, 6):
        scene = f'test{number}'
        field_spec = metadata['fields'][scene]
        model_path = sources.track(field_spec['path'])
        require(sources.hashes[str(model_path)] == field_spec['sha256'], f'{scene}: field hash mismatch')
        field = FieldModel.load(model_path)
        parameters_equal(field.parameters, field_spec['parameters'], f'{scene}: recorded field parameters')
        truth = field.truth(points)
        truth_grid = field.truth(grid).reshape(16, 16, 3)
        current = ours_root / scene / 'others' / OURS
        ours = index_metrics(sources.csv(current / 'eval_metrics.csv'), 'ppo')
        ours_measurement_delta = validate_trajectories(sources.csv(current / 'trajectories.csv'), truth_grid)
        checkpoint = metadata['provenance'][scene][OURS]
        checkpoint_path = sources.track(checkpoint['checkpoint'])
        require(sources.hashes[str(checkpoint_path)] == checkpoint['sha256'], 'V3.5.7 checkpoint hash mismatch')
        methods = {OURS: ours}
        for algorithm, label in METHODS.items():
            entry = dict(map_id=scene, baseline=label, status='failed')
            audit.append(entry)
            directory = sources.path(baseline_sources[algorithm]) if number == 1 else comparison / f'test_maps_seed0/{scene}/{algorithm}'
            config = sources.config(directory / 'config.yaml')
            validate_protocol(config)
            require(config['evaluation'].get('cover_random_start') is True, 'Random-start protocol mismatch')
            base_meta = sources.json(directory / 'metadata.json')
            require(base_meta['algorithm'] == algorithm, 'Baseline metadata identity mismatch')
            base_scene = 'perseverance' if number == 1 else scene
            require(config['field']['test_scenario'] == base_scene, 'Baseline map identity mismatch')
            spec = base_meta['fields'][base_scene] if number == 1 else base_meta['field']
            base_model_path = sources.track(config['field']['models'][base_scene])
            require(sources.hashes[str(base_model_path)] == spec['sha256'], 'Historical field hash mismatch')
            base_field = FieldModel.load(base_model_path)
            parameters_equal(base_field.parameters, spec['parameters'], 'Historical recorded kernel parameters')
            entry['kernel_max_abs_delta'] = parameters_equal(base_field.parameters, field.parameters, 'Paired kernel parameters')
            entry['map_truth_max_abs_delta'] = equivalent(base_field.truth(points), truth, 'Paired map truth')
            with np.load(sources.track(directory / 'field_snapshot.npz')) as snapshot:
                require(np.unique(snapshot['points'], axis=0).shape[0] == 961, 'Incomplete evaluation grid')
                # Snapshot ordering is retained when predicting the reference.
                require(set(map(tuple, snapshot['points'])) == set(map(tuple, points)), 'Evaluation grid mismatch')
                entry['snapshot_truth_max_abs_delta'] = equivalent(snapshot['reference'], field.truth(snapshot['points']), 'Snapshot reference truth')
            baseline = index_metrics(sources.csv(directory / 'eval_metrics.csv'), algorithm)
            entry['measurement_max_abs_delta'] = validate_trajectories(sources.csv(directory / 'trajectories.csv'), truth_grid)
            entry['ours_measurement_max_abs_delta'] = ours_measurement_delta
            initial_columns = ('start_i', 'start_j', 'rmse', 'trr', 'trace_prior', 'trace_current')
            entry['initial_max_abs_delta'] = equivalent(
                [[float(ours[s, stage, 0][k]) for k in initial_columns] for s in SEEDS for stage in ('prior', 'reset')],
                [[float(baseline[s, stage, 0][k]) for k in initial_columns] for s in SEEDS for stage in ('prior', 'reset')],
                f'{scene}/{label}: initial conditions')
            entry.update(status='passed', available_pairs=50, main_pairs=10, noise_rate=0.,
                         source_directory=str(directory), map_hash_identical=field_spec['sha256'] == spec['sha256'])
            methods[label] = baseline
        for label, records in methods.items():
            for seed in SEEDS:
                row = records[seed, 'step', 70]
                raw.append(dict(map_id=scene, seed=seed, method=label, algorithm=row['algorithm'],
                                episode=integer(row['episode']), step=70,
                                **{k: float(row[k]) for k in ('rmse', 'trr', 'repeated_visits', 'start_i', 'start_j')},
                                source_file=row['_source_file'], source_line=row['_source_line']))
    return raw


def pair_samples(raw):
    indexed = {}
    for row in raw:
        key = row['map_id'], row['seed'], row['method']
        require(key not in indexed, f'Duplicate sample: {key}')
        indexed[key] = row
    keys = sorted({(r['map_id'], r['seed']) for r in raw})
    result = []
    for scene, seed in keys:
        require((scene, seed, OURS) in indexed, f'Missing ours sample: {scene}/{seed}')
        ours = indexed[scene, seed, OURS]
        for label in METHODS.values():
            require((scene, seed, label) in indexed, f'Missing baseline sample: {scene}/{seed}/{label}')
            baseline = indexed[scene, seed, label]
            row = dict(map_id=scene, seed=seed, baseline=label,
                       ours_source_file=ours['source_file'], ours_source_line=ours['source_line'],
                       baseline_source_file=baseline['source_file'], baseline_source_line=baseline['source_line'])
            for metric in (*METRICS, 'repeated_visits'):
                row[f'{metric}_ours'] = ours[metric]
                row[f'{metric}_baseline'] = baseline[metric]
                row[f'{metric}_difference'] = ours[metric] - baseline[metric]
            result.append(row)
    return result


def paired_statistics(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    require(x.shape == y.shape and x.ndim == 1 and x.size >= 3, 'Invalid paired sample shape')
    require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite paired sample')
    d = x - y
    n, mean, sd = d.size, float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    # A zero-variance difference does not provide a valid estimated standard error.
    if sd == 0:
        return dict(n=n, df=n-1, difference_mean=mean, difference_sd=sd, se=se,
                    t=None, p_raw=None, ci95_low=None, ci95_high=None, status='constant_difference')
    result = stats.ttest_rel(x, y, alternative='two-sided')
    equivalent(result.statistic, mean / se, 'Manual versus SciPy t')
    margin = float(stats.t.ppf(.975, n-1) * se)
    return dict(n=n, df=n-1, difference_mean=mean, difference_sd=sd, se=se,
                t=float(result.statistic), p_raw=float(result.pvalue),
                ci95_low=mean-margin, ci95_high=mean+margin, status='ok')


def wilcoxon_statistics(differences):
    d = np.asarray(differences, dtype=float)
    require(d.ndim == 1 and len(d) >= 3 and np.isfinite(d).all(), 'Invalid Wilcoxon differences')
    nonzero = d[d != 0]
    absolute_ranks = stats.rankdata(np.abs(nonzero))
    positive = float(absolute_ranks[nonzero > 0].sum())
    negative = float(absolute_ranks[nonzero < 0].sum())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if len(nonzero) == 0:
            statistic, pvalue = 0., 1.
        else:
            result = stats.wilcoxon(d, alternative='two-sided', zero_method='wilcox', method='auto', correction=False)
            statistic, pvalue = float(result.statistic), float(result.pvalue)
    return dict(n=len(d), difference_mean=float(d.mean()), difference_median=float(np.median(d)),
                statistic=statistic, p_raw=pvalue, rank_sum_positive=positive, rank_sum_negative=negative,
                rank_biserial=(positive-negative)/(positive+negative) if len(nonzero) else 0.,
                zero_count=int((d == 0).sum()), absolute_tie_count=len(nonzero)-len(np.unique(np.abs(nonzero))),
                zero_method='wilcox', method='auto', correction=False,
                warnings='; '.join(str(w.message) for w in caught))


def holm(results):
    """Always retain four planned hypotheses, including unavailable tests."""
    require(len(results) == 4, 'Holm family must contain four baseline comparisons')
    corrected = multipletests([r['p_raw'] if r['p_raw'] is not None else 1. for r in results],
                             alpha=.05, method='holm')[1]
    for row, adjusted in zip(results, corrected):
        available = row['p_raw'] is not None
        row['p_holm'] = float(adjusted) if available else None
        row['significant_holm'] = bool(available and adjusted < .05)
        row['conclusion'] = ('无法检验' if not available else
                             '差异未达显著' if adjusted >= .05 else
                             'V3.5.7 显著更低' if row['difference_mean'] < 0 else 'Baseline 显著更低')
        if row['significant_holm'] and 'rank_sum_positive' in row:
            # Means can have the opposite sign from the signed-rank direction.
            row['conclusion'] = ('负差值秩占优' if row['rank_sum_negative'] > row['rank_sum_positive']
                                 else '正差值秩占优')


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def fmt(value):
    return 'NA' if value is None else f'{value:.6g}'


def analyze(raw, output, seeds_per_map, command):
    output.mkdir(parents=True, exist_ok=True)
    selected = [r for r in raw if 1000 <= r['seed'] < 1000 + seeds_per_map]
    paired = pair_samples(selected)
    expected = 5 * seeds_per_map
    require(len(selected) == expected * 5 and len(paired) == expected * 4, 'Unexpected complete sample counts')
    write_csv(output / 'raw_step70.csv', selected)
    write_csv(output / 'paired_samples.csv', paired)
    tests, ranks, diagnostics, outliers, by_map, descriptive = [], [], [], [], [], []
    figures = output / 'figures'
    figures.mkdir()
    for metric in METRICS:
        family, rank_family = [], []
        for label in METHODS.values():
            samples = [r for r in paired if r['baseline'] == label]
            x = np.array([r[f'{metric}_ours'] for r in samples])
            y = np.array([r[f'{metric}_baseline'] for r in samples])
            d = x - y
            require(len(samples) == expected, 'Incomplete baseline comparison')
            row = dict(metric=metric, baseline=label, **paired_statistics(x, y))
            family.append(row)
            rank_family.append(dict(metric=metric, baseline=label, **wilcoxon_statistics(d)))
            shapiro = stats.shapiro(d) if row['status'] == 'ok' else None
            q1, q3 = np.percentile(d, [25, 75])
            lower, upper = q1-3*(q3-q1), q3+3*(q3-q1)
            extreme = (d < lower) | (d > upper)
            diagnostics.append(dict(metric=metric, baseline=label, n=len(d),
                                    shapiro_w=float(shapiro.statistic) if shapiro else None,
                                    shapiro_p=float(shapiro.pvalue) if shapiro else None,
                                    status=row['status'], outlier_lower=float(lower), outlier_upper=float(upper),
                                    extreme_count=int(extreme.sum()), zero_count=int((d == 0).sum())))
            for sample, value, flag in zip(samples, d, extreme):
                if flag:
                    outliers.append(dict(metric=metric, baseline=label, map_id=sample['map_id'], seed=sample['seed'],
                                         difference=float(value), lower=float(lower), upper=float(upper)))
            for scene in sorted({r['map_id'] for r in samples}):
                values = np.array([r[f'{metric}_difference'] for r in samples if r['map_id'] == scene])
                by_map.append(dict(map_id=scene, metric=metric, baseline=label, n=len(values),
                                   difference_mean=float(values.mean()), difference_sd=float(values.std(ddof=1))))
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
            axes[0].hist(d, bins='auto', edgecolor='white')
            axes[0].axvline(0, color='black', linestyle='--', linewidth=1)
            axes[0].set(xlabel=f'Difference in {metric} (ours - baseline)', ylabel='Count')
            if row['status'] == 'ok':
                stats.probplot(d, dist='norm', plot=axes[1])
            else:
                axes[1].text(.5, .5, 'Constant differences; Q-Q diagnostic undefined', ha='center', wrap=True)
            fig.suptitle(f'{OURS} vs {label}: {metric}, n={len(d)}')
            fig.tight_layout()
            for suffix in ('png', 'pdf'):
                fig.savefig(figures / f'{metric}_{label.lower().replace("-", "_")}.{suffix}', dpi=180)
            plt.close(fig)
        holm(family)
        holm(rank_family)
        tests.extend(family)
        ranks.extend(rank_family)
    for label in (OURS, *METHODS.values()):
        for metric in (*METRICS, 'repeated_visits'):
            values = np.array([r[metric] for r in selected if r['method'] == label])
            descriptive.append(dict(method=label, metric=metric, n=len(values), mean=float(values.mean()),
                                    sd=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max())))
    write_csv(output / 'paired_ttest.csv', tests)
    write_csv(output / 'paired_wilcoxon.csv', ranks)
    write_csv(output / 'diagnostics.csv', diagnostics)
    write_csv(output / 'extreme_outliers.csv', outliers,
              ['metric', 'baseline', 'map_id', 'seed', 'difference', 'lower', 'upper'])
    write_csv(output / 'per_map_differences.csv', by_map)
    write_csv(output / 'descriptive_statistics.csv', descriptive)
    tex = [r'\begin{tabular}{llrrrrrl}', r'\hline',
           r'Metric & Baseline & $n$ & Mean diff. & 95\% CI & $p$ & $p_{\mathrm{Holm}}$ & Conclusion \\', r'\hline']
    for row in tests:
        conclusion = ('NA' if row['p_holm'] is None else 'NS' if not row['significant_holm'] else
                      'Ours lower' if row['difference_mean'] < 0 else 'Baseline lower')
        interval = f'[{fmt(row["ci95_low"])}, {fmt(row["ci95_high"])}]'
        tex.append(f'{row["metric"].upper()} & {row["baseline"]} & {row["n"]} & {fmt(row["difference_mean"])} & {interval} & {fmt(row["p_raw"])} & {fmt(row["p_holm"])} & {conclusion}' + r' \\')
    tex += [r'\hline', r'\end{tabular}',
            '% Difference = V3.5.7 - baseline. Trr differences are percentage points.',
            '% Two-sided paired t-tests; unadjusted 95% CIs. Holm within each metric (4 tests).']
    (output / 'table_significance.tex').write_text('\n'.join(tex) + '\n')
    lines = [f'# V3.5.7 显著性分析（{expected} 对）', '',
             f'每图固定取 seed 1000–{999+seeds_per_map}，五张固定地图，共 {expected} 对；df={expected-1}。',
             '250 对结果属于补充敏感性分析，不替代预定的 50 对主分析。', '',
             '## 方法与结论', '',
             '差值 = V3.5.7 − Baseline。RMSE 单位 nT；Trr 原值单位 %，差值单位为百分点；两项均越小越好。',
             '双侧配对 t 检验，α=0.05；RMSE、Trr 各有四项比较并分别作 Holm 校正。95% CI 为未经多重比较校正的差值均值置信区间。', '',
             '|指标|Baseline|n|差值均值|差值 SD|SE|t|df|95% CI|原始 p|Holm p|校正后结论|',
             '|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|']
    for row in tests:
        lines.append('|'+ '|'.join([row['metric'], row['baseline'], str(row['n']), fmt(row['difference_mean']),
                                    fmt(row['difference_sd']), fmt(row['se']), fmt(row['t']), str(row['df']),
                                    f'[{fmt(row["ci95_low"])}, {fmt(row["ci95_high"])}]', fmt(row['p_raw']),
                                    fmt(row['p_holm']), row['conclusion']])+'|')
    lines += ['', '## 差值诊断与稳健性', '',
              '直方图和 Q-Q 图见 figures/。Shapiro–Wilk 仅作诊断，不能证明正态性，也不据其 p 值机械切换检验。',
              '极端值定义为低于 Q1−3×IQR 或高于 Q3+3×IQR，详见 extreme_outliers.csv；全部保留。', '',
              '|指标|Baseline|Shapiro p|极端值数量|Wilcoxon 原始 p|Wilcoxon Holm p|校正后符号秩结论|',
              '|---|---|---:|---:|---:|---:|---|']
    for diag, rank in zip(diagnostics, ranks):
        lines.append(f'|{diag["metric"]}|{diag["baseline"]}|{fmt(diag["shapiro_p"])}|{diag["extreme_count"]}|{fmt(rank["p_raw"])}|{fmt(rank["p_holm"])}|{rank["conclusion"]}|')
    flagged = [f'{d["metric"]}/{d["baseline"]}' for d in diagnostics if d['shapiro_p'] is not None and d['shapiro_p'] < .05]
    lines += ['', f'Shapiro p<0.05 的比较：{", ".join(flagged) or "无"}。这些结果提示正态性问题，应结合图形、极端值及地图异质性解释 t 检验。',
              'Wilcoxon 对全部比较统一运行，独立按指标作 Holm 校正；零差值使用 wilcox 规则排除，method=auto，不作连续性校正。',
              'Wilcoxon 同样依赖配对间独立性；位置差异解释还需差值分布对称，不是无条件解决异质性的替代方法。',
              '符号秩方向按正、负差值的秩和判定，不能用差值均值方向代替；正差值秩占优表示正方向秩更大，负差值秩占优表示负方向秩更大。',
              '恒定差值的 t 检验标记为无法检验，置信区间留空；若出现，仍保留该项在四项 Holm 检验族中。', '',
              '## 数据来源、配对与局限', '',
              '按 (map_id, seed) 配对，不使用 episode 编号或行顺序；paired_samples.csv 保留双方原始文件与行号。',
              'GP-Entropy 对应历史 gp_greedy：单点熵是方差的单调函数，GPEntropy 继承 GPGreedy 的动作选择及并列处理。',
              'test1 对应历史 perseverance；来源由 sources.json 追溯。地图哈希可不同，必须通过真值、核参数和初始状态的数值等价检查。',
              '核验容差 rtol=atol=1e-8，地图参考网格 31×31、动作网格 16×16，观测噪声为零；在线 GP 噪声方差 1e-6，重复位置仅纳入一次 GP。',
              '配对审计和源文件哈希见主目录 pairing_audit.csv、source_manifest.json；依赖版本见 environment.json。',
              '不同地图差值可能尺度和分布不同；per_map_differences.csv 提供分地图描述。合并 t 检验按回合差值独立、近似同分布计算，固定地图混合可能削弱这些假设。',
              '不同地图复用相同 seed，起点相同，可能引入跨地图相关；250 对并不代表 250 个独立地图或训练重复。',
              '结论仅针对这些固定地图和固定训练 checkpoint，不能推断对新地图或多次独立训练普遍显著。',
              '主分析取预定 seed 1000–1009，其余 40 个 seed/图仅用于补充；不按结果选择样本。',
              '重复采样次数仅在 descriptive_statistics.csv 中描述，未作显著性检验。', '',
              '## 复现', '', '在无 conda 的系统 Python venv 中安装 scripts/requirements-significance.txt，然后运行：',
              '', '输出目录必须为空；复跑时请将下列 --output-dir 改为新的空目录。',
              '', '```bash', command, '```', '']
    discrepancies = [f'{t["metric"]}/{t["baseline"]}' for t, w in zip(tests, ranks)
                     if t['significant_holm'] != w['significant_holm']]
    mixed_maps = []
    for metric in METRICS:
        for label in METHODS.values():
            means = [r['difference_mean'] for r in by_map if r['metric'] == metric and r['baseline'] == label]
            if min(means) < 0 < max(means):
                mixed_maps.append(f'{metric}/{label}')
    interpretation = ['', '### 诊断对结论的影响', '',
                      f't 与 Wilcoxon 校正后显著性不一致的比较：{", ".join(discrepancies) or "无"}。',
                      f'分地图差值均值方向不一致的比较：{", ".join(mixed_maps) or "无"}。',
                      '分地图均值符号变化反映实际地图异质性；极端值标记不等同于记录错误。',
                      '若 t 显著而符号秩不显著，应表述为均值差异在配对 t 检验下显著，稳健性证据不足；两种检验针对的统计特征不同。',
                      '结合差值分布、地图异质性及检验间分歧，不能据汇总结果声称在所有地图、指标上全面显著优于全部基线。', '']
    position = lines.index('## 数据来源、配对与局限')
    lines[position:position] = interpretation
    (output / 'significance_report.md').write_text('\n'.join(lines))
    print(f'\n{output}: {expected} pairs/baseline')
    for row in tests:
        print(f'{row["metric"]:4s} {row["baseline"]:10s} n={row["n"]} mean(d)={fmt(row["difference_mean"])} '
              f'SD={fmt(row["difference_sd"])} SE={fmt(row["se"])} t={fmt(row["t"])} df={row["df"]} '
              f'CI=[{fmt(row["ci95_low"])}, {fmt(row["ci95_high"])}] '
              f'p={fmt(row["p_raw"])} Holm={fmt(row["p_holm"])} {row["conclusion"]}')
    return tests


def run(project_root, output_dir):
    root, output = Path(project_root).resolve(), Path(output_dir).resolve()
    # Fresh output avoids mixing a previous successful analysis with a failed audit.
    require(not output.exists() or not any(output.iterdir()), f'Output must be empty: {output}')
    require(not root.is_relative_to(output), 'Output may not contain the project root')
    output.mkdir(parents=True, exist_ok=True)
    sources, audit = Sources(root), []
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), '--project-root', str(root), '--output-dir', str(output)])
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy', 'scipy', 'statsmodels', 'matplotlib', 'GPy', 'pandas', 'PyYAML', 'paramz')}
    write_json(output / 'environment.json', dict(python=sys.version, executable=sys.executable,
                                                base_executable=getattr(sys, '_base_executable', None),
                                                versions=versions, command=command))
    try:
        for path in (Path(__file__), Path(__file__).with_name('requirements-significance.txt'),
                     root / 'src/mars_ipp/field/field_model.py', root / 'src/mars_ipp/gp/kernels.py',
                     root / 'src/mars_ipp/envs/grid_env.py', root / 'src/mars_ipp/evaluation/evaluator.py'):
            sources.track(path)
        raw = audit_inputs(root, sources, audit)
        sources.verify()
    except Exception as error:
        write_csv(output / 'pairing_audit.csv', audit, list(dict.fromkeys(['map_id', 'baseline', 'status'] + [k for r in audit for k in r])))
        write_json(output / 'source_manifest.json', dict(status='failed', error=str(error), sha256=sources.hashes))
        (output / 'significance_report.md').write_text(f'# 配对审计未通过\n\n{type(error).__name__}: {error}\n\n未生成显著性检验结论。\n')
        raise
    write_csv(output / 'pairing_audit.csv', audit)
    write_json(output / 'source_manifest.json', dict(status='audit_passed', rtol=RTOL, atol=ATOL, sha256=sources.hashes))
    # Stage the analysis inside the output so a failed final hash check cannot
    # leave apparently publishable tables in their final locations.
    import shutil
    import tempfile
    staging = Path(tempfile.mkdtemp(prefix='.analysis-', dir=output))
    try:
        analyze(raw, staging / 'main', 10, command)
        analyze(raw, staging / 'main/supplementary_250', 50, command)
        sources.verify()
        for path in (staging / 'main').iterdir():
            shutil.move(str(path), output / path.name)
    except Exception as error:
        (output / 'significance_report.md').write_text(f'# 分析未完成\n\n{type(error).__name__}: {error}\n\n未发布显著性检验结论。\n')
        write_json(output / 'source_manifest.json', dict(status='failed', error=str(error), sha256=sources.hashes))
        raise
    finally:
        shutil.rmtree(staging)
    write_json(output / 'source_manifest.json', dict(status='complete', original_inputs_unchanged=True,
                                                  rtol=RTOL, atol=ATOL, sha256=sources.hashes))
    print(f'\nAudit passed: {len(audit)} map/baseline combinations; {len(sources.hashes)} unchanged source files.\nResults: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    run(args.project_root, args.output_dir or args.project_root / 'results/comparison/v3_5_7_significance')
