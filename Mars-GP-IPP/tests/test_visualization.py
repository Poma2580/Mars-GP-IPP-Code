"""Regression checks for the published layouts, coordinate mapping and statistics."""
import hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pytest
from mars_ipp.visualization import field_maps, comparison
from mars_ipp.utils.logger import write_csv


@pytest.mark.parametrize('scenario,origin', [('perseverance', (65., 5.)), ('zhurong', (95., 10.))])
def test_field_layout_coordinates_and_trajectory(tmp_path, monkeypatch, scenario, origin):
    x, y = np.meshgrid([0., 15., 30.], [0., 15., 30.])
    points = np.column_stack((x.ravel(), y.ravel()))
    reference = np.column_stack((x.ravel() + y.ravel(), x.ravel() - y.ravel(), 2*y.ravel()))
    mean = .8 * reference
    trajectory = np.array([[0., 0.], [4., 2.], [28., 28.]])
    path = tmp_path / 'field_snapshot.npz'
    np.savez(path, points=points, reference=reference, mean=mean, trajectory=trajectory)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    figures, contours = {}, []
    original_contourf = Axes.contourf

    def contourf(ax, x, y, z, **kwargs):
        contours.append((np.array(x), np.array(y), np.array(z)))
        return original_contourf(ax, x, y, z, **kwargs)

    monkeypatch.setattr(Axes, 'contourf', contourf)
    monkeypatch.setattr(field_maps, 'save_figure', lambda fig, prefix, formats: figures.setdefault(Path(prefix).name, fig))
    try:
        field_maps.plot_snapshot(path, tmp_path / 'field_map', scenario=scenario)
        assert set(figures) == {'field_map', 'error_trajectory'}
        field_axes = figures['field_map'].axes[:6]
        error_axes = figures['error_trajectory'].axes[:3]
        assert len(figures['field_map'].axes) == 12  # Six panels and six colorbars.
        assert len(figures['error_trajectory'].axes) == 6
        for ax in field_axes:
            assert ax.get_subplotspec().get_topmost_subplotspec().get_gridspec().get_geometry() == (3, 2)
            assert not ax.lines and ax.collections[0].cmap.name == 'hot'
            assert ax.collections[0].get_alpha() == .8
        assert [ax.get_title() for ax in field_axes] == ['Br(GPR)', 'Br_predict', 'Bφ(GPR)', 'Bφ_predict', 'Bψ(GPR)', 'Bψ_predict']
        for i in range(3):
            np.testing.assert_array_equal(field_axes[2*i].collections[0].levels,
                                          field_axes[2*i+1].collections[0].levels)
        for ax in field_axes + error_axes:
            assert ax.get_xlabel() == 'Longitude (°)' and ax.get_ylabel() == 'Latitude (°)'
            np.testing.assert_allclose(ax.get_xlim(), [origin[0], origin[0]+30])
            np.testing.assert_allclose(ax.get_ylim(), [origin[1], origin[1]+30])
            np.testing.assert_allclose(np.diff(ax.get_xticks()), 5)
            np.testing.assert_allclose(np.diff(ax.get_yticks()), 5)
        for ax in error_axes:
            assert ax.get_subplotspec().get_topmost_subplotspec().get_gridspec().get_geometry() == (1, 3)
            assert ax.collections[0].cmap.name == 'cool'
            line = ax.lines[0]
            assert line.get_color() == 'red' and line.get_linestyle() == ':' and line.get_marker() == '|'
            np.testing.assert_allclose(line.get_xydata(), trajectory + origin)
            np.testing.assert_allclose(ax.collections[1].get_offsets(), [trajectory[0] + origin])
            np.testing.assert_allclose(ax.collections[2].get_offsets(), [trajectory[-1] + origin])
        assert [t.get_text() for t in error_axes[-1].get_legend().get_texts()] == ['Trajectory', 'initial position', 'final position']
        for cx, cy, _ in contours:
            np.testing.assert_array_equal(cx, x + origin[0])
            np.testing.assert_array_equal(cy, y + origin[1])
        for i in range(3):
            np.testing.assert_array_equal(contours[2*i][2], reference[:, i].reshape(3, 3))
            np.testing.assert_array_equal(contours[2*i+1][2], mean[:, i].reshape(3, 3))
            np.testing.assert_array_equal(contours[6+i][2], np.abs(reference[:, i]-mean[:, i]).reshape(3, 3))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
    finally:
        for fig in figures.values():
            plt.close(fig)


def _evaluation_rows(algorithm):
    rows = []
    for seed, base in [(1000, 1.), (1001, 3.)]:
        for stage, step, value in [('prior', 0, 999.), ('reset', 0, base), ('step', 1, 2*base)]:
            rows.append(dict(episode=200, algorithm=algorithm, seed=seed, stage=stage, step=step,
                             rmse=value, trace_current=10*value, rmse_Br=value,
                             rmse_B2=2*value, rmse_B3=3*value))
    return rows


@pytest.mark.parametrize('algorithms', [('ppo',), ('ppo', 'gp_trace')])
def test_evaluation_layout_styles_and_reset_statistics(tmp_path, monkeypatch, algorithms):
    figures = {}
    rows = [row for algorithm in algorithms for row in _evaluation_rows(algorithm)]
    write_csv(tmp_path / 'eval_metrics.csv', rows)
    original = (tmp_path / 'eval_metrics.csv').read_bytes()
    monkeypatch.setattr(comparison, 'save_figure', lambda fig, prefix, formats: figures.setdefault(Path(prefix).name, fig))
    try:
        comparison.plot_results(tmp_path)
        evaluation = figures['evaluation']
        assert len(evaluation.axes) == 2
        for ax in evaluation.axes:
            assert ax.get_subplotspec().get_gridspec().get_geometry() == (1, 2)
            for line in ax.lines:
                np.testing.assert_array_equal(line.get_xdata(), [0, 1])
        np.testing.assert_allclose(evaluation.axes[0].lines[0].get_ydata(), [2., 4.])
        np.testing.assert_allclose(evaluation.axes[1].lines[0].get_ydata(), [20., 40.])
        # Population SD is 1 and 2; prior's 999 must not enter either group.
        steps, means, stds = comparison._step_statistics(rows, algorithms[0], 'rmse')
        assert steps == [0, 1]
        np.testing.assert_allclose(means, [2., 4.])
        np.testing.assert_allclose(stds, [1., 2.])
        expected = {'component_rmse'} if len(algorithms) == 1 else {f'component_rmse_{a}' for a in algorithms}
        assert set(figures) == {'evaluation', *expected}
        for name in expected:
            fig = figures[name]
            assert len(fig.axes) == 1
            ax = fig.axes[0]
            assert len(ax.lines) == 3 and not ax.collections
            assert [line.get_color() for line in ax.lines] == ['teal', 'red', 'purple']
            assert [line.get_label() for line in ax.lines] == ['Br error', 'Bφ error', 'Bψ error']
            assert all(line.get_linestyle() == '--' and line.get_linewidth() == 1.5 for line in ax.lines)
            for i, line in enumerate(ax.lines):
                np.testing.assert_allclose(line.get_ydata(), np.array([2., 4.]) * (i+1))
        assert (tmp_path / 'eval_metrics.csv').read_bytes() == original
    finally:
        for fig in figures.values():
            plt.close(fig)


def _reward_rows():
    rewards = [1., 3.] * 50 + [4.] * 100 + [0., 0., 5., 5., 5.]
    return [dict(episode=i+1, reward=r, r_gp=r+1, r_rep=-1.) for i, r in enumerate(rewards)]


@pytest.mark.parametrize('mode,unit', [(None, 'Episode reward'),
    ('gp_trace_global_raw', 'Episode reward (nT²)'),
    ('global_raw_b', 'Episode reward (nT²)'),
    ('global_relative_rep', 'Episode reward (dimensionless)'),
    ('global_relative_mh', 'Episode reward (dimensionless)')])
def test_reward_reference_layout_blocks_and_population_sd(tmp_path, monkeypatch, mode, unit):
    from mars_ipp.visualization import learning_curves as plots
    figures = []
    monkeypatch.setattr(plots, 'save_figure', lambda fig, *args: figures.append(fig))
    rows = _reward_rows()
    original = [dict(r) for r in rows]
    means, stds = plots._training_blocks(rows)
    np.testing.assert_allclose(means[:, 0], [50.5, 150.5, 203.])
    np.testing.assert_allclose(means[:, 1], [2., 4., 3.])
    np.testing.assert_allclose(stds[:, 1], [1., 0., np.sqrt(6)])
    try:
        plots.plot_training_rewards(rows, tmp_path, mode)
        fig, = figures
        np.testing.assert_allclose(fig.get_size_inches(), [8, 7])
        assert len(fig.axes) == 2
        top, bottom = fig.axes
        assert top.get_ylabel() == unit
        assert top.get_xlabel() == bottom.get_xlabel() == 'Training episode'
        np.testing.assert_allclose(top.lines[0].get_xdata(), [50.5, 150.5, 203.])
        np.testing.assert_allclose(top.lines[0].get_ydata(), [2., 4., 3.])
        assert top.collections[0].get_alpha() == .2
        vertices = top.collections[0].get_paths()[0].vertices
        for x, mean, sd in zip([50.5, 150.5, 203.], [2., 4., 3.], [1., 0., np.sqrt(6)]):
            ys = vertices[vertices[:, 0] == x, 1]
            np.testing.assert_allclose([ys.min(), ys.max()], [mean-sd, mean+sd])
        assert [line.get_label() for line in bottom.lines] == ['gp', 'rep']
        np.testing.assert_allclose(bottom.lines[0].get_ydata(), [3., 5., 4.])
        np.testing.assert_allclose(bottom.lines[1].get_ydata(), [-1., -1., -1.])
        assert '100-episode blocks' in bottom.get_title()
        assert rows == original
    finally:
        for fig in figures:
            plt.close(fig)


def test_short_smoke_reward_block(tmp_path, monkeypatch):
    from mars_ipp.visualization import learning_curves as plots
    figures = []
    monkeypatch.setattr(plots, 'save_figure', lambda fig, *args: figures.append(fig))
    try:
        plots.plot_training_rewards(_reward_rows()[:30], tmp_path, title='SMOKE')
        fig, = figures
        np.testing.assert_allclose(fig.axes[0].lines[0].get_xydata(), [[15.5, 2.]])
        assert fig._suptitle.get_text() == 'SMOKE'
    finally:
        for fig in figures:
            plt.close(fig)


def test_training_only_replot_uses_provenance_and_preserves_other_files(tmp_path, monkeypatch):
    import json
    import yaml
    import mars_ipp.experiments.v3_pipeline as pipeline
    from mars_ipp.visualization import learning_curves as plots
    source = tmp_path / 'historical_source'
    source.mkdir()
    write_csv(source / 'train_metrics.csv', _reward_rows())
    (source / 'config.yaml').write_text('reward:\n  mode: gp_trace_global_raw\n')
    train = tmp_path / 'train'
    protected = {}
    for version, mode in [('v3_1','gp_trace_global_raw'), ('v3_2','global_raw_b'),
                          ('v3_3','global_relative_rep'), ('v3_4','global_relative_mh')]:
        aux = train/version/'others'; aux.mkdir(parents=True)
        if version == 'v3_1':
            (aux/'provenance.json').write_text(json.dumps({'source': str(source)}))
        else:
            write_csv(aux/'train_metrics.csv', _reward_rows())
            (aux/'config.yaml').write_text(yaml.safe_dump({'reward': {'mode': mode}}))
        figures = train/version/'figures'; figures.mkdir()
        (figures/'training_evaluation.png').write_bytes(b'unchanged monitor')
        (train/version/'checkpoint.pt').write_bytes(b'no model loading')
    for path in tmp_path.rglob('*'):
        if path.is_file():
            protected[path] = path.read_bytes()
    def no_inference(*args, **kwargs):
        raise AssertionError('Plotting must not invoke inference or evaluation')
    monkeypatch.setattr(pipeline, 'load_policy', no_inference)
    monkeypatch.setattr(pipeline, 'evaluate', no_inference)
    monkeypatch.setattr(pipeline, 'historical_policy', no_inference)
    pipeline.replot_training_rewards(train)
    assert all(path.read_bytes() == data for path, data in protected.items())
    assert len(list(train.glob('*/figures/training.png'))) == 4
