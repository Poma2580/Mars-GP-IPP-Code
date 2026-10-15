"""Paper-style field comparison and error/trajectory figures from saved arrays."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml
from mars_ipp.field.dataset import SCENARIOS
from .learning_curves import save_figure

DISPLAY_COMPONENTS = ('Br', 'Bφ', 'Bψ')


def _levels(low, high):
    if low == high:
        high = low + 1.
    return np.linspace(low, high, 11)


def _field_axes(ax, x, y):
    ax.set(xlabel='Longitude (°)', ylabel='Latitude (°)',
           xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))
    ax.set_xticks(np.arange(np.ceil(x.min() / 5) * 5, x.max() + .01, 5))
    ax.set_yticks(np.arange(np.ceil(y.min() / 5) * 5, y.max() + .01, 5))
    ax.set_aspect('equal')
    ax.grid(color='gray', linestyle='--', linewidth=1., alpha=.7)


def plot_snapshot(path, output, formats=('png',), scenario=None):
    """Use absolute scenario coordinates without changing saved local arrays."""
    path, output = Path(path), Path(output)
    if scenario is None:
        config = yaml.safe_load((path.parent / 'config.yaml').read_text())
        scenario = config['field']['test_scenario']
    origin = np.asarray(SCENARIOS[scenario]['origin'])
    with np.load(path) as snapshot:
        points, reference, mean, trajectory = (snapshot[k] for k in ('points', 'reference', 'mean', 'trajectory'))
        size = int(np.sqrt(len(points)))
        coordinates = points + origin
        x = coordinates[:, 0].reshape(size, size)
        y = coordinates[:, 1].reshape(size, size)
        trajectory = trajectory + origin

        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        for component, name in enumerate(DISPLAY_COMPONENTS):
            low = min(reference[:, component].min(), mean[:, component].min())
            high = max(reference[:, component].max(), mean[:, component].max())
            levels = _levels(low, high)
            for column, (values, title) in enumerate(((reference, f'{name}(GPR)'), (mean, f'{name}_predict'))):
                ax = axes[component, column]
                surface = ax.contourf(x, y, values[:, component].reshape(size, size),
                                      levels=levels, cmap='hot', alpha=.8)
                ax.set_title(title)
                _field_axes(ax, x, y)
                fig.colorbar(surface, ax=ax, label='nT')
        save_figure(fig, output, formats)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        error = np.abs(reference - mean)
        for component, (ax, name) in enumerate(zip(axes, DISPLAY_COMPONENTS)):
            values = error[:, component].reshape(size, size)
            surface = ax.contourf(x, y, values, levels=_levels(0., values.max()),
                                  cmap='cool', alpha=.8)
            ax.set_title(f'{name} Error')
            _field_axes(ax, x, y)
            ax.plot(trajectory[:, 0], trajectory[:, 1], color='red', linestyle=':',
                    marker='|', linewidth=1.2, markersize=4, label='Trajectory')
            ax.scatter(*trajectory[0], color=plt.cm.Set1(0), s=20, marker='o',
                       label='initial position', zorder=4)
            ax.scatter(*trajectory[-1], color=plt.cm.Set1(0), s=20, marker='s',
                       label='final position', zorder=4)
            fig.colorbar(surface, ax=ax, label='nT')
        axes[-1].legend(loc='lower right', fontsize=9)
        save_figure(fig, output.parent / 'error_trajectory', formats)
