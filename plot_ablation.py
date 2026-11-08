# /// script
# dependencies = [
#   "matplotlib",
# ]
# ///

"""
Compare runs written by train_cifar10.py --log-dir, grouped by variant

run directories are grouped by the part of their name before '-seed', e.g.

    runs/norm-ablation/pre-seed0, pre-seed1, post-seed0, post-seed1  ->  variants 'pre' and 'post'

usage:
    python plot_ablation.py runs/norm-ablation
    python plot_ablation.py runs/norm-ablation pre post      # optional variant order, which also fixes the colors

writes accuracy.png, loss.png, grad_norm.png, overfit.png and summary.md into the given directory
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker

# style - light chart surface, recessive grid and axes, text in text colors
# variants take categorical slots in fixed order: blue, orange, aqua

SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e4e3df'

SERIES = ['#2a78d6', '#eb6834', '#1baf7a']

plt.rcParams.update({
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'font.size': 10,
    'text.color': TEXT_PRIMARY,
    'axes.labelcolor': TEXT_SECONDARY,
    'axes.titlecolor': TEXT_PRIMARY,
    'axes.titlesize': 10.5,
    'axes.edgecolor': GRID,
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.8,
    'xtick.color': TEXT_SECONDARY,
    'ytick.color': TEXT_SECONDARY,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'lines.solid_capstyle': 'round',
})

SEED_NOTE = 'thin lines: individual seeds · thick line: mean across seeds'

# data

def load_runs(root, order = ()):
    variants = defaultdict(list)

    for metrics_path in sorted(root.glob('*/metrics.json')):
        variant = metrics_path.parent.name.split('-seed')[0]
        variants[variant].append(json.loads(metrics_path.read_text()))

    assert len(variants) > 0, f'no */metrics.json found under {root}'

    if len(order) > 0:
        assert set(order) == set(variants), f'order {order} must name exactly the variants found: {sorted(variants)}'
        variants = {name: variants[name] for name in order}

    assert len(variants) <= len(SERIES), f'at most {len(SERIES)} variants can be compared on one chart'
    return dict(variants)

def ema(values, beta = 0.9):
    smoothed, avg = [], None

    for value in values:
        avg = value if avg is None else beta * avg + (1. - beta) * value
        smoothed.append(avg)

    return smoothed

def mean_curve(curves):
    length = min(len(curve) for curve in curves)
    return [statistics.fmean(curve[i] for curve in curves) for i in range(length)]

# figure helpers

def new_figure(title, subtitle, ncols = 1):
    fig, axes = plt.subplots(1, ncols, figsize = (8 if ncols == 1 else 11, 4.4), dpi = 150, squeeze = False)
    fig.suptitle(title, x = 0.06, y = 0.97, ha = 'left', fontsize = 13, fontweight = 'bold', color = TEXT_PRIMARY)
    fig.text(0.06, 0.905, subtitle, ha = 'left', va = 'top', fontsize = 9.5, color = TEXT_SECONDARY, linespacing = 1.4)

    top = 0.78 - 0.045 * subtitle.count('\n')
    fig.subplots_adjust(left = 0.08 if ncols > 1 else 0.11, right = 0.97 if ncols > 1 else 0.83, top = top, bottom = 0.13, wspace = 0.22)
    return fig, axes[0]

def plot_variants(ax, variants, get_x, get_y, smooth = False, markers = False):
    # every seed as a thin line, the mean across seeds as a thick line, one color per variant

    for color, (variant, runs) in zip(SERIES, variants.items()):
        xs = [get_x(run) for run in runs]
        ys = [get_y(run) for run in runs]

        if smooth:
            ys = [ema(y) for y in ys]

        for x, y in zip(xs, ys):
            ax.plot(x, y, color = color, linewidth = 0.9, alpha = 0.35)

        mean_y = mean_curve(ys)
        marker_kwargs = dict(marker = 'o', markersize = 4, markeredgecolor = SURFACE, markeredgewidth = 1.) if markers else dict()

        ax.plot(xs[0][:len(mean_y)], mean_y, color = color, linewidth = 2., label = f'{variant}-norm', **marker_kwargs)

def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')

# charts

def plot_accuracy(variants, out_dir):
    fig, (train_ax, test_ax) = new_figure('Accuracy: pre-norm vs post-norm', SEED_NOTE + ' · train accuracy is on augmented batches with dropout', ncols = 2)

    epoch_x = lambda run: [m['epoch'] for m in run['epochs']]

    plot_variants(train_ax, variants, epoch_x, lambda run: [100 * m['train_acc'] for m in run['epochs']], markers = True)
    plot_variants(test_ax, variants, epoch_x, lambda run: [100 * m['test_acc'] for m in run['epochs']], markers = True)

    for ax, title in ((train_ax, 'train'), (test_ax, 'test')):
        ax.set_title(title, loc = 'left')
        ax.set_xlabel('epoch')
        ax.set_ylim(20, 90)
        ax.set_xlim(0, None)

    train_ax.set_ylabel('accuracy (%)')
    test_ax.legend(loc = 'lower right', labelcolor = TEXT_PRIMARY)
    save(fig, out_dir / 'accuracy.png')

def plot_loss(variants, out_dir):
    fig, (train_ax, test_ax) = new_figure(
        'Loss: pre-norm vs post-norm',
        SEED_NOTE + '\ntrain: per step, EMA-smoothed, includes label smoothing · test: plain cross-entropy per epoch',
        ncols = 2
    )

    plot_variants(train_ax, variants, lambda run: [m['epoch'] for m in run['steps']], lambda run: [m['loss'] for m in run['steps']], smooth = True)
    plot_variants(test_ax, variants, lambda run: [m['epoch'] for m in run['epochs']], lambda run: [m['test_loss'] for m in run['epochs']], markers = True)

    for ax, title in ((train_ax, 'train'), (test_ax, 'test')):
        ax.set_title(title, loc = 'left')
        ax.set_xlabel('epoch')
        ax.set_xlim(0, None)

    train_ax.set_ylabel('loss')
    test_ax.legend(loc = 'upper right', labelcolor = TEXT_PRIMARY)
    save(fig, out_dir / 'loss.png')

def plot_grad_norm(variants, out_dir):
    fig, (ax,) = new_figure('Gradient norm: pre-norm vs post-norm', SEED_NOTE + '\nglobal gradient norm before clipping, per step, EMA-smoothed, log scale')

    plot_variants(ax, variants, lambda run: [m['epoch'] for m in run['steps']], lambda run: [m['grad_norm'] for m in run['steps']], smooth = True)

    ax.axhline(1., color = TEXT_SECONDARY, linewidth = 1., linestyle = (0, (4, 3)))
    ax.annotate('clip threshold 1.0', (ax.get_xlim()[1], 1.), xytext = (6, 0), textcoords = 'offset points', va = 'center', fontsize = 9, color = TEXT_SECONDARY, annotation_clip = False)

    ax.set_yscale('log')
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1., 1.2, 1.4, 1.6, 1.8, 2.]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel('epoch')
    ax.set_ylabel('gradient norm')
    ax.set_xlim(0, None)
    ax.legend(loc = 'upper right', labelcolor = TEXT_PRIMARY)
    save(fig, out_dir / 'grad_norm.png')

def plot_overfit(variants, out_dir):
    if not all(len(run['overfit']) > 0 for runs in variants.values() for run in runs):
        return

    fig, (ax,) = new_figure('Sanity check: overfitting a single batch', SEED_NOTE + ' · 32 images, no dropout or augmentation · log scale')

    plot_variants(ax, variants, lambda run: [m['step'] for m in run['overfit']], lambda run: [m['loss'] for m in run['overfit']])

    ax.set_yscale('log')
    ax.set_xlabel('step')
    ax.set_ylabel('loss')
    ax.set_xlim(0, None)
    ax.legend(loc = 'upper right', labelcolor = TEXT_PRIMARY)
    save(fig, out_dir / 'overfit.png')

# summary table

def write_summary(variants, out_dir):
    lines = [
        '| variant | seed | final test acc | best test acc | final test loss |',
        '|---|---|---|---|---|',
    ]

    for variant, runs in variants.items():
        finals = []

        for run in runs:
            epochs = run['epochs']
            final = epochs[-1]['test_acc']
            best = max(m['test_acc'] for m in epochs)
            finals.append(final)
            lines.append(f'| {variant}-norm | {run["config"]["seed"]} | {final:.2%} | {best:.2%} | {epochs[-1]["test_loss"]:.4f} |')

        spread = f' ± {statistics.stdev(finals):.2%}' if len(finals) > 1 else ''
        lines.append(f'| **{variant}-norm mean** | | **{statistics.fmean(finals):.2%}**{spread} | | |')

    path = out_dir / 'summary.md'
    path.write_text('\n'.join(lines) + '\n')
    print(path.read_text())

# main

def main():
    assert len(sys.argv) >= 2, 'usage: python plot_ablation.py <directory of run directories> [variant order ...]'

    root = Path(sys.argv[1])
    variants = load_runs(root, order = sys.argv[2:])

    plot_accuracy(variants, root)
    plot_loss(variants, root)
    plot_grad_norm(variants, root)
    plot_overfit(variants, root)
    write_summary(variants, root)

if __name__ == '__main__':
    main()
