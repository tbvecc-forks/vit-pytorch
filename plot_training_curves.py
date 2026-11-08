# /// script
# dependencies = [
#   "matplotlib",
# ]
# ///

"""
Plot the curves written by train_cifar10.py --log-dir <dir>

usage:
    python plot_training_curves.py runs/cifar10-baseline

writes overfit.png, loss.png, accuracy.png and lr.png next to metrics.json
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# style - light chart surface, recessive grid and axes, text in text colors, two categorical series

SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e4e3df'

TRAIN = '#2a78d6'       # categorical slot 1 - blue
TEST = '#eb6834'        # categorical slot 2 - orange

plt.rcParams.update({
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'font.size': 10,
    'text.color': TEXT_PRIMARY,
    'axes.labelcolor': TEXT_SECONDARY,
    'axes.edgecolor': GRID,
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.8,
    'xtick.color': TEXT_SECONDARY,
    'ytick.color': TEXT_SECONDARY,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'lines.linewidth': 1.8,
    'lines.solid_capstyle': 'round',
})

# helpers

def ema(values, beta = 0.9):
    smoothed, avg = [], None

    for value in values:
        avg = value if avg is None else beta * avg + (1. - beta) * value
        smoothed.append(avg)

    return smoothed

def new_figure(title, subtitle = None):
    fig, ax = plt.subplots(figsize = (8, 4.2), dpi = 150)
    fig.suptitle(title, x = 0.07, y = 0.97, ha = 'left', fontsize = 13, fontweight = 'bold', color = TEXT_PRIMARY)

    if subtitle is not None:
        fig.text(0.07, 0.905, subtitle, ha = 'left', va = 'top', fontsize = 9.5, color = TEXT_SECONDARY, linespacing = 1.4)

    top = 0.8 - 0.045 * (subtitle or '').count('\n')
    fig.subplots_adjust(left = 0.09, right = 0.87, top = top, bottom = 0.13)
    return fig, ax

def end_label(ax, x, y, text, color_text = TEXT_PRIMARY):
    # direct label at the end of a series, in text color (the line itself carries the identity)
    ax.annotate(text, (x, y), xytext = (6, 0), textcoords = 'offset points', va = 'center', fontsize = 9, color = color_text)

def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')

# charts

def plot_overfit(metrics, out_dir):
    overfit = metrics['overfit']
    if len(overfit) == 0:
        return

    steps = [m['step'] for m in overfit]
    losses = [m['loss'] for m in overfit]

    fig, ax = new_figure(
        'Sanity check: overfitting a single batch',
        f'{metrics["config"]["overfit_batch_size"]} images, no dropout or augmentation · cross-entropy loss (log scale)'
    )

    ax.plot(steps, losses, color = TRAIN)
    ax.set_yscale('log')
    ax.set_xlabel('step')
    ax.set_ylabel('loss')
    ax.set_xlim(0, steps[-1])

    end_label(ax, steps[-1], losses[-1], f'{losses[-1]:.4f}')
    save(fig, out_dir / 'overfit.png')

def plot_loss(metrics, out_dir):
    steps, epochs = metrics['steps'], metrics['epochs']

    fig, ax = new_figure(
        'Loss',
        'train: per step, EMA-smoothed, includes label smoothing, dropout and augmentation\n'
        'test: plain cross-entropy per epoch - so train sitting above test is expected'
    )

    train_x = [m['epoch'] for m in steps]
    train_y = ema([m['loss'] for m in steps])
    test_x = [m['epoch'] for m in epochs]
    test_y = [m['test_loss'] for m in epochs]

    ax.plot(train_x, train_y, color = TRAIN, label = 'train')
    ax.plot(test_x, test_y, color = TEST, marker = 'o', markersize = 4.5, markeredgecolor = SURFACE, markeredgewidth = 1.2, label = 'test')

    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_xlim(0, test_x[-1])
    ax.legend(loc = 'upper right', labelcolor = TEXT_PRIMARY)

    end_label(ax, train_x[-1], train_y[-1], f'train {train_y[-1]:.3f}')
    end_label(ax, test_x[-1], test_y[-1], f'test {test_y[-1]:.3f}')
    save(fig, out_dir / 'loss.png')

def plot_accuracy(metrics, out_dir):
    epochs = metrics['epochs']

    x = [m['epoch'] for m in epochs]
    train_acc = [100 * m['train_acc'] for m in epochs]
    test_acc = [100 * m['test_acc'] for m in epochs]

    best_index = max(range(len(test_acc)), key = lambda i: test_acc[i])

    fig, ax = new_figure(
        'Accuracy',
        f'per epoch · train accuracy is measured on augmented batches with dropout · best test {test_acc[best_index]:.2f}% (epoch {x[best_index]})'
    )

    marker_kwargs = dict(marker = 'o', markersize = 4.5, markeredgecolor = SURFACE, markeredgewidth = 1.2)

    ax.plot(x, train_acc, color = TRAIN, label = 'train', **marker_kwargs)
    ax.plot(x, test_acc, color = TEST, label = 'test', **marker_kwargs)

    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy (%)')
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, 100)
    ax.legend(loc = 'lower right', labelcolor = TEXT_PRIMARY)

    # keep the two end labels from colliding when the curves finish close together

    gap = abs(train_acc[-1] - test_acc[-1])
    offset = (6. - gap) / 2 if gap < 6. else 0.
    upper_is_train = train_acc[-1] >= test_acc[-1]

    end_label(ax, x[-1], train_acc[-1] + (offset if upper_is_train else -offset), f'train {train_acc[-1]:.1f}%')
    end_label(ax, x[-1], test_acc[-1] + (-offset if upper_is_train else offset), f'test {test_acc[-1]:.1f}%')
    save(fig, out_dir / 'accuracy.png')

def plot_lr(metrics, out_dir):
    steps = metrics['steps']
    config = metrics['config']

    fig, ax = new_figure(
        'Learning rate',
        f'linear warmup over the first {config["warmup_frac"]:.0%} of steps, then cosine decay to zero · AdamW, peak {config["lr"]:g}'
    )

    ax.plot([m['epoch'] for m in steps], [m['lr'] * 1e3 for m in steps], color = TRAIN)
    ax.set_xlabel('epoch')
    ax.set_ylabel('learning rate (×10⁻³)')
    ax.set_xlim(0, steps[-1]['epoch'])
    ax.set_ylim(0, None)

    save(fig, out_dir / 'lr.png')

# main

def main():
    assert len(sys.argv) == 2, 'usage: python plot_training_curves.py <log dir containing metrics.json>'

    out_dir = Path(sys.argv[1])
    metrics = json.loads((out_dir / 'metrics.json').read_text())

    plot_overfit(metrics, out_dir)

    if len(metrics['epochs']) > 0:
        plot_loss(metrics, out_dir)
        plot_accuracy(metrics, out_dir)
        plot_lr(metrics, out_dir)

if __name__ == '__main__':
    main()
