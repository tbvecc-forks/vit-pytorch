# /// script
# dependencies = [
#   "datasets",
#   "matplotlib",
#   "vit-pytorch",
# ]
# ///

"""
Inspect the attention of ViTs trained with train_cifar10.py --save, using vit_pytorch.recorder.Recorder

for every model:
    1. rebuild the architecture with make_vit(norm = ...) and load the saved weights
    2. take CIFAR-10 test images (Hugging Face datasets, uoft-cs/cifar10)
    3. record the attention matrices of all layers with Recorder
    4. plot the cls token -> patch attention per layer (averaged over heads) next to each image

and compute per layer statistics over many test images

    cls self-attention   - weight the cls token puts on itself
    normalized entropy   - entropy of the cls -> patch distribution / log(64): 1 = uniform, lower = focused
    map correlation      - mean pearson correlation of each image's cls -> patch map with the average map over all images
                           high = the same pattern for every image (positional), low = depends on the image (content)

usage:
    python train_cifar10.py --skip-overfit --norm pre --save runs/models/pre.pt
    python train_cifar10.py --skip-overfit --norm post --save runs/models/post.pt
    python inspect_attention.py --models pre=runs/models/pre.pt post=runs/models/post.pt --out-dir runs/attention
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from datasets import load_dataset

from vit_pytorch.recorder import Recorder

# reuse the model definition and normalization from the training script

_spec = importlib.util.spec_from_file_location('train_cifar10', Path(__file__).parent / 'train_cifar10.py')
train_cifar10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_cifar10)

make_vit, normalize = train_cifar10.make_vit, train_cifar10.normalize

# style - light surface, text in text colors, variants in fixed categorical order, sequential blue ramp for attention

SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e4e3df'

SERIES = ['#2a78d6', '#eb6834', '#1baf7a']

ATTENTION_CMAP = LinearSegmentedColormap.from_list('blue_sequential', ['#f4f8fe', '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])

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
    'grid.color': GRID,
    'grid.linewidth': 0.8,
    'xtick.color': TEXT_SECONDARY,
    'ytick.color': TEXT_SECONDARY,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
})

# arguments

def parse_args():
    parser = argparse.ArgumentParser(description = 'inspect cls token attention of trained CIFAR-10 ViTs')
    parser.add_argument('--models', nargs = '+', required = True, help = 'name=path pairs, e.g. pre=runs/models/pre.pt (name is the --norm used for training)')
    parser.add_argument('--out-dir', default = 'runs/attention')
    parser.add_argument('--num-grid-images', type = int, default = 6, help = 'images shown in the attention grid, one per class')
    parser.add_argument('--num-stat-images', type = int, default = 2000, help = 'test images used for the per layer statistics')
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

# data

def load_test_set(device):
    split = load_dataset('uoft-cs/cifar10')['test']
    images = np.stack([np.asarray(img.convert('RGB'), dtype = np.uint8) for img in split['img']])
    images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().to(device)
    labels = torch.tensor(split['label'], dtype = torch.long, device = device)
    return images, labels, split.features['label'].names

def pick_one_per_class(labels, count):
    # first test image of each of the first `count` classes, for a deterministic grid
    return torch.stack([(labels == c).nonzero()[0, 0] for c in range(count)])

# model

def load_model(norm, path, device):
    model = make_vit(norm = norm)
    model.load_state_dict(torch.load(path, map_location = device))
    return model.to(device).eval()

@torch.no_grad()
def record_attention(model, images, batch_size = 500):
    # returns predictions (n,) and cls -> all tokens attention (n, depth, heads, 65), on cpu

    recorder = Recorder(model)
    preds, cls_rows = [], []

    for start in range(0, images.shape[0], batch_size):
        logits, attns = recorder(normalize(images[start:start + batch_size]))
        preds.append(logits.argmax(dim = -1).cpu())
        cls_rows.append(attns[:, :, :, 0, :].float().cpu())        # only the cls token's row - (b, depth, heads, tokens)

    recorder.eject()
    return torch.cat(preds), torch.cat(cls_rows)

# statistics

def normalized_entropy(maps):
    # entropy of attention renormalized over the patches, divided by log(num patches): 1 = uniform
    dist = maps / maps.sum(dim = -1, keepdim = True)
    return -(dist * dist.clamp(min = 1e-12).log()).sum(dim = -1) / math.log(dist.shape[-1])

def layer_statistics(cls_rows):
    # cls_rows: (n, depth, heads, 1 + num_patches)

    rows = cls_rows.mean(dim = 2)                                    # head-averaged, like the plots - (n, depth, tokens)
    self_attention = rows[..., 0]                                    # (n, depth)
    patch_maps = rows[..., 1:]

    # how much the head average hides - entropy of each head's own map

    per_head_entropy = normalized_entropy(cls_rows[..., 1:]).mean(dim = 2)

    # share of attention on the central 4x4 of the 8x8 grid (CIFAR objects tend to be centered) - 0.25 if uniform

    grid = int(math.sqrt(patch_maps.shape[-1]))
    margin = grid // 4
    centre = torch.zeros(grid, grid, dtype = torch.bool)
    centre[margin:grid - margin, margin:grid - margin] = True

    patch_dist = patch_maps / patch_maps.sum(dim = -1, keepdim = True)
    centre_mass = patch_dist[..., centre.flatten()].sum(dim = -1)

    # same pattern for every image (positional) vs image dependent (content)

    centered = patch_maps - patch_maps.mean(dim = -1, keepdim = True)
    mean_map = patch_maps.mean(dim = 0, keepdim = True)
    mean_centered = mean_map - mean_map.mean(dim = -1, keepdim = True)
    correlation = (centered * mean_centered).sum(-1) / (centered.norm(dim = -1) * mean_centered.norm(dim = -1)).clamp(min = 1e-12)

    return dict(
        cls_self_attention = self_attention.mean(dim = 0).tolist(),
        normalized_entropy = normalized_entropy(patch_maps).mean(dim = 0).tolist(),
        per_head_entropy = per_head_entropy.mean(dim = 0).tolist(),
        centre_mass = centre_mass.mean(dim = 0).tolist(),
        uniform_centre_mass = centre.float().mean().item(),
        map_correlation_with_mean = correlation.mean(dim = 0).tolist()
    )

# plots

def plot_attention_grid(name, images, labels, preds, cls_rows, class_names, path):
    num_images, depth = cls_rows.shape[0], cls_rows.shape[1]
    grid = int(math.sqrt(cls_rows.shape[-1] - 1))

    fig, axes = plt.subplots(num_images, depth + 1, figsize = (1.55 * (depth + 1), 1.55 * num_images + 1.1), dpi = 150, squeeze = False)

    fig.suptitle(f'cls token attention per layer: {name}-norm ViT', x = 0.02, y = 0.995, ha = 'left', fontsize = 13, fontweight = 'bold')
    fig.text(0.02, 0.965, 'cls → 64 patches, averaged over 8 heads · each panel scaled to its own maximum (shows where, not how much)\n'
                          'number above a panel: weight the cls token keeps on itself', ha = 'left', va = 'top', fontsize = 9, color = TEXT_SECONDARY, linespacing = 1.4)

    for row in range(num_images):
        image_ax = axes[row, 0]
        image_ax.imshow(images[row].permute(1, 2, 0).cpu().numpy(), interpolation = 'nearest')

        correct = preds[row] == labels[row]
        image_ax.set_title(f'{class_names[labels[row]]}\npred: {class_names[preds[row]]}' + ('' if correct else ' ✗'), fontsize = 8.5, color = TEXT_PRIMARY if correct else '#d03b3b')

        for layer in range(depth):
            ax = axes[row, layer + 1]
            head_mean = cls_rows[row, layer].mean(dim = 0)
            ax.imshow(head_mean[1:].reshape(grid, grid).numpy(), cmap = ATTENTION_CMAP, interpolation = 'nearest')
            ax.set_title(f'{head_mean[0]:.2f}', fontsize = 8, color = TEXT_SECONDARY)

            if row == 0:
                ax.annotate(f'layer {layer}', (0.5, 1.45), xycoords = 'axes fraction', ha = 'center', fontsize = 9.5, fontweight = 'bold')

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left = 0.02, right = 0.98, top = 1 - 1.35 / (1.55 * num_images + 1.1), bottom = 0.01, hspace = 0.55, wspace = 0.08)
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')

def plot_statistics(stats, num_images, path):
    # (key, title, y label, y limits) - each panel is its own measure on its own axis

    panels = [
        ('normalized_entropy', 'entropy of the head-averaged map', '1 = uniform, lower = focused', (0, 1)),
        ('per_head_entropy', 'entropy of each head, averaged', '1 = uniform, lower = focused', (0, 1)),
        ('centre_mass', 'attention on the central 4×4 patches', 'share of cls → patch attention', (0, 0.6)),
        ('map_correlation_with_mean', 'map similarity across images', 'correlation with the mean map', (0, 1)),
        ('cls_self_attention', 'cls self-attention', 'weight on itself', (0, None)),
    ]

    fig, axes = plt.subplots(2, 3, figsize = (12, 7.4), dpi = 150)
    fig.suptitle('cls token attention statistics per layer', x = 0.05, y = 0.985, ha = 'left', fontsize = 13, fontweight = 'bold')
    fig.text(0.05, 0.945, f'mean over {num_images} CIFAR-10 test images', ha = 'left', va = 'top', fontsize = 9.5, color = TEXT_SECONDARY)

    for ax, (key, title, ylabel, ylim) in zip(axes.flat, panels):
        for color, (name, variant_stats) in zip(SERIES, stats.items()):
            values = variant_stats[key]
            ax.plot(range(len(values)), values, color = color, linewidth = 2., marker = 'o', markersize = 4.5, markeredgecolor = SURFACE, markeredgewidth = 1.2, label = f'{name}-norm')

        if key == 'centre_mass':
            uniform = next(iter(stats.values()))['uniform_centre_mass']
            ax.axhline(uniform, color = TEXT_SECONDARY, linewidth = 1., linestyle = (0, (4, 3)))
            ax.annotate('uniform', (len(values) - 1, uniform), xytext = (0, -12), textcoords = 'offset points', ha = 'right', fontsize = 8.5, color = TEXT_SECONDARY)

        ax.set_title(title, loc = 'left')
        ax.set_xlabel('layer')
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(values)))
        ax.set_ylim(*ylim)
        ax.grid(True, axis = 'y')

    # last cell holds the legend instead of a chart

    legend_ax = axes.flat[-1]
    legend_ax.axis('off')
    legend_ax.legend(*axes.flat[0].get_legend_handles_labels(), loc = 'center', labelcolor = TEXT_PRIMARY, fontsize = 11)

    fig.subplots_adjust(left = 0.06, right = 0.98, top = 0.87, bottom = 0.08, wspace = 0.3, hspace = 0.45)
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')

# main

def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    images, labels, class_names = load_test_set(device)
    grid_indices = pick_one_per_class(labels.cpu(), args.num_grid_images)

    stats = dict()

    for spec in args.models:
        name, path = spec.split('=', 1)
        model = load_model(name, path, device)

        preds, cls_rows = record_attention(model, images)
        accuracy = (preds == labels.cpu()).float().mean().item()
        print(f'{name}-norm: test accuracy {accuracy:.2%} (sanity check that the weights loaded)')

        stats[name] = dict(test_accuracy = accuracy, **layer_statistics(cls_rows[:args.num_stat_images]))

        plot_attention_grid(
            name,
            images[grid_indices],
            labels[grid_indices].cpu(),
            preds[grid_indices],
            cls_rows[grid_indices],
            class_names,
            out_dir / f'attention_grid_{name}.png'
        )

    plot_statistics(stats, args.num_stat_images, out_dir / 'attention_stats.png')

    (out_dir / 'attention_stats.json').write_text(json.dumps(stats, indent = 2))
    print(json.dumps(stats, indent = 2))

if __name__ == '__main__':
    main()
