# /// script
# dependencies = [
#   "datasets",
#   "matplotlib",
#   "vit-pytorch",
# ]
# ///

"""
Measure the residual stream of ViTs trained with train_cifar10.py --save, layer by layer, with forward hooks

the residual stream never passes through a module boundary in vit.py (x = attn(x) + x happens inside Transformer.forward),
so it is read as the *input* of each block's attention module, plus the stream after the last block

    pre-norm  - after the last block = input of the final LayerNorm (transformer.norm)
    post-norm - after the last block = output of the transformer (each block already ends with a LayerNorm)

per layer it reports

    stream norm       mean per token L2 norm of the residual stream entering the block
    relative change   mean per token ||x_{i+1} - x_i|| / ||x_i|| - how much one block changes the representation

usage:
    python train_cifar10.py --skip-overfit --norm pre --save runs/models/pre.pt
    python train_cifar10.py --skip-overfit --norm post --save runs/models/post.pt
    python inspect_residual_stream.py --models pre=runs/models/pre.pt post=runs/models/post.pt --out-dir runs/residual_stream
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# reuse data loading, model loading and chart style from inspect_attention.py

_spec = importlib.util.spec_from_file_location('inspect_attention', Path(__file__).parent / 'inspect_attention.py')
inspect_attention = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inspect_attention)

load_test_set, load_model, normalize = inspect_attention.load_test_set, inspect_attention.load_model, inspect_attention.normalize
SURFACE, TEXT_PRIMARY, SERIES = inspect_attention.SURFACE, inspect_attention.TEXT_PRIMARY, inspect_attention.SERIES

# arguments

def parse_args():
    parser = argparse.ArgumentParser(description = 'measure the residual stream of trained CIFAR-10 ViTs per layer')
    parser.add_argument('--models', nargs = '+', required = True, help = 'name=path pairs, e.g. pre=runs/models/pre.pt (name is the --norm used for training)')
    parser.add_argument('--out-dir', default = 'runs/residual_stream')
    parser.add_argument('--num-images', type = int, default = 1000)
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

# measurement

@torch.no_grad()
def record_stream(model, norm, images):
    # returns a list of residual stream tensors (n, tokens, dim): the input of every block, then the stream after the last block

    streams = []

    # observation hooks must return None - a forward hook's return value replaces the module's output

    def block_input_hook(module, inputs, output):
        streams.append(inputs[0].float().cpu())

    def final_hook(module, inputs, output):
        streams.append((inputs[0] if norm == 'pre' else output).float().cpu())

    handles = [attn.register_forward_hook(block_input_hook) for attn, _ in model.transformer.layers]
    final_target = model.transformer.norm if norm == 'pre' else model.transformer
    handles.append(final_target.register_forward_hook(final_hook))

    model(normalize(images))

    for handle in handles:
        handle.remove()

    return streams

def stream_statistics(streams):
    stream_norm = [s.norm(dim = -1).mean().item() for s in streams]

    relative_change = [
        ((after - before).norm(dim = -1) / before.norm(dim = -1).clamp(min = 1e-6)).mean().item()
        for before, after in zip(streams[:-1], streams[1:])
    ]

    return dict(stream_norm = stream_norm, relative_change = relative_change)

# plot

def plot(stats, num_images, path):
    fig, (norm_ax, change_ax) = plt.subplots(1, 2, figsize = (11, 4.3), dpi = 150)
    fig.suptitle('Residual stream per layer: pre-norm vs post-norm', x = 0.06, y = 0.98, ha = 'left', fontsize = 13, fontweight = 'bold')
    fig.text(0.06, 0.905, f'mean over {num_images} CIFAR-10 test images and all 65 tokens · trained models', ha = 'left', va = 'top', fontsize = 9.5, color = inspect_attention.TEXT_SECONDARY)

    marker_kwargs = dict(linewidth = 2., marker = 'o', markersize = 4.5, markeredgecolor = SURFACE, markeredgewidth = 1.2)

    for color, (name, variant_stats) in zip(SERIES, stats.items()):
        norms = variant_stats['stream_norm']
        changes = variant_stats['relative_change']

        norm_ax.plot(range(len(norms)), norms, color = color, label = f'{name}-norm', **marker_kwargs)
        change_ax.plot(range(len(changes)), changes, color = color, label = f'{name}-norm', **marker_kwargs)

        norm_ax.annotate(f'{norms[-1]:.1f}', (len(norms) - 1, norms[-1]), xytext = (6, 0), textcoords = 'offset points', va = 'center', fontsize = 9, color = TEXT_PRIMARY)

    depth = len(next(iter(stats.values()))['relative_change'])

    norm_ax.set_title('stream norm entering each block', loc = 'left')
    norm_ax.set_xticks(range(depth + 1), [str(i) for i in range(depth)] + ['out'])
    norm_ax.set_xlabel('block (out = after the last block)')
    norm_ax.set_ylabel('mean token L2 norm')
    norm_ax.set_ylim(0, None)

    change_ax.set_title('relative change made by each block', loc = 'left')
    change_ax.set_xticks(range(depth))
    change_ax.set_xlabel('block')
    change_ax.set_ylabel('‖x after − x before‖ / ‖x before‖')
    change_ax.set_ylim(0, None)
    change_ax.legend(loc = 'lower left', labelcolor = TEXT_PRIMARY)

    for ax in (norm_ax, change_ax):
        ax.grid(True, axis = 'y')

    fig.subplots_adjust(left = 0.07, right = 0.97, top = 0.8, bottom = 0.14, wspace = 0.25)
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')

# main

def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    images, _, _ = load_test_set(device)
    images = images[:args.num_images]

    stats = dict()

    for spec in args.models:
        name, path = spec.split('=', 1)
        model = load_model(name, path, device)
        stats[name] = stream_statistics(record_stream(model, name, images))

        print(f'{name}-norm stream norm     :', ' '.join(f'{v:6.1f}' for v in stats[name]['stream_norm']))
        print(f'{name}-norm relative change :', ' '.join(f'{v:6.2f}' for v in stats[name]['relative_change']))

    plot(stats, args.num_images, out_dir / 'residual_stream.png')
    (out_dir / 'residual_stream.json').write_text(json.dumps(stats, indent = 2))

if __name__ == '__main__':
    main()
