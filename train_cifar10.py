# /// script
# dependencies = [
#   "datasets",
#   "vit-pytorch",
# ]
# ///

"""
Train a ViT on CIFAR-10, in two phases

1. sanity check - overfit a single batch until the loss is ~0, proving the model and training loop can learn
2. train on the full CIFAR-10 train set, evaluating on the test set after every epoch

CIFAR-10 is loaded with Hugging Face datasets (uoft-cs/cifar10) and kept entirely on device as uint8 (~150MB),
with augmentation (random crop with padding, horizontal flip) also done on device - no DataLoader needed

usage:
    python train_cifar10.py
    python train_cifar10.py --epochs 50
    python train_cifar10.py --epochs 0                  # only the single batch overfit check
    python train_cifar10.py --skip-overfit --save vit_cifar10.pt
"""

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from datasets import load_dataset

from vit_pytorch import ViT

# constants

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# arguments

def parse_args():
    parser = argparse.ArgumentParser(description = 'overfit one batch, then train a ViT on CIFAR-10')

    parser.add_argument('--epochs', type = int, default = 30)
    parser.add_argument('--batch-size', type = int, default = 256)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--weight-decay', type = float, default = 0.05)
    parser.add_argument('--warmup-frac', type = float, default = 0.05, help = 'fraction of total steps used for linear warmup')
    parser.add_argument('--label-smoothing', type = float, default = 0.1)

    parser.add_argument('--overfit-batch-size', type = int, default = 32)
    parser.add_argument('--overfit-steps', type = int, default = 500)
    parser.add_argument('--skip-overfit', action = 'store_true')

    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save', default = None, help = 'path to save the final model state dict')

    return parser.parse_args()

# model

def make_vit(dropout = 0.1):
    # 32x32 images with 4x4 patches -> 8x8 = 64 patch tokens + 1 cls token

    return ViT(
        image_size = 32,
        patch_size = 4,
        num_classes = 10,
        dim = 256,
        depth = 6,
        heads = 8,
        dim_head = 32,
        mlp_dim = 512,
        dropout = dropout,
        emb_dropout = dropout
    )

# data

def load_cifar10(device):
    dataset = load_dataset('uoft-cs/cifar10')

    def to_tensors(split):
        images = np.stack([np.asarray(img.convert('RGB'), dtype = np.uint8) for img in split['img']])
        images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()        # (n, 32, 32, 3) -> (n, 3, 32, 32)
        labels = torch.tensor(split['label'], dtype = torch.long)
        return images.to(device), labels.to(device)

    return (*to_tensors(dataset['train']), *to_tensors(dataset['test']))

def normalize(images):
    mean = torch.tensor(CIFAR10_MEAN, device = images.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device = images.device).view(1, 3, 1, 1)
    return (images.float() / 255. - mean) / std

def augment(images, padding = 4):
    # random crop with reflect padding + random horizontal flip, per image, on device

    batch, channels, height, width = images.shape
    device = images.device

    padded = F.pad(images, (padding,) * 4, mode = 'reflect')

    top = torch.randint(0, 2 * padding + 1, (batch,), device = device)
    left = torch.randint(0, 2 * padding + 1, (batch,), device = device)

    rows = (top[:, None] + torch.arange(height, device = device))[:, None, :, None]
    cols = (left[:, None] + torch.arange(width, device = device))[:, None, None, :]

    batch_index = torch.arange(batch, device = device)[:, None, None, None]
    channel_index = torch.arange(channels, device = device)[None, :, None, None]

    cropped = padded[batch_index, channel_index, rows, cols]

    flip = torch.rand(batch, device = device) < 0.5
    return torch.where(flip[:, None, None, None], cropped.flip(-1), cropped)

# phase 1 - overfit a single batch

def overfit_one_batch(images, labels, steps, device, log_every = 50):
    print(f'\n=== phase 1: overfit a single batch of {images.shape[0]} images ===')

    model = make_vit(dropout = 0.).to(device)          # no dropout, no augmentation - we want memorization
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.)

    x, y = normalize(images), labels

    for step in range(1, steps + 1):
        loss = F.cross_entropy(model(x), y)

        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            print(f'step {step:4d} | loss {loss.item():.4f}')

    with torch.no_grad():
        model.eval()
        logits = model(x)
        final_loss = F.cross_entropy(logits, y).item()
        accuracy = (logits.argmax(dim = -1) == y).float().mean().item()

    passed = final_loss < 0.05 and accuracy == 1.
    print(f'final loss {final_loss:.4f} | batch accuracy {accuracy:.0%} | {"PASSED" if passed else "FAILED - the model could not memorize one batch, check the model / training loop"}')
    return passed

# phase 2 - full training

@torch.no_grad()
def evaluate(model, images, labels, batch_size = 1000):
    model.eval()
    correct = 0

    for start in range(0, images.shape[0], batch_size):
        x = normalize(images[start:start + batch_size])
        y = labels[start:start + batch_size]

        with autocast(x.device):
            correct += (model(x).argmax(dim = -1) == y).sum().item()

    return correct / images.shape[0]

def autocast(device):
    # bfloat16 mixed precision on cuda, full precision elsewhere
    return torch.autocast(device_type = 'cuda', dtype = torch.bfloat16, enabled = device.type == 'cuda')

def train(args, train_images, train_labels, test_images, test_labels):
    print(f'\n=== phase 2: train on CIFAR-10 ({train_images.shape[0]} train / {test_images.shape[0]} test images) ===')

    device = train_images.device
    model = make_vit().to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model: ViT with {num_params:,} trainable parameters')

    # no weight decay on biases, norms, cls token and positional embedding

    decay = [p for p in model.parameters() if p.ndim >= 2 and p is not model.pos_embedding and p is not model.cls_token]
    no_decay = [p for p in model.parameters() if not any(p is q for q in decay)]

    optimizer = torch.optim.AdamW([
        dict(params = decay, weight_decay = args.weight_decay),
        dict(params = no_decay, weight_decay = 0.)
    ], lr = args.lr)

    # linear warmup, then cosine decay to zero

    steps_per_epoch = train_images.shape[0] // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(args.warmup_frac * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1. + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    best_accuracy = 0.

    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()

        permutation = torch.randperm(train_images.shape[0], device = device)
        total_loss = total_correct = total_seen = 0

        for step in range(steps_per_epoch):
            indices = permutation[step * args.batch_size:(step + 1) * args.batch_size]

            x = augment(normalize(train_images[indices]))
            y = train_labels[indices]

            with autocast(device):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing = args.label_smoothing)

            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * y.shape[0]
            total_correct += (logits.argmax(dim = -1) == y).sum().item()
            total_seen += y.shape[0]

        test_accuracy = evaluate(model, test_images, test_labels)
        best_accuracy = max(best_accuracy, test_accuracy)

        print(
            f'epoch {epoch:3d}/{args.epochs} | '
            f'train loss {total_loss / total_seen:.4f} | '
            f'train acc {total_correct / total_seen:.2%} | '
            f'test acc {test_accuracy:.2%} | '
            f'lr {scheduler.get_last_lr()[0]:.2e} | '
            f'{time.time() - start_time:.1f}s'
        )

    print(f'\nfinal test accuracy {test_accuracy:.2%} | best {best_accuracy:.2%}')
    return model

# main

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f'device: {device}')

    load_start = time.time()
    train_images, train_labels, test_images, test_labels = load_cifar10(device)
    print(f'loaded CIFAR-10 in {time.time() - load_start:.1f}s')

    if not args.skip_overfit:
        overfit_one_batch(
            train_images[:args.overfit_batch_size],
            train_labels[:args.overfit_batch_size],
            steps = args.overfit_steps,
            device = device
        )

    if args.epochs <= 0:
        return

    model = train(args, train_images, train_labels, test_images, test_labels)

    if args.save is not None:
        torch.save(model.state_dict(), args.save)
        print(f'saved model to {args.save}')

if __name__ == '__main__':
    main()
