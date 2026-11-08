# vit-pytorch — Codebase Analysis

> Scope: the full `vit_pytorch` package (85 modules, ~20.9k LoC), tests, packaging, example notebook and README, at commit `42298e1` (v1.26.14, 2026-11-07).
> Method: every module was read in full and every bug listed in [§10](#10-known-issues--verified-by-reading-the-source) was re-checked against the source lines it cites. Items §10.1 #1, #2, #3, #4, #5, #7 and #10 were also **reproduced at runtime** (torch 2.5.0+cu124, einops 0.8.2). Everything else comes from static reading.

---

## Table of contents

1. [What this repository is](#1-what-this-repository-is)
2. [Repository layout](#2-repository-layout)
3. [Design philosophy and conventions](#3-design-philosophy-and-conventions)
4. [The reference architecture: `ViT` and `SimpleViT`](#4-the-reference-architecture-vit-and-simplevit)
5. [Public API](#5-public-api)
6. [Model catalog](#6-model-catalog)
7. [Training / self-supervised wrappers and the encoder contract](#7-training--self-supervised-wrappers-and-the-encoder-contract)
8. [Inspection utilities: `Recorder` and `Extractor`](#8-inspection-utilities-recorder-and-extractor)
9. [Dependencies, packaging, tests, tooling](#9-dependencies-packaging-tests-tooling)
10. [Known issues (verified by reading the source)](#10-known-issues--verified-by-reading-the-source)
11. [Practical guidance](#11-practical-guidance)
12. [Study plan](#12-study-plan)

---

## 1. What this repository is

`vit-pytorch` is Phil Wang's (lucidrains) collection of **Vision Transformer implementations in PyTorch**. It is a fork of the upstream project, which now lives on Codeberg per `pyproject.toml`. It started in October 2020 as one file (`vit.py`). Six years and 434 commits later it is a catalog of ~80 architectures and training methods taken from papers:

- **Image classifiers.** Faithful or near-faithful re-implementations of published ViT variants: CaiT, DeepViT, T2T, CvT, PiT, LeViT, NesT, Twins, CrossFormer, RegionViT, ScalableViT, SepViT, MaxViT, MobileViT, XCiT, CCT, and others.
- **Single-idea research ablations** on top of `SimpleViT`: register tokens, QK-norm, value residuals, hyper-connections, attention residuals, nGPT-style normalization, PoPE/RoPE, decorrelation losses, and others.
- **Other modalities and shapes:** 1D, 3D/video, N-D (1–7 dims), variable-resolution (NaViT), and vision-action models for robotics (VAT / VAAT / SigLIP-VAT).
- **Training wrappers:** masked image modelling (MAE, SimMIM, MPP, MP3), self-distillation (DINO, EsViT), JEPA-style (LeJEPA, LeVJEPA), and DeiT distillation.
- **Utilities:** attention-map recording and embedding extraction.

It is a **reference/teaching library**, not a framework. There are no shared base classes, no registry, no pretrained-weight hub (except a SigLIP weight loader), and almost no tests. The README (~2.5k lines, 68 BibTeX citations) is the main documentation and works as a cookbook: each section is a paper, a figure, and a copy-pasteable snippet.

---

## 2. Repository layout

```
vit-pytorch/
├── pyproject.toml          # setuptools build, v1.26.14, deps: torch>=2.5, torchvision, einops>=0.8.2
├── README.md               # the documentation: one section per architecture/method + citations
├── LICENSE                 # MIT
├── MANIFEST.in
├── images/                 # 34 paper figures referenced by README
├── examples/
│   └── cats_and_dogs.ipynb # Kaggle Dogs-vs-Cats with efficient.ViT + Linformer (CUDA, extra deps)
├── train_vit_decorr.py     # PEP-723 script: CIFAR-100 training of vit_with_decorr.ViT (accelerate, wandb)
├── tests/
│   ├── test_vit.py         # 1 forward-shape smoke test for ViT
│   └── test_dino_loss.py   # 3 numerical tests for dino.loss_fn
└── vit_pytorch/            # 85 flat modules, no sub-packages
    ├── __init__.py         # exports ViT, SimpleViT, MAE, Dino, LoopedViT only
    ├── vit.py, simple_vit.py         # reference implementations
    ├── <paper>.py                    # one file per architecture/method
    └── simple_vit_with_<idea>.py     # one file per ablation on SimpleViT
```

The package is **flat**. Every module stands alone, and the file name is the import path (`from vit_pytorch.cait import CaiT`).

### Intra-package dependency graph

There is very little reuse. These are the only cross-module imports:

```
vit.py ──Transformer──▶ mae.py, t2t.py
       ──Attention────▶ recorder.py          (isinstance-based hook discovery)
       ──ViT──────────▶ distill.py
t2t.py ──T2TViT───────▶ distill.py
efficient.py ─ViT─────▶ distill.py
vit_nd_masking.py ────▶ vit_nd.py, vit_nd_rotary.py, vit_nd_pope.py, levjepa_nd.py
lejepa.py ─sigreg_loss▶ wwt.py
vivit_with_moss.py ─MOSS▶ accept_video_wrapper.py
```

Everything else (`Attention`, `FeedForward`, `Transformer`, `pair`, `exists`, `default`, `posemb_sincos_2d`, channel `LayerNorm`, `PEG`, `MBConv`, `EMA`, `NetWrapper`, …) is **copy-pasted per file**, often with small differences.

---

## 3. Design philosophy and conventions

| Convention | What it looks like | Notes |
|---|---|---|
| **One paper = one file** | Each file contains its own `Attention`, `FeedForward`, `Transformer`, model class, and helpers. | Reading a file tells you everything about that model; a fix in one file never reaches its siblings. |
| **einops everywhere** | Patchify is `Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)')`. Head splits are `rearrange(t, 'b n (h d) -> b h n d')`. `pack`/`unpack` attach cls/register tokens. | Shapes are self-documenting in the patterns. |
| **Keyword-only constructors** | `def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, ...)` | Exceptions: `MobileViT`, `CCT`, `MPP`, `Dino`, the LeJEPA family. |
| **Pre-norm residual blocks** | LayerNorm lives *inside* `Attention`/`FeedForward`. The `Transformer` does `x = attn(x) + x; x = ff(x) + x` and ends with a final LayerNorm. | Exceptions: CCT (timm-style hybrid), nViT (L2 norm and lerp residuals), Keel (post-LN). |
| **Shared hyper-parameter vocabulary** | `dim`, `depth`, `heads`, `dim_head=64`, `mlp_dim`, `dropout`, `emb_dropout`, `channels=3`, `pool ∈ {'cls','mean'}` | Names drift in hierarchical models: `mlp_mult`, `ff_mult`, `expansion_factor`, `mlp_ratio`; per-stage tuples for `dim`/`depth`. |
| **`forward(img) -> logits`** | Classifiers return `(b, num_classes)`. | Extras are opt-in flags (`return_loss`, `return_hiddens`, `return_sampled_token_ids`, …) or always-tuples (`vit_with_decorr`). |
| **Wrappers return a scalar loss** | `loss = MAE(encoder=v, ...)(images); loss.backward()` | EMA teachers are stepped manually: `learner.update_moving_average()`. |
| **Inline smoke tests** | Newer files end with `if __name__ == '__main__':` building the model and running one forward pass. | Not collected by pytest; two of them call `.cuda()`. |
| **Paper link at top of file** | Newer files carry a URL comment. | Older files rely on README citations. |

There are two coding "generations" in the codebase:

- **2020–2022:** `nn.Module`, explicit `matmul` + `softmax`, learned 3-D `pos_embedding` of shape `(1, N+1, dim)`, `cls_token` of shape `(1, 1, dim)`.
- **2024–2026:** `from torch.nn import Module, ModuleList`, `F.scaled_dot_product_attention`, `nn.RMSNorm`, `LayerNorm(bias=False)`, headwise sigmoid gates, flex-attention, `__main__` tests, 2-D `pos_embedding`/`cls_token`.

The two layouts no longer agree, which is the root cause of the most important bug cluster (see [§7](#7-training--self-supervised-wrappers-and-the-encoder-contract)).

---

## 4. The reference architecture: `ViT` and `SimpleViT`

### 4.1 `vit_pytorch/vit.py` — `ViT`

```python
ViT(*, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
    pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.)
```

Data flow (`b` = batch, `N = (H/ph)·(W/pw)` patches):

```
img (b, C, H, W)
 │ to_patch_embedding = Sequential(
 │     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)'),   # [0] patchify
 │     LayerNorm(patch_dim), Linear(patch_dim, dim),          # [1], [2]
 │     LayerNorm(dim))                                        # [3]
 ▼
x (b, N, dim)
 │ cat(cls_token repeated → (b, 1, dim))        # only if pool == 'cls' (cls_token is (0, dim) otherwise)
 │ + pos_embedding[:seq]                        # learned, shape (N + num_cls, dim)  ← 2-D since 2025-11
 │ Dropout(emb_dropout)
 ▼
Transformer: depth × [ x = Attention(x) + x ; x = FeedForward(x) + x ] → LayerNorm
   Attention   = LayerNorm → Linear(dim, 3·heads·dim_head, bias=False) → softmax(QKᵀ·d^-½) → dropout → V
                 → Linear(inner, dim) + Dropout   (Identity if heads==1 and dim_head==dim)
   FeedForward = LayerNorm → Linear → GELU → Dropout → Linear → Dropout
 ▼
x (b, N+1, dim)
 │ if num_classes == 0 → return tokens (b, N+num_cls, dim)     # backbone mode
 │ pool: x[:, 0] (cls) or x.mean(1) (mean)
 │ to_latent (Identity — exists as a hook point for Dino/Extractor)
 ▼
mlp_head = Linear(dim, num_classes) → logits (b, num_classes)
```

Facts that matter for downstream code:

- **The position embedding has a fixed size.** It is sliced by `[:seq]`, so smaller inputs work, but larger ones fail.
- `to_patch_embedding` is a `Sequential` whose **indices are relied upon** by MAE/SimMIM/MPP (`[0]` is patchify, `[2]` is the Linear).
- `to_latent` is a deliberate no-op hook point.
- `num_classes=0` turns the classifier into a backbone. Only a few other files support this.

### 4.2 `vit_pytorch/simple_vit.py` — `SimpleViT`

From "Better plain ViT baselines for ImageNet-1k". It is the base for ~20 ablation files.

```python
SimpleViT(*, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64)
```

Differences from `ViT`:

- **Fixed 2-D sin-cos positional embedding** (`posemb_sincos_2d`, `dim % 4 == 0`). It is a plain tensor attribute, not a parameter or buffer, and is moved to the device on every forward.
- No cls token: it always **mean-pools**.
- No dropout anywhere; `to_out` has no bias.
- `linear_head` is required; there is no `num_classes=0` path.

### 4.3 Minimal usage

```python
import torch
from vit_pytorch import ViT, SimpleViT

v = ViT(image_size=256, patch_size=32, num_classes=1000, dim=1024, depth=6,
        heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1)
logits = v(torch.randn(1, 3, 256, 256))      # (1, 1000)

backbone = ViT(..., num_classes=0)           # returns (b, N+1, dim) tokens
```

---

## 5. Public API

### 5.1 Top-level exports (`vit_pytorch/__init__.py`)

| Name | Source | Kind |
|---|---|---|
| `ViT` | `vit.py` | classifier |
| `SimpleViT` | `simple_vit.py` | classifier |
| `MAE` | `mae.py` | self-supervised wrapper |
| `Dino` | `dino.py` | self-supervised wrapper |
| `LoopedViT` | `looped_vit.py` | classifier with looped blocks / adaptive exit |

**Everything else must be imported by module path**, e.g. `from vit_pytorch.max_vit import MaxViT`. Watch for name collisions: many modules export a class called `ViT` or `SimpleViT` (`efficient.ViT`, `parallel_vit.ViT`, `vit_for_small_dataset.ViT`, `ats_vit.ViT`, `vit_1d.ViT`, `vit_3d.ViT`, `vit_5.ViT`, `mp3.ViT`, `vat.ViT`, `simple_vit_with_*.SimpleViT`, …). The module path is the real identifier.

### 5.2 Calling conventions by category

| Category | Construct | Call | Returns |
|---|---|---|---|
| Classifier | `Model(*, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, ...)` | `model(img)` | `(b, num_classes)` |
| Backbone mode | `num_classes=0` (ViT, vit_5, keel, detpool, jet_vit) | `model(img)` | tokens `(b, n, dim)` |
| Aux-loss model | e.g. `vit_with_decorr.ViT`, `SimpleViTWithSparseGating` | `model(img[, return_loss=True])` | `(logits, aux_loss)` |
| SSL wrapper | `Wrapper(encoder, ...)` | `loss = w(images)` | scalar loss |
| EMA wrapper | `Dino`, `EsViTTrainer`, `LeJEPA(use_ema=True)` | `loss = w(x); loss.backward(); opt.step(); w.update_moving_average()` | scalar |
| Distillation | `DistillWrapper(teacher=, student=DistillableViT(...))` | `loss = w(img, labels)` | scalar; `student.to_vit()` exports a plain ViT |
| Inspection | `Recorder(vit)` / `Extractor(vit)` | `preds, attns = r(img)` / `preds, emb = e(img)` | tuple; `.eject()` returns the bare model |
| Variable-res | `na_vit.NaViT` | `model([[img1, img2], [img3]])` or `model([...], group_images=True)` | `(num_images, num_classes)` |
| Video | `vivit.ViViT`, `vit_3d.ViT`, `simple_vit_3d.SimpleViT` | `model(video)` with video `(b, c, f, h, w)` | logits |
| N-D | `vit_nd*.ViTND(ndim=k, input_shape=..., ...)` | `model(x)` with x `(b, c, *dims)` | logits |
| Robotics | `VAT(vit, ...)`, `VAAT(vit, ast, ...)`, `SigLIPVAT(...)` | `model(video, extra=, tasks=, actions=)` | L1 loss if `actions` is given, else `(b, chunk, dim_action)` |

---

## 6. Model catalog

The models fall into five families.

### 6.1 "Classic" ViT variants (token sequence `(b, n, d)`, cls or mean pooling)

| Module → class | Paper / idea | What differs from `ViT` |
|---|---|---|
| `efficient.ViT` | Efficient-attention shell | Takes any `transformer=` module (Linformer, Nyströmformer, …). |
| `deepvit.DeepViT` | DeepViT | Re-Attention: post-softmax head-mixing matrix + LayerNorm over heads. |
| `cait.CaiT` | CaiT | No cls in the patch stack; `cls_depth` class-attention layers; LayerScale; talking heads; layer-drop. |
| `xcit.XCiT` | XCiT | Cross-covariance (d×d) attention, Local Patch Interaction convs, class-attention head. |
| `t2t.T2TViT` | Tokens-to-Token | Repeated `nn.Unfold` soft-split + small token transformers as the patch embedding. |
| `cross_vit.CrossViT` | CrossViT | Two branches at two patch sizes; cls tokens cross-attend to the other branch. |
| `pit.PiT` | Pooling-based ViT | Overlapping patches; stride-2 depthwise conv pooling between stages, doubling `dim`. |
| `local_vit.LocalViT` | LocalViT | Depthwise 3×3 conv inside the FFN (cls excluded). |
| `rvt.RvT` | Rotary ViT | 2-D axial RoPE, depthwise-separable conv queries, GEGLU. |
| `parallel_vit.ViT` | "Three things everyone should know" | `num_parallel_branches` attn/FF blocks summed per layer. |
| `vit_for_small_dataset.ViT` | SPT + LSA | Shifted-patch tokenization; locality self-attention (learned temperature, masked diagonal). |
| `ats_vit.ViT` | Adaptive Token Sampling | Per-layer Gumbel sampling of tokens by cls attention; `return_sampled_token_ids`. |
| `vit_with_patch_merger.ViT` | Patch Merger | Learned queries compress tokens to K mid-network. |
| `vit_with_patch_dropout.ViT` | PatchDropout | Random token subset during training (default 0.25). |
| `learnable_memory_vit.ViT` + `Adapter` | Learnable Memory | Frozen backbone + per-layer memory tokens per task; original outputs are unaffected. |
| `jumbo_vit.JumboViT` | Jumbo CLS | Wide cls token split into k tokens for attention, with its own wide FFN. |
| `look_vit.LookViT` | LookupViT | Small "main" token set cross-attends to high-res tokens; reuses the transposed similarity. |
| `jet_vit.JetViT` | JetViT | Per-layer choice of window, linear, or full attention (searchable). |
| `vit_5.ViT` | ViT-5 | RMSNorm, QK-norm, LayerScale, registers, axial RoPE + learned abs-pos. |
| `vit_with_keel_post_ln.ViT` | Keel ("Post-LN is back") | `x = LN(block(x) + α·x)`. |
| `vit_with_decorr.ViT` | Decorrelation | Off-diagonal correlation penalty as aux loss; **always returns `(logits, loss)`**. |
| `vit_detpool.ViTDetPool` | DetPool | Object-mask restricted attention and masked mean pooling; optional frozen mask generator. |
| `looped_vit.LoopedViT` | Looped transformer | Pre-loop → shared looped stack × N → post-loop; deep supervision; learned adaptive exit gate; 1–7-D input. |
| `wwt.WWT` | WWT (slots + tokens hierarchy) | Mutual attention between token and slot sets; carried attention masks; optional SIGReg. Polymorphic return. |
| `normalized_vit.nViT` | nGPT on the hypersphere | L2-normalized weights/activations, lerp residuals, SwiGLU; **call `model.norm_weights_()` after each optimizer step**. |

### 6.2 `SimpleViT` research ablations (one idea per file)

All of these copy `simple_vit.py` and change a single mechanism:

| Module | Idea |
|---|---|
| `simple_vit_with_register_tokens` | Register tokens, packed in and discarded before pooling. |
| `simple_vit_with_patch_dropout` | PatchDropout (default 0.5). |
| `simple_vit_with_qk_norm` | Per-head RMSNorm on q, k (ViT-22B). ⚠ see §10. |
| `simple_vit_with_value_residual` | First-layer values mixed into every layer (learned per-token gate). |
| `simple_vit_with_fft` | Second token stream from `fft2(img)` patches. |
| `simple_vit_with_specialized_cls` | Separate norms and early QKV for cls vs patch tokens. |
| `simple_vit_orthog_residual_update` | Only the component orthogonal to the stream is added (fp64 by default). |
| `simple_vit_with_hyper_connections` | Multi-stream residual with dynamic width/depth connections. |
| `simple_vit_attn_residual` (`SimpleViTAttnResidual`) | Attention over depth replaces additive residuals; supports `history`. |
| `attn_residual_with_dca_simple_vit` | Depth attention with separate q/k/v mixes (DeepCrossAttention) + grouped/gated variants. |
| `simple_vit_with_sparse_gating` (`SimpleViTWithSparseGating`) | Sigmoid key/query gates in log-space + sparsity aux loss; registers as sinks. |
| `simple_vit_with_relative_proj_pos_bias` | Content-dependent relative position bias from a learned distance basis. |
| `simple_flash_attn_vit`, `simple_flash_attn_vit_3d` | `F.scaled_dot_product_attention` with GPU-dependent kernel selection. ⚠ non-flash path broken. |
| `simple_uvit` (`SimpleUViT`) | U-ViT long skip connections (first-half → second-half). |

### 6.3 Hierarchical / convolutional hybrids (feature maps `(b, c, h, w)`)

These work on feature maps, use 1×1 convs as linears, take position from convs or relative biases, downsample between stages, and finish with global average pooling. Most take **per-stage tuples** for `dim`/`depth`/`heads`.

| Module → class | Key mechanism |
|---|---|
| `cvt.CvT` | Strided conv embeddings; depthwise-conv QKV projections; 3 stages (`s1_*`, `s2_*`, `s3_*` kwargs). |
| `levit.LeViT` | Conv stem, BatchNorm, Hardswish, attention-based downsampling, optional distillation head. |
| `nest.NesT` | Local attention inside nested blocks; conv + maxpool aggregation. |
| `twins_svt.TwinsSVT` | Alternating locally-grouped and global sub-sampled attention; PEG (`s1_*`…`s4_*` kwargs). |
| `crossformer.CrossFormer` | Multi-kernel cross-scale embeddings; short/long-distance attention; dynamic position bias. |
| `regionvit.RegionViT` | Regional + local token streams (regional-to-local attention). |
| `scalable_vit.ScalableViT` | Scalable self-attention + interactive windowed attention. |
| `sep_vit.SepViT` | Depthwise (window) + pointwise (window-token) separable attention. |
| `max_vit.MaxViT`, `max_vit_with_registers.MaxViT` | MBConv + block attention + grid attention (± registers). |
| `mobile_vit.MobileViT` | MobileNetV2 + MobileViT blocks (unfold → transformer → fold). Positional args; `channels` is a *list of widths*. |
| `cct.CCT`, `cct_3d.CCT` + `cct_2 … cct_16` factories | Conv tokenizer + SeqPool; 3-D version for video. |

`cvt` and `twins_svt` parse per-stage kwargs by name prefix from `locals()` (`group_by_key_prefix_and_remove_prefix`).

### 6.4 Other dimensionalities, variable resolution, video

| Module → class | Input | Notes |
|---|---|---|
| `vit_1d.ViT`, `simple_vit_1d.SimpleViT` | `(b, c, seq_len)` | Copy of vit/simple_vit with 1-D patterns. |
| `vit_3d.ViT`, `simple_vit_3d.SimpleViT` | `(b, c, f, h, w)` | Tubelet patches (`frame_patch_size`, `image_patch_size`). |
| `vivit.ViViT` | `(b, c, f, h, w)`, optional frame `mask` | `variant='factorized_encoder'` (spatial then temporal) or `'factorized_self_attention'`. |
| `vivit_with_moss.ViViT`, `MOSS` | video | High-order space-time self-similarity features (MOSS) added to cls; `MOSS` is reusable. ⚠ see §10. |
| `vit_nd.ViTND` | `(b, c, *dims)`, `ndim ∈ 1..7` | einops patterns generated from axis letters; learned pos-emb; per-axis causal masks. |
| `vit_nd_rotary.ViTND` | same | Golden-Gate N-d RoPE (no abs-pos, so any input size); `muon_parameters()`. |
| `vit_nd_pope.ViTND` | same | PoPE (polar position embedding). |
| `vit_nd_masking` | helper | `get_nd_causal_mask_fn`, `create_nd_block_mask`, `nd_attention`; flex-attention with a dense fallback (`flex_attention_supported`). |
| `na_vit.NaViT` | `List[Tensor]` or `List[List[Tensor]]` of `(c, h, w)` | **Patch n' Pack.** Images are concatenated per row, padded, and given block-diagonal masks; `group_images=True` greedily packs to `group_max_seq_len`. Factorized h/w pos-emb, qk-RMSNorm, per-image attention pooling, token dropout. |
| `na_vit_nested_tensor.NaViT`, `…_3d.NaViT` | `List[(c, h, w)]` / `List[(c, f, h, w)]` | Same idea using `torch.nested` jagged tensors instead of padding and masks. ⚠ see §10. |
| `accept_video_wrapper.AcceptVideoWrapper` | `(b, c, t, h, w)` | Folds time into batch, runs any image model, adds time pos-emb and optional MOSS. |

### 6.5 Vision-action models (robotics)

`vat.py`, `vaat.py` and `vat_siglip.py` are near-duplicates that share one design:

```
encoder(s) ──per-layer hidden states──┐
                                       ▼
query tokens = [registers | advantage token | action_chunk_len action tokens | extra(proprio) token]
   for i in depth:
       FiLM(task)  →  cross-attn to encoder layer vit_layer_indices[i] (+ time & view embeddings)
                   →  (optional) self-attn  →  FF        [sigmoid-gated attention heads]
   action tokens → Linear → actions (b, action_chunk_len, dim_action);  L1 loss if `actions` given
```

- **VAT** uses its own `vat.ViT` (which returns per-layer hiddens).
- **VAAT** adds an Audio Spectrogram Transformer (`AST`, via torchaudio) with a second cross-attention per layer.
- **SigLIPVAT** embeds a SigLIP-so400m re-implementation, with `load_siglip()` pulling HF weights (`huggingface_hub`, `safetensors`).

---

## 7. Training / self-supervised wrappers and the encoder contract

The wrappers use one of two integration styles.

**A. Structural duck-typing.** The wrapper reaches into the encoder's internals:

| Wrapper | Accesses on `encoder` | Returns |
|---|---|---|
| `MAE(encoder, decoder_dim, masking_ratio=0.75, decoder_depth=1, ...)` | `pos_embedding.shape[-2:]`, `to_patch_embedding[0]`, `[1:]`, `[2].weight`, `pool`, `transformer` | MSE on masked pixels |
| `SimMIM(encoder, masking_ratio=0.5)` | same, always `pos_embedding[:, 1:n+1]` | L1 on masked pixels |
| `MPP(transformer, patch_size, dim, ...)` | `to_patch_embedding[1:]`, 3-D `cls_token`, `pos_embedding[:, :n+1]`, `dropout`, `transformer` | CE over 512 quantized colours |
| `DistillWrapper` + `Distillable{ViT,T2TViT,EfficientViT}` | subclasses that append a distillation token; `cls_token` repeated with the 2-D pattern `'n d -> b n d'` | `(1-α)·CE + α·KD` |
| `mp3.MP3(vit, masking_ratio)` | only its own `mp3.ViT` (needs `transformer(x, context)`) | CE over patch positions |

**B. Hook-based.** The wrapper only needs a submodule name or index:

| Wrapper | Notes |
|---|---|
| `Dino(net, image_size, hidden_layer=-2, ...)` | Forward hook on `children()[hidden_layer]` or a named module; student sees local crops, EMA teacher sees global crops; centering + temperature-sharpened CE (`loss_fn`, tested). `update_moving_average()` after each step. |
| `EsViTTrainer(...)` | Same as Dino plus region-level matching loss; hooked layer **must output `(b, c, h, w)`** (e.g. CvT), not tokens. |
| `LeJEPA(net, image_size, ...)` | Local crops → encoder; global crops → stop-grad (or EMA) target; MSE + SIGReg (`sigreg_loss`). |
| `LeVJEPA(encoder, ...)` + `levjepa.VideoTransformer` | Video JEPA with block-causal flex attention and token dropping; ships its own encoder. |
| `LeVJEPAND` + `NDTransformer` | N-D generalization of LeVJEPA. |

### 7.1 Compatibility matrix (static analysis)

`pos_embedding` / `cls_token` layouts:

- **"2-D"** means `(N+cls, dim)` / `(cls, dim)`: current `vit.ViT`.
- **"3-D"** means `(1, N+1, dim)` / `(1, 1, dim)`: `efficient.ViT`, `t2t.T2TViT`, `deepvit`, `parallel_vit`, and other older files.

| Encoder ↓ / Wrapper → | MAE | SimMIM | MPP | Distill | Dino / LeJEPA | Recorder | Extractor |
|---|---|---|---|---|---|---|---|
| `vit.ViT(pool='cls')` (default) | ❌ | ❌ | ❌ | ✅ (`DistillableViT`) | ✅ | ✅ | ✅ |
| `vit.ViT(pool='mean')` | ✅ | ❌ | ❌ | — | ✅ | ✅ | ✅ |
| `SimpleViT` | ✅ (mean path) | ❌ | ❌ | — | ✅ | ⚠ returns `None` attns | ✅ |
| 3-D legacy (`deepvit`, `parallel_vit`, …) | ✅ | ✅ | ✅* | — | ✅ | ⚠ | ✅ |
| `t2t.T2TViT` / `efficient.ViT` | — | — | — | ❌ (3-D cls vs 2-D pattern) | ✅ | ✅ (t2t reuses `vit.Transformer`) | ✅ |

\* MPP also needs a `.dropout` attribute, which `efficient.ViT` lacks.

**Root cause.** Commit `fdaf7f9` (2025-11-27, "fix positional embed for mean pool case and cleanup") changed `vit.ViT.pos_embedding` from `(1, N+1, dim)` to `(N+num_cls, dim)` and `cls_token` to `(num_cls, dim)`. The structural wrappers were not updated. As a result `pos_embedding[:, 1:n+1]` now slices the **feature** axis. **The README's own MAE example (`ViT(...)` with the default `pool='cls'` → `MAE(encoder=v, ...)`) should no longer run.** Only the `pool='mean'` path in MAE is correct. No test covers any wrapper except `dino.loss_fn`.

---

## 8. Inspection utilities: `Recorder` and `Extractor`

```python
from vit_pytorch.recorder import Recorder
v = Recorder(v)
preds, attns = v(img)        # attns: (b, depth, heads, n, n) — post-softmax, pre-dropout
v = v.eject()                # remove hooks, get the original model back

from vit_pytorch.extractor import Extractor
v = Extractor(v)             # layer_name='transformer' by default
logits, embeddings = v(img)  # embeddings: (b, n+1, dim) post-final-norm tokens
```

- `Recorder` finds attention modules with **`isinstance(m, vit_pytorch.vit.Attention)`** and hooks `.attend`. It therefore only works with models built from `vit.py`'s `Attention` (`ViT`, `T2TViT`, `Distillable*`). With any other model it silently returns `attns=None`.
- `Extractor` hooks any named module (`layer_name=`) or an explicit `layer=`. It can save the module's input instead of its output, and supports `return_embeddings_only`. It is the portable choice.

---

## 9. Dependencies, packaging, tests, tooling

**Declared dependencies** (`pyproject.toml`): `torch>=2.5`, `torchvision`, `einops>=0.8.2`. Python `>=3.8` is declared, but several files use `X | None` annotations (3.10+ unless `from __future__ import annotations` is present). The test extra pins `torch==2.5.0`, `torchvision==0.20.0`.

**Undeclared, used by specific modules:**

| Package | Used by |
|---|---|
| `torchaudio` | `vaat.py` (top-level import, so the module won't import without it) |
| `huggingface_hub`, `safetensors` | `vat_siglip.load_siglip` (lazy) |
| `packaging` | `simple_flash_attn_vit*.py`, `na_vit_nested_tensor*.py` |
| `linformer`, `sklearn`, `pandas`, `matplotlib`, `tqdm`, PIL | `examples/cats_and_dogs.ipynb` |
| `accelerate`, `wandb` | `train_vit_decorr.py` (declared inline via PEP-723 header) |

**PyTorch feature requirements:** `F.scaled_dot_product_attention` (many files), `nn.RMSNorm` (≥2.4), `torch.nn.attention.flex_attention` (vit_nd*, levjepa*; LeVJEPA has **no** dense fallback), `torch.nested` jagged layout (na_vit_nested_tensor*), `torch.backends.cuda.sdp_kernel` (deprecated; simple_flash_attn_vit).

**Tests**: `pytest`, configured in `pyproject.toml`:
- `tests/test_vit.py`: one ViT forward-shape check.
- `tests/test_dino_loss.py`: three numeric tests that `dino.loss_fn` matches a `log_softmax` reference and stays finite in fp16/bf16.

That is the whole suite, for 85 modules. There is no CI config in this fork (`.github/` is absent), no linter, no type checking. The inline `__main__` blocks are the only other executable checks.

**Training script** `train_vit_decorr.py`: trains `vit_with_decorr.ViT` (dim 128, depth 6, patch 4) on CIFAR-100 (auto-download to `./data`) with Adam 3e-4 for 10 epochs; loss = CE + 0.1·decorr. It uses `accelerate`, and `wandb` is disabled unless `TRACK_EXPERIMENT_ONLINE=True`. There is no eval loop or checkpointing.

---

## 10. Known issues (verified by reading the source)

Every item below was confirmed by reading the cited lines. Items marked **▶ reproduced** were also triggered at runtime (torch 2.5.0+cu124, einops 0.8.2), with the error shown. The rest have not been executed.

### 10.1 High impact (wrong results or a crash on the default / documented path)

| # | Location | Issue |
|---|---|---|
| 1 | `mae.py:53`, `simmim.py:46`, `mpp.py` | Index `pos_embedding[:, 1:n+1]` and use 3-D `cls_token`; incompatible with the 2-D layout `vit.ViT` has used since `fdaf7f9`. `MAE(ViT(pool='cls'))`, the README example, breaks. See §7.1. **▶ reproduced:** MAE and SimMIM raise `RuntimeError: size of tensor a (64) must match … b (63)`; `MAE(ViT(pool='mean'))` runs. |
| 2 | `distill.py:28` | `repeat(cls_token, 'n d -> b n d')` fails for `DistillableT2TViT` / `DistillableEfficientViT`, whose `cls_token` is `(1,1,dim)` (`t2t.py:52`, `efficient.py:26`). **▶ reproduced** (`DistillableT2TViT`: `EinopsError`). |
| 3 | `simple_vit_with_qk_norm.py:129` | `self.linear_head = nn.LayerNorm(dim)`: no projection to `num_classes`; outputs `(b, dim)`. **▶ reproduced:** `num_classes=10, dim=64` gives output shape `(2, 64)`. |
| 4 | `vivit_with_moss.py:303,337,405,416` | Default `moss_causal=True` makes the temporal transformer causal while the cls token sits at position 0 and is read out with `x[:, 0]`, so with `pool='cls'` the logits cannot depend on the video. **▶ reproduced:** two random videos give identical logits in eval mode. |
| 5 | `na_vit_nested_tensor.py:150,237` (also `_3d`) | `token_dropout_prob=None` by default, but `self.token_dropout_prob > 0` is evaluated in training mode → `TypeError`. Also builds inputs with `nested_tensor(...)` (L274/289), which creates a new leaf and likely cuts gradients to patch/pos embeddings and pool queries. **▶ reproduced** (the `TypeError`; the gradient issue was not tested). |
| 6 | `scalable_vit.py:216-228` | Layers are built as `[ssa, ff, peg, ff, iwsa]` but unpacked as `ssa, ff1, peg, iwsa, ff2`. The effective order is SSA→FF→PEG→FF→IWSA (no FF after windowed attention). Runs silently. |
| 7 | `local_vit.py:45,56` | `LayerNorm(dim)` applied after rearranging to `b c h w`, so it normalizes the width axis. This crashes unless grid width equals `dim`. **▶ reproduced** (`RuntimeError: Given normalized_shape=[64] … got input of size [2, 64, 8, 8]`). |
| 8 | `vat.py:376`, `vaat.py:566`, `vat_siglip.py:360` | `rearrange(x, 'b 1 c h w')` has no `->`, so the single-view 4-D input path crashes. |
| 9 | `vat.py:38-44` (+ vaat, vat_siglip) | FiLM zero-inits both γ and β and applies `tokens*γ + β`. With `tasks` passed, the token stream is zeroed at init (the usual form is `tokens*(1+γ)+β`). |
| 10 | `simple_flash_attn_vit.py:74,82`, `…_3d.py:69,77` | Non-flash path calls `einops.einsum("pattern", q, k)` with the pattern *first*, but einops requires it *last*; the pattern itself (`b j d` for 4-D k/v) is also wrong. `use_flash=False` (and non-CUDA fallbacks) fail. The latest commit (`53334b3`) fixed the missing `einsum` import but not the call. **▶ reproduced** (`ValueError: The last argument passed to einops.einsum must be a string`). |
| 11 | `levjepa.py` (~L421) | Concatenates global views (`image_size`) and local views (`local_size`), so it crashes unless the two are equal. Batched torchvision transforms give every view the same crop. |

### 10.2 Medium (wrong hyper-parameter plumbing, silent no-ops)

- `sep_vit.py:274`: per-stage `window_size` and user `dim_head` are never passed to `Transformer`, so the defaults (7, 32) are always used.
- `xcit.py:47`: `elif 18 > depth <= 24` can never be true; LayerScale init is 1e-6 for every layer deeper than 18.
- `mobile_vit.py:206`: 4th stem block is `MV2Block(channels[2], channels[3])`; it should take `channels[3]` as input (works only because the README config has equal values).
- `cct.py:270`, `cct_3d.py:299`: `self.n_channels` is undefined → `AttributeError` with `positional_embedding='none'` on short inputs. `cct_3d.py` also passes `stochastic_depth=` (a misspelling, silently swallowed by `**kwargs`).
- `jet_vit.py:237-242`: a `list` `attn_layer` leaves `key` unbound. All three attention variants are instantiated in every layer (~3× parameters).
- `vit_for_small_dataset.py:110`: raw `patch_size` is passed to `SPT`, so a tuple patch size crashes.
- `rvt.py:29`: `int(sqrt(seq_len))` assumes a square grid; `emb_dropout` is accepted but unused.
- `vivit.py:68`, `vivit_with_moss.py:93`: SDPA `dropout_p` is not gated on `self.training`, so attention dropout stays active in eval.
- `vit_nd_pope.py:84`: in-place `uniform_` on a leaf Parameter outside `no_grad`, so `pope_init_learned_bias_uniform=True` should raise.
- `es_vit.py`: region `L2Norm` normalizes `dim=1` (token axis); still uses `softmax`→`log(+eps)` (the numerically safer `log_softmax` fix landed only in `dino.py`).
- `dino.py`, `es_vit.py`, `lejepa.py`: MLP `is_last` check never true (extra GELU after the last hidden layer); `num_classes_K=65336` (likely meant 65536); forward hooks are never removable.
- `crossformer.py` (~L114/146): the relative-position index uses stride `2w-1` while the bias table spans `2w+1`, so offsets read the wrong bias entries.
- `simple_vit_with_specialized_cls.py:54`: `if is_tensor:` tests the function object (always true).
- `distill.py:56,74,92`: `self.dim = kwargs['dim']` → positional `dim` raises `KeyError`.
- `vit_detpool.py:79`: `~mask` assumes a bool mask.
- `max_vit.py` / `max_vit_with_registers.py`: `Dropsample` builds `torch.FloatTensor((b,1,1,1))`, a 4-element tensor rather than a mask of that shape (dead code; never enabled).

### 10.3 Systemic

- **Duplication drift.** The same helper exists in up to ~30 copies with diverging details: patch-embed with or without LayerNorms, final norm present or absent, `project_out` shortcut, `posemb_sincos_2d` in four variants, `pos_embedding.to(x.dtype)` vs `img.dtype`. Bugs are duplicated too (VAT ×3, flash ×2, CCT ×2, Dropsample ×2, NaViT-nested ×2).
- **Fixed-resolution assumptions.** Learned or unsliced position tables, square-grid `sqrt(n)` tricks, and `channels` hard-coded to 3 in `cait`, `xcit`, `levit`, `twins_svt`, `regionvit` (local stream), `mobile_vit`, `vat_siglip`.
- **Stochasticity ignoring `self.training`.** CaiT/XCiT layer-drop, ATS sampling, JetViT layer selection, SIGReg in WWT.
- **README drift.** The table of contents is missing several sections (Accessing Embeddings, JetViT, ViT-5) and some orderings differ. The ViViT snippet imports `from vit_pytorch.vivit import ViT`, but the class is `ViViT`.
- **Test coverage** is essentially nil outside `ViT` and the DINO loss.

---

## 11. Practical guidance

**Using the library**

- For a plain classifier or backbone, use `vit.ViT` (with `num_classes=0` for features) or `simple_vit.SimpleViT`. These are the most-used and most-maintained files.
- For MAE pre-training today, use `ViT(..., pool='mean')` or `SimpleViT` as the encoder, or patch `mae.py:53` to `pos_embedding[1:num_patches+1]`.
- For attention maps, use `Recorder` only with `vit.ViT`-family models; otherwise use `Extractor` with a custom hook or read the module.
- For variable-resolution input, prefer `na_vit.NaViT` (padding + masks) over the nested-tensor variants until §10.1 #5 is fixed.
- Treat any non-reference file as **a readable starting point for a paper, not a battle-tested implementation**. Run its `__main__` block and a quick overfit test before relying on it.

**Reading the code**

1. Start with `vit.py` (138 lines); everything is a diff against it.
2. Then `simple_vit.py`, and diff any `simple_vit_with_*.py` against it to isolate the idea (`diff vit_pytorch/simple_vit.py vit_pytorch/simple_vit_with_register_tokens.py`).
3. For hierarchical models, read `cvt.py` (stage-prefix kwargs and conv projections), then `max_vit.py`.
4. For training wrappers, read `mae.py` (structural contract) and `dino.py` (hook contract).
5. For N-D and causal masking, read `vit_nd_masking.py` then `vit_nd.py`.

**Improving the codebase**, in rough priority order:

1. Add a parametrized smoke test that builds every exported/README-documented model and runs a tiny forward (and backward) pass. Most of §10.1 would surface immediately.
2. Restore the encoder contract for MAE/SimMIM/MPP/Distill against the 2-D `vit.ViT` layout (or accept both via `pos_embedding.ndim`).
3. Fix the one-line bugs: `qk_norm` head, `scalable_vit` unpack order, `sep_vit` kwargs, `xcit` condition, the VAT `rearrange`, the flash einsum pattern, the `na_vit_nested_tensor` `None > 0` comparison, and `vivit_with_moss` causal cls.
4. Declare optional extras in `pyproject.toml` (`audio = [torchaudio]`, `siglip = [huggingface_hub, safetensors]`, plus `packaging`).
5. Optionally factor shared primitives (`Attention`, `FeedForward`, sincos embeddings, `EMA`/`NetWrapper`) into a private `_layers.py`. This conflicts with the "one self-contained file per paper" philosophy, so it's a judgment call; tests (item 1) give most of the benefit without that trade-off.

---

## 12. Study plan

A staged path from "what is a ViT" to "I can read any file here and spot what's wrong with it". Each stage lists what to read, what to do, and a checkpoint to test yourself on before moving on. The time estimates assume roughly 1–2 hours of focused work per session. Total: about 10–12 sessions.

### Stage 0: setup (½ session)

- **Environment:** `pip install -e '.[test]' jupyter`, then run `pytest`. All 5 tests should pass (4 test functions; one is parametrized over fp16/bf16). A CPU is enough for every exercise up to Stage 6.
- **Make a scratch folder** outside `vit_pytorch/` (e.g. `scratch/`) for your exercise scripts, so the package stays clean.
- **Learn just enough einops:** `rearrange`, `repeat`, `reduce`, `pack`/`unpack`, and the `Rearrange` layer. The [einops basics tutorial](https://einops.rocks/1-einops-basics/) covers this. Every file in this repo depends on it.

> ✅ **Checkpoint:** without running it, write the einops pattern that turns `(b, 3, 224, 224)` into 16×16 patches of shape `(b, 196, 768)`. Then compare it with `vit.py`'s `to_patch_embedding[0]`.

### Stage 1: the reference ViT (2 sessions)

- **Read:** the ViT paper (Dosovitskiy et al., 2020), sections 3.1 and 4.1. Then read `vit_pytorch/vit.py` line by line alongside [§4.1](#41-vit_pytorchvitpy--vit).
- **Do:**
  1. Instantiate a tiny `ViT` (`dim=64, depth=2, heads=4, image_size=32, patch_size=4`). Use forward hooks or `print` statements to record the tensor shape after every step in the §4.1 diagram.
  2. Count parameters by hand for `Attention` and `FeedForward`, then check against `sum(p.numel() for p in m.parameters())`.
  3. Try `num_classes=0` and `pool='mean'` and explain how the output shape changes and why the `cls_token` becomes `(0, dim)`.
  4. Feed a 16×16 image into a model built for 32×32. Explain why it works, given `pos_embedding[:seq]`, and why a 64×64 image doesn't.
  5. Train it on CIFAR-10 for a few epochs, or overfit a single batch of 32 images until loss ≈ 0. This shows the model actually learns.
- **Read next:** `simple_vit.py`. Run `diff vit_pytorch/vit.py vit_pytorch/simple_vit.py` and list every difference: positional embedding, pooling, dropout, biases, head.

> ✅ **Checkpoint:** explain why the norm sits *inside* `Attention`/`FeedForward` (pre-norm) and why there is one more `LayerNorm` at the end of `Transformer`. What does `to_latent` exist for? (Hint: §8 and `dino.py`.)

### Stage 2: inspection tools and the "one idea per file" pattern (1 session)

- **Read:** `recorder.py` and `extractor.py` (about 150 lines together), then [§8](#8-inspection-utilities-recorder-and-extractor).
- **Do:**
  1. Wrap your Stage-1 model in `Recorder`. Plot the cls token's attention over the patches for one image, as a heat-map reshaped to the patch grid.
  2. Wrap a `SimpleViT` in `Recorder`. You should get `attns=None`; find the `isinstance` line that causes it.
  3. Diff `simple_vit.py` against three ablations: `simple_vit_with_register_tokens.py`, `simple_vit_with_qk_norm.py` and `simple_vit_with_value_residual.py`. For each, write one sentence describing the idea and one naming the exact lines that implement it.
- **Bug hunt:** in the `qk_norm` diff, find why the model never outputs `num_classes` logits (§10.1 #3), and fix it locally.

> ✅ **Checkpoint:** given any `simple_vit_with_*.py`, you can locate the changed mechanism in under 5 minutes using `diff`.

### Stage 3: classic ViT variants (2 sessions)

Read these in order. Each one adds a single new concept on top of what you already know:

| Order | File | Concept to extract |
|---|---|---|
| 1 | `efficient.py` | Dependency injection of the transformer. |
| 2 | `vit_for_small_dataset.py` | Changing the tokenizer (SPT) and attention temperature (LSA). |
| 3 | `deepvit.py` | Modifying the attention matrix after softmax. |
| 4 | `cait.py` | Class-attention, LayerScale, layer-drop. |
| 5 | `t2t.py` | `nn.Unfold` as a tokenizer; reuse of `vit.Transformer`. |
| 6 | `cross_vit.py` | Multi-branch models; cross-attention between cls tokens. |
| 7 | `ats_vit.py` | Dynamic token count, padding masks, `pad_sequence`. |
| 8 | `learnable_memory_vit.py` | Freezing a backbone; attention masks as an API. |

- **Do:** for 2–3 of these, write the shapes at each step, just as you did in Stage 1. For `ats_vit.py`, call it with `return_sampled_token_ids=True` and check that the token count shrinks at each depth.
- **Bug hunt:** find the `xcit.py:47` condition that can never be true, and the tuple-`patch_size` crash in `vit_for_small_dataset.py` (§10.2).

> ✅ **Checkpoint:** explain the difference between a cls token taking part in every layer (ViT) and a cls token only attending in the last layers (CaiT), and give one reason the paper prefers the latter.

### Stage 4: hierarchical and convolutional hybrids (2 sessions)

- **Read first:** [§6.3](#63-hierarchical--convolutional-hybrids-feature-maps-b-c-h-w). The mental switch is from token sequences `(b, n, d)` to feature maps `(b, c, h, w)`, where 1×1 convolutions act as linear layers.
- **Read in order:**
  1. `pit.py`: the smallest step, pooling between ViT stages.
  2. `cvt.py`: conv projections, plus the `group_by_key_prefix_and_remove_prefix` kwargs trick.
  3. `twins_svt.py`: local vs global attention, and PEG.
  4. `max_vit.py`: block vs grid attention (compare the two `rearrange` patterns closely), MBConv, and relative position bias.
- **Do:**
  1. For `MaxViT`, print the feature-map shape after each stage for a 224×224 input.
  2. Draw (on paper) which pixels a token can attend to in block attention vs grid attention, with `window_size=7`.
- **Bug hunt:** compare the `layers.append` list and the `for` unpacking in `scalable_vit.py:216-228`, then work out the effective block order (§10.1 #6). Then find why `sep_vit.py`'s per-stage `window_size` is ignored.

> ✅ **Checkpoint:** explain how each of CvT, Twins, CrossFormer and MaxViT gets positional information without a learned absolute positional embedding.

### Stage 5: training wrappers and the encoder contract (2 sessions)

- **Read:** the MAE paper (He et al., 2021), section 3. Then `mae.py`, `simmim.py` and [§7](#7-training--self-supervised-wrappers-and-the-encoder-contract).
- **Do (this is the most valuable exercise in the plan):**
  1. Run the README's MAE snippet. It should fail (§10.1 #1). Read the traceback and connect it to `git show fdaf7f9 -- vit_pytorch/vit.py`.
  2. Fix `mae.py` so it works with **both** the 2-D (`vit.ViT`) and 3-D (`deepvit.DeepViT`) `pos_embedding` layouts. Write a pytest that builds each and checks that `loss.backward()` runs.
  3. Pre-train a tiny ViT with MAE on CIFAR-10 for a few epochs. Then visualize the reconstruction of masked patches; you'll need to return `pred_pixel_values` yourself.
- **Then read:** the DINO paper (Caron et al., 2021), section 3, followed by `dino.py`. Focus on `NetWrapper` (the hook), `EMA`/`update_moving_average`, centering, and `loss_fn`. Read `tests/test_dino_loss.py` to see why `log_softmax` replaced `softmax→log`.
- **Bug hunt:** compare `dino.py`'s loss with `es_vit.py`'s. Find the numerical-stability fix that was applied to only one of them, and the `is_last` condition that is never true in the shared `MLP`.

> ✅ **Checkpoint:** explain why the hook-based wrappers (Dino, Extractor) survived the `pos_embedding` refactor while the structural ones (MAE, SimMIM, MPP) broke. What would a stable encoder interface look like?

### Stage 6: beyond 2-D images (2 sessions, pick the branches you care about)

- **Video:** read `vit_3d.py` (a diff against `vit.py`), then `vivit.py`. Compare the two ViViT variants, and run one with a frame `mask`.
  - **Bug hunt:** `vivit_with_moss.py`, where a causal temporal transformer is combined with a cls token at position 0 (§10.1 #4). Prove it empirically: two different videos give identical logits.
- **Variable resolution:** read the NaViT paper (section 2), then `na_vit.py`. Feed it `[[img_64x64, img_32x48], [img_128x128]]` and print the block-diagonal attention mask.
  - **Bug hunt:** construct `na_vit_nested_tensor.NaViT` with defaults and call it in training mode (§10.1 #5).
- **N-D and masking:** read `vit_nd_masking.py`, then `vit_nd.py`. Use `ndim=3` with `causal_dims=(0,)` and visualize the mask from `get_nd_causal_mask_fn` for a tiny grid.
  - Then read `vit_nd_rotary.py` and explain why it accepts any input size while `vit_nd.py` does not.
- **Modern training-stability ideas (optional):** `normalized_vit.py` (nGPT), `vit_5.py`, `simple_vit_with_hyper_connections.py` and `looped_vit.py`. Read each alongside its paper's abstract and method figure.

> ✅ **Checkpoint:** for one model per branch, state the input format, how positions are encoded, how attention is masked, and how the output is pooled.

### Stage 7: capstone (1–2 sessions)

Pick one of these:

1. **Test-suite capstone.** Write `tests/test_smoke.py`: a parametrized test that builds each README-documented model at tiny size and runs forward + backward on CPU. Triage every failure against §10, and open a fix-up commit for 2–3 of them.
2. **Research capstone.** Implement a new single-idea ablation as `simple_vit_with_<idea>.py`, following the repo conventions (§3): a paper link at the top, keyword-only constructor, a `__main__` smoke test, and a README section. Train it against `SimpleViT` on CIFAR-10 with the same budget and compare.
3. **Refactor capstone.** Prototype a private `_layers.py` with shared `Attention`, `FeedForward` and sincos embeddings. Port 3 files to it, and write a short note on whether it is worth it given the one-file-per-paper philosophy.

> ✅ **Final checkpoint:** open a file you've never read (e.g. `wwt.py` or `vat.py`). Within ~30 minutes you should be able to sketch its data flow, list its constructor's important knobs, and name at least one thing that looks suspicious.

### Reference reading list

Every paper the repo implements or cites, plus a few closely related ones, grouped by the stage where they're most useful.

- **Sources:** the README's [Citations](README.md#citations) section (68 BibTeX entries) and paper links in the code (`grep -E "arxiv|openreview" vit_pytorch/*.py`).
- **(related):** not in the repo, but directly relevant to something covered in this document or the exercises.
- **Links:** arXiv IDs come from the README or the code. Where the README gives only a Semantic Scholar or OpenReview link, that link is used instead of an ID.
- **★:** core reading for the plan. The rest is reference material to read when you reach that file.

#### Foundations (Stages 1–2)

| | Paper | Link | Repo |
|---|---|---|---|
| ★ | *Attention Is All You Need* (Transformer, post-norm) | [arXiv 1706.03762](https://arxiv.org/abs/1706.03762) | all |
| ★ | *An Image is Worth 16x16 Words* (ViT) | [arXiv 2010.11929](https://arxiv.org/abs/2010.11929) | [`vit.py`](vit_pytorch/vit.py) |
| ★ | *Better plain ViT baselines for ImageNet-1k* (SimpleViT) | [arXiv 2205.01580](https://arxiv.org/abs/2205.01580) | [`simple_vit.py`](vit_pytorch/simple_vit.py) |
| | *Training data-efficient image transformers & distillation through attention* (DeiT) | [arXiv 2012.12877](https://arxiv.org/abs/2012.12877) | [`distill.py`](vit_pytorch/distill.py) |
| | *Dual PatchNorm*: the LayerNorms around the patch embedding in [`vit.py`](vit_pytorch/vit.py) | [arXiv 2302.01327](https://arxiv.org/abs/2302.01327) | [`vit.py`](vit_pytorch/vit.py), [`simple_vit.py`](vit_pytorch/simple_vit.py) |
| ★ | *On Layer Normalization in the Transformer Architecture* (pre-LN vs post-LN, warmup) **(related)** | [arXiv 2002.04745](https://arxiv.org/abs/2002.04745) | your norm ablation |
| | *Understanding the Difficulty of Training Transformers* (why post-LN is unstable, yet better when it trains) **(related)** | [arXiv 2004.08249](https://arxiv.org/abs/2004.08249) | your norm ablation |
| ★ | *Post-LayerNorm Is Back: Stable, ExpressivE, and Deep* (Keel) | [arXiv 2601.19895](https://arxiv.org/abs/2601.19895) | [`vit_with_keel_post_ln.py`](vit_pytorch/vit_with_keel_post_ln.py) |
| | *Decoupled Weight Decay Regularization* (AdamW) **(related)** | [arXiv 1711.05101](https://arxiv.org/abs/1711.05101) | [`train_cifar10.py`](train_cifar10.py) |
| | *Bag of Tricks for Image Classification with CNNs* (no bias decay) **(related)** | [arXiv 1812.01187](https://arxiv.org/abs/1812.01187) | weight-decay groups |
| | Weight decay in LayerNorm: blog post cited in the README | [medium.com/@ohadrubin](https://medium.com/@ohadrubin/exploring-weight-decay-in-layer-normalization-challenges-and-a-reparameterization-solution-ad4d12c24950) | [`normalized_vit.py`](vit_pytorch/normalized_vit.py) |
| | *RoFormer: Enhanced Transformer with Rotary Position Embedding* (RoPE) | [arXiv 2104.09864](https://arxiv.org/abs/2104.09864) | [`rvt.py`](vit_pytorch/rvt.py), [`vit_5.py`](vit_pytorch/vit_5.py), [`vit_nd_rotary.py`](vit_pytorch/vit_nd_rotary.py) |
| | *Quantifying Attention Flow in Transformers* (attention rollout) **(related)** | [arXiv 2005.00928](https://arxiv.org/abs/2005.00928) | [`Recorder`](vit_pytorch/recorder.py), [PR #2](https://github.com/tbvecc-forks/vit-pytorch/pull/2) |
| | *Attention is not Explanation* **(related)** | [arXiv 1902.10186](https://arxiv.org/abs/1902.10186) | caveat for [PR #2](https://github.com/tbvecc-forks/vit-pytorch/pull/2) |

#### Single-idea `SimpleViT` variants (Stage 2)

| | Paper | Link | Repo |
|---|---|---|---|
| ★ | *Vision Transformers Need Registers* | [arXiv 2309.16588](https://arxiv.org/abs/2309.16588) | [`simple_vit_with_register_tokens.py`](vit_pytorch/simple_vit_with_register_tokens.py), [`max_vit_with_registers.py`](vit_pytorch/max_vit_with_registers.py) |
| ★ | *Query-Key Normalization for Transformers* (QKNorm, origin of the idea) **(related)** | [arXiv 2010.04245](https://arxiv.org/abs/2010.04245) | [`simple_vit_with_qk_norm.py`](vit_pytorch/simple_vit_with_qk_norm.py) |
| ★ | *Scaling Vision Transformers to 22 Billion Parameters* (ViT-22B, qk-layernorm) **(related)** | [arXiv 2302.05442](https://arxiv.org/abs/2302.05442) | [`simple_vit_with_qk_norm.py`](vit_pytorch/simple_vit_with_qk_norm.py) |
| | *Small-scale proxies for large-scale Transformer training instabilities* (attention-logit growth at high LR) **(related)** | [arXiv 2309.14322](https://arxiv.org/abs/2309.14322) | qk-norm LR sweep |
| | *Value Residual Learning For Alleviating Attention Concentration In Transformers* | [Semantic Scholar](https://api.semanticscholar.org/CorpusID:273532030) | [`simple_vit_with_value_residual.py`](vit_pytorch/simple_vit_with_value_residual.py) |
| | *PatchDropout: Economizing Vision Transformers Using Patch Dropout* | [arXiv 2208.07220](https://arxiv.org/abs/2208.07220) | [`simple_vit_with_patch_dropout.py`](vit_pytorch/simple_vit_with_patch_dropout.py), [`vit_with_patch_dropout.py`](vit_pytorch/vit_with_patch_dropout.py) |
| | *Revisiting [CLS] and Patch Token Interaction in Vision Transformers* | [arXiv 2602.08626](https://arxiv.org/abs/2602.08626) | [`simple_vit_with_specialized_cls.py`](vit_pytorch/simple_vit_with_specialized_cls.py) |
| | *Revisiting Residual Connections: Orthogonal Updates for Stable and Efficient Deep Networks* | [arXiv 2505.11881](https://arxiv.org/abs/2505.11881) | [`simple_vit_orthog_residual_update.py`](vit_pytorch/simple_vit_orthog_residual_update.py) |
| | *Hyper-Connections* | [arXiv 2409.19606](https://arxiv.org/abs/2409.19606) | [`simple_vit_with_hyper_connections.py`](vit_pytorch/simple_vit_with_hyper_connections.py) |
| | *Attention Residuals* | [arXiv 2603.15031](https://arxiv.org/abs/2603.15031) | [`simple_vit_attn_residual.py`](vit_pytorch/simple_vit_attn_residual.py) |
| | *Multi-Head Attention Residuals* | [arXiv 2607.27230](https://arxiv.org/abs/2607.27230) | [`attn_residual_with_dca_simple_vit.py`](vit_pytorch/attn_residual_with_dca_simple_vit.py) |
| | *DeepCrossAttention: Supercharging Transformer Residual Connections* | [arXiv 2502.06785](https://arxiv.org/abs/2502.06785) | [`attn_residual_with_dca_simple_vit.py`](vit_pytorch/attn_residual_with_dca_simple_vit.py) |
| | *All are Worth Words: A ViT Backbone for Diffusion Models* (U-ViT) | [Semantic Scholar](https://api.semanticscholar.org/CorpusID:253581703) | [`simple_uvit.py`](vit_pytorch/simple_uvit.py) |
| | *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* | [arXiv 2205.14135](https://arxiv.org/abs/2205.14135) | [`simple_flash_attn_vit.py`](vit_pytorch/simple_flash_attn_vit.py), [`simple_flash_attn_vit_3d.py`](vit_pytorch/simple_flash_attn_vit_3d.py) |
| | *Gated Attention for Large Language Models* (headwise sigmoid gates) | [arXiv 2505.06708](https://arxiv.org/abs/2505.06708) | [`looped_vit.py`](vit_pytorch/looped_vit.py), [`vat.py`](vit_pytorch/vat.py), [`vat_siglip.py`](vit_pytorch/vat_siglip.py), [`attn_residual_with_dca_simple_vit.py`](vit_pytorch/attn_residual_with_dca_simple_vit.py) |
| | *Decorrelation Speeds Up Vision Transformers* | [arXiv 2510.14657](https://arxiv.org/abs/2510.14657) | [`vit_with_decorr.py`](vit_pytorch/vit_with_decorr.py), [`train_vit_decorr.py`](train_vit_decorr.py) |
| | *nGPT: Normalized Transformer with Representation Learning on the Hypersphere* | [arXiv 2410.01131](https://arxiv.org/abs/2410.01131) | [`normalized_vit.py`](vit_pytorch/normalized_vit.py) |
| | *Deep Hyperspherical Learning* | [Semantic Scholar](https://api.semanticscholar.org/CorpusID:5104558) | [`normalized_vit.py`](vit_pytorch/normalized_vit.py) |
| ★ | *ViT-5: Vision Transformers for The Mid-2020s* (RMSNorm, QK-norm, LayerScale, registers, RoPE) | [arXiv 2602.08071](https://arxiv.org/abs/2602.08071) | [`vit_5.py`](vit_pytorch/vit_5.py) |
| | *Looped Diffusion Transformer* | [arXiv 2609.40305](https://arxiv.org/abs/2609.40305) | [`looped_vit.py`](vit_pytorch/looped_vit.py) |

#### Classic ViT variants (Stage 3)

| | Paper | Link | Repo |
|---|---|---|---|
| | Linformer / Nyströmformer, used as drop-in efficient attention **(related)** | [arXiv 2006.04768](https://arxiv.org/abs/2006.04768) / [arXiv 2102.03902](https://arxiv.org/abs/2102.03902) | [`efficient.py`](vit_pytorch/efficient.py), [`examples/cats_and_dogs.ipynb`](examples/cats_and_dogs.ipynb) |
| ★ | *Vision Transformer for Small-Size Datasets* (SPT + LSA) | [arXiv 2112.13492](https://arxiv.org/abs/2112.13492) | [`vit_for_small_dataset.py`](vit_pytorch/vit_for_small_dataset.py) |
| | *DeepViT: Towards Deeper Vision Transformer* (Re-Attention) | [arXiv 2103.11886](https://arxiv.org/abs/2103.11886) | [`deepvit.py`](vit_pytorch/deepvit.py) |
| ★ | *Going deeper with Image Transformers* (CaiT) | [arXiv 2103.17239](https://arxiv.org/abs/2103.17239) | [`cait.py`](vit_pytorch/cait.py) |
| ★ | *Tokens-to-Token ViT* | [arXiv 2101.11986](https://arxiv.org/abs/2101.11986) | [`t2t.py`](vit_pytorch/t2t.py) |
| | *CrossViT: Cross-Attention Multi-Scale Vision Transformer* | [arXiv 2103.14899](https://arxiv.org/abs/2103.14899) | [`cross_vit.py`](vit_pytorch/cross_vit.py) |
| | *ATS: Adaptive Token Sampling for Efficient Vision Transformers* | [arXiv 2111.15667](https://arxiv.org/abs/2111.15667) | [`ats_vit.py`](vit_pytorch/ats_vit.py) |
| | *Fine-tuning Image Transformers using Learnable Memory* | [arXiv 2203.15243](https://arxiv.org/abs/2203.15243) | [`learnable_memory_vit.py`](vit_pytorch/learnable_memory_vit.py) |
| | *Three things everyone should know about Vision Transformers* (parallel blocks) | [arXiv 2203.09795](https://arxiv.org/abs/2203.09795) | [`parallel_vit.py`](vit_pytorch/parallel_vit.py) |
| | *Learning to Merge Tokens in Vision Transformers* (Patch Merger) | [arXiv 2202.12015](https://arxiv.org/abs/2202.12015) | [`vit_with_patch_merger.py`](vit_pytorch/vit_with_patch_merger.py) |
| | *XCiT: Cross-Covariance Image Transformers* | [arXiv 2106.09681](https://arxiv.org/abs/2106.09681) | [`xcit.py`](vit_pytorch/xcit.py) |
| | *Simpler Fast Vision Transformers with a Jumbo CLS Token* | [arXiv 2502.15021](https://arxiv.org/abs/2502.15021) | [`jumbo_vit.py`](vit_pytorch/jumbo_vit.py) |
| | *LookupViT: Compressing visual information to a limited number of tokens* | [Semantic Scholar](https://api.semanticscholar.org/CorpusID:271244592) | [`look_vit.py`](vit_pytorch/look_vit.py) |
| | *JetViT: Efficient High-Resolution Vision Transformer with Post-Training Attention Search* | [arXiv 2605.26636](https://arxiv.org/abs/2605.26636) | [`jet_vit.py`](vit_pytorch/jet_vit.py) |
| | *What-Where Transformer: A Slot-Centric Visual Backbone* (WWT) | [arXiv 2605.12021](https://arxiv.org/abs/2605.12021) | [`wwt.py`](vit_pytorch/wwt.py) |
| | *Object-Centric Learning with Slot Attention* (slot softmax) | [arXiv 2006.15055](https://arxiv.org/abs/2006.15055) | [`wwt.py`](vit_pytorch/wwt.py) |

#### Hierarchical and convolutional hybrids (Stage 4)

| | Paper | Link | Repo |
|---|---|---|---|
| ★ | *Rethinking Spatial Dimensions of Vision Transformers* (PiT) | [arXiv 2103.16302](https://arxiv.org/abs/2103.16302) | [`pit.py`](vit_pytorch/pit.py) |
| ★ | *CvT: Introducing Convolutions to Vision Transformers* | [arXiv 2103.15808](https://arxiv.org/abs/2103.15808) | [`cvt.py`](vit_pytorch/cvt.py) |
| ★ | *Twins: Revisiting Spatial Attention Design in Vision Transformers* | [arXiv 2104.13840](https://arxiv.org/abs/2104.13840) | [`twins_svt.py`](vit_pytorch/twins_svt.py) |
| ★ | *MaxViT: Multi-Axis Vision Transformer* | [arXiv 2204.01697](https://arxiv.org/abs/2204.01697) | [`max_vit.py`](vit_pytorch/max_vit.py), [`max_vit_with_registers.py`](vit_pytorch/max_vit_with_registers.py) |
| | *Swin Transformer* (shifted windows; the reference hierarchical ViT) **(related)** | [arXiv 2103.14030](https://arxiv.org/abs/2103.14030) | none, but explains the windowed-attention designs here |
| | *LeViT: a Vision Transformer in ConvNet's Clothing* | [arXiv 2104.01136](https://arxiv.org/abs/2104.01136) | [`levit.py`](vit_pytorch/levit.py) |
| | *LocalViT: Bringing Locality to Vision Transformers* | [arXiv 2104.05707](https://arxiv.org/abs/2104.05707) | [`local_vit.py`](vit_pytorch/local_vit.py) |
| | *Aggregating Nested Transformers* (NesT) | [arXiv 2105.12723](https://arxiv.org/abs/2105.12723) | [`nest.py`](vit_pytorch/nest.py) |
| | *RegionViT: Regional-to-Local Attention* | [arXiv 2106.02689](https://arxiv.org/abs/2106.02689) | [`regionvit.py`](vit_pytorch/regionvit.py) |
| | *CrossFormer: Cross-scale Attention* | [arXiv 2108.00154](https://arxiv.org/abs/2108.00154) | [`crossformer.py`](vit_pytorch/crossformer.py) |
| | *ScalableViT* | [arXiv 2203.10790](https://arxiv.org/abs/2203.10790) | [`scalable_vit.py`](vit_pytorch/scalable_vit.py) |
| | *SepViT: Separable Vision Transformer* | [arXiv 2203.15380](https://arxiv.org/abs/2203.15380) | [`sep_vit.py`](vit_pytorch/sep_vit.py) |
| | *MobileViT* | [arXiv 2110.02178](https://arxiv.org/abs/2110.02178) | [`mobile_vit.py`](vit_pytorch/mobile_vit.py) |
| | *MobileNetV2: Inverted Residuals and Linear Bottlenecks* (MV2 blocks) | [arXiv 1801.04381](https://arxiv.org/abs/1801.04381) | [`mobile_vit.py`](vit_pytorch/mobile_vit.py) |
| | *Escaping the Big Data Paradigm with Compact Transformers* (CCT) | [arXiv 2104.05704](https://arxiv.org/abs/2104.05704) | [`cct.py`](vit_pytorch/cct.py), [`cct_3d.py`](vit_pytorch/cct_3d.py) |

#### Self-supervised and training wrappers (Stage 5)

| | Paper | Link | Repo |
|---|---|---|---|
| ★ | *Masked Autoencoders Are Scalable Vision Learners* (MAE) | [arXiv 2111.06377](https://arxiv.org/abs/2111.06377) | [`mae.py`](vit_pytorch/mae.py) |
| | *SimMIM: A Simple Framework for Masked Image Modeling* | [arXiv 2111.09886](https://arxiv.org/abs/2111.09886) | [`simmim.py`](vit_pytorch/simmim.py) |
| | Masked patch prediction (section 4.6 of the ViT paper) | [arXiv 2010.11929](https://arxiv.org/abs/2010.11929) | [`mpp.py`](vit_pytorch/mpp.py) |
| | *Position Prediction as an Effective Pretraining Strategy* (MP3) **(related)** | [arXiv 2207.07611](https://arxiv.org/abs/2207.07611) | [`mp3.py`](vit_pytorch/mp3.py) |
| ★ | *Emerging Properties in Self-Supervised Vision Transformers* (DINO) | [arXiv 2104.14294](https://arxiv.org/abs/2104.14294) | [`dino.py`](vit_pytorch/dino.py) |
| | *Efficient Self-supervised Vision Transformers for Representation Learning* (EsViT) | [arXiv 2106.09785](https://arxiv.org/abs/2106.09785) | [`es_vit.py`](vit_pytorch/es_vit.py) |
| | *BYOL works even without batch statistics* (weight standardization) | [arXiv 2010.10241](https://arxiv.org/abs/2010.10241) | [`levjepa.py`](vit_pytorch/levjepa.py), [`levjepa_nd.py`](vit_pytorch/levjepa_nd.py) |
| | *BEiT: BERT Pre-Training of Image Transformers* **(related)** | [arXiv 2106.08254](https://arxiv.org/abs/2106.08254) | context for MAE/SimMIM |
| | *DINOv2: Learning Robust Visual Features without Supervision* **(related)** | [arXiv 2304.07193](https://arxiv.org/abs/2304.07193) | context for [`dino.py`](vit_pytorch/dino.py) |
| ★ | *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics* | [arXiv 2511.08544](https://arxiv.org/abs/2511.08544) | [`lejepa.py`](vit_pytorch/lejepa.py), [`wwt.py`](vit_pytorch/wwt.py) |
| | *Weak-SIGReg: Covariance Regularization for Stable Deep Learning* | [arXiv 2603.05924](https://arxiv.org/abs/2603.05924) | [`lejepa.py`](vit_pytorch/lejepa.py) |
| | *LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics* | [arXiv 2608.27395](https://arxiv.org/abs/2608.27395) | [`levjepa.py`](vit_pytorch/levjepa.py), [`levjepa_nd.py`](vit_pytorch/levjepa_nd.py) |

#### Beyond 2-D images: video, N-D, variable resolution, robotics (Stage 6)

| | Paper | Link | Repo |
|---|---|---|---|
| ★ | *ViViT: A Video Vision Transformer* | [arXiv 2103.15691](https://arxiv.org/abs/2103.15691) | [`vivit.py`](vit_pytorch/vivit.py), [`vit_3d.py`](vit_pytorch/vit_3d.py) |
| ★ | *Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution* | [arXiv 2307.06304](https://arxiv.org/abs/2307.06304) | [`na_vit.py`](vit_pytorch/na_vit.py), [`na_vit_nested_tensor.py`](vit_pytorch/na_vit_nested_tensor.py), [`na_vit_nested_tensor_3d.py`](vit_pytorch/na_vit_nested_tensor_3d.py) |
| | *Exploring High-Order Self-Similarity for Video Understanding* (MOSS) | [OpenReview Co6SCyBIjo](https://openreview.net/forum?id=Co6SCyBIjo) | [`vivit_with_moss.py`](vit_pytorch/vivit_with_moss.py), [`accept_video_wrapper.py`](vit_pytorch/accept_video_wrapper.py) |
| | *RLDX-1 Technical Report*: robotics application of MOSS, cited in [`vivit_with_moss.py`](vit_pytorch/vivit_with_moss.py) | [arXiv 2605.03269](https://arxiv.org/abs/2605.03269) | [`vivit_with_moss.py`](vit_pytorch/vivit_with_moss.py) |
| | *On n-dimensional rotary positional embeddings* (Golden-Gate RoPE, blog) | [jerryxio.ng/posts/nd-rope](https://jerryxio.ng/posts/nd-rope/) | [`vit_nd_rotary.py`](vit_pytorch/vit_nd_rotary.py), [`vit_nd_pope.py`](vit_pytorch/vit_nd_pope.py) |
| | *Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings* (PoPE) | [arXiv 2509.10534](https://arxiv.org/abs/2509.10534) | [`vit_nd_pope.py`](vit_pytorch/vit_nd_pope.py) |
| | *VAT: Vision Action Transformer by Unlocking Full Representation of ViT* | [OpenReview TalHOvvLZu](https://openreview.net/forum?id=TalHOvvLZu) | [`vat.py`](vit_pytorch/vat.py), [`vaat.py`](vit_pytorch/vaat.py), [`vat_siglip.py`](vit_pytorch/vat_siglip.py) |
| | *π0.6: a VLA That Learns From Experience* (advantage conditioning) | [arXiv 2511.14759](https://arxiv.org/abs/2511.14759) | [`vat.py`](vit_pytorch/vat.py), [`vat_siglip.py`](vit_pytorch/vat_siglip.py) |
| | *AST: Audio Spectrogram Transformer* | [arXiv 2104.01778](https://arxiv.org/abs/2104.01778) | [`vaat.py`](vit_pytorch/vaat.py) |
| | *Sigmoid Loss for Language Image Pre-Training* (SigLIP) **(related)** | [arXiv 2303.15343](https://arxiv.org/abs/2303.15343) | [`vat_siglip.py`](vit_pytorch/vat_siglip.py) |
| | *Learning to Grasp Anything By Playing with Random Toys* (DetPool) | [OpenReview NZDaMcpXZm](https://openreview.net/forum?id=NZDaMcpXZm) | [`vit_detpool.py`](vit_pytorch/vit_detpool.py) |

**Repo files without a paper:** [`simple_vit_with_sparse_gating.py`](vit_pytorch/simple_vit_with_sparse_gating.py) (no citation), [`simple_vit_with_relative_proj_pos_bias.py`](vit_pytorch/simple_vit_with_relative_proj_pos_bias.py) (attributed to Thinking Machines' "Inkling" model in a docstring, with no paper), [`simple_vit_with_fft.py`](vit_pytorch/simple_vit_with_fft.py), and [`rvt.py`](vit_pytorch/rvt.py) (RoPE-based, no dedicated README section).

**BibTeX:** the README's [Citations](README.md#citations) section has BibTeX for every repo paper above that isn't marked (related).
