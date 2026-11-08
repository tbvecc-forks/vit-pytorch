# Reading list

Papers for understanding this repository: every paper `vit-pytorch` implements or cites, plus closely related work. They're grouped by the stage of the study plan in [ANALYSIS.md §12](ANALYSIS.md#12-study-plan) where they're most useful.

- **Sources:** the README's [Citations](README.md#citations) section (68 BibTeX entries) and paper links in the code (`grep -E "arxiv|openreview" vit_pytorch/*.py`).
- **(related):** not in the repo, but directly relevant to something covered in [ANALYSIS.md](ANALYSIS.md) or the study-plan exercises.
- **Links:** arXiv IDs come from the README or the code. Where the README gives only a Semantic Scholar or OpenReview link, that link is used instead of an ID.
- **★:** core reading for the plan. The rest is reference material to read when you reach that file.

## Foundations (Stages 1–2)

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

## Single-idea `SimpleViT` variants (Stage 2)

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

## Classic ViT variants (Stage 3)

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

## Hierarchical and convolutional hybrids (Stage 4)

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

## Self-supervised and training wrappers (Stage 5)

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

## Beyond 2-D images: video, N-D, variable resolution, robotics (Stage 6)

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
