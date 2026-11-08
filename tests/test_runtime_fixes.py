import pytest
import torch

# small shared config

KW = dict(image_size = 32, patch_size = 8, num_classes = 10, dim = 64, depth = 1, heads = 4, mlp_dim = 128)

@pytest.fixture
def img():
    return torch.randn(2, 3, 32, 32)

# MAE / SimMIM - must accept encoders with (n, d) and (1, n, d) positional embeddings, with or without cls token

def _encoders():
    from vit_pytorch import ViT, SimpleViT
    from vit_pytorch.deepvit import DeepViT

    return dict(
        vit_cls = lambda: ViT(**KW),
        vit_mean = lambda: ViT(**KW, pool = 'mean'),
        simple_vit = lambda: SimpleViT(**KW),
        deepvit_cls = lambda: DeepViT(**KW),
        deepvit_mean = lambda: DeepViT(**KW, pool = 'mean'),
    )

@pytest.mark.parametrize('encoder_name', list(_encoders().keys()))
def test_mae(encoder_name, img):
    from vit_pytorch import MAE
    mae = MAE(encoder = _encoders()[encoder_name](), decoder_dim = 32)
    loss = mae(img)
    loss.backward()
    assert torch.isfinite(loss)

@pytest.mark.parametrize('encoder_name', list(_encoders().keys()))
def test_simmim(encoder_name, img):
    from vit_pytorch.simmim import SimMIM
    simmim = SimMIM(encoder = _encoders()[encoder_name]())
    loss = simmim(img)
    loss.backward()
    assert torch.isfinite(loss)

# distillation - students with (n, d) and (1, n, d) cls token

def _students():
    from vit_pytorch.vit import Transformer
    from vit_pytorch.distill import DistillableViT, DistillableT2TViT, DistillableEfficientViT

    return dict(
        vit = lambda: DistillableViT(**KW),
        t2t = lambda: DistillableT2TViT(image_size = 32, num_classes = 10, dim = 64, depth = 1, heads = 4, mlp_dim = 128),
        efficient = lambda: DistillableEfficientViT(image_size = 32, patch_size = 8, num_classes = 10, dim = 64, transformer = Transformer(64, 1, 4, 16, 128)),
    )

@pytest.mark.parametrize('student_name', list(_students().keys()))
def test_distill(student_name, img):
    from vit_pytorch import ViT
    from vit_pytorch.distill import DistillWrapper

    student = _students()[student_name]()
    wrapper = DistillWrapper(teacher = ViT(**KW), student = student)

    loss = wrapper(img, torch.tensor([1, 2]))
    loss.backward()

    assert torch.isfinite(loss)
    assert student.to_vit()(img).shape == (2, 10)

# qk norm simple vit must project to num_classes

def test_simple_vit_with_qk_norm_outputs_logits(img):
    from vit_pytorch.simple_vit_with_qk_norm import SimpleViT
    assert SimpleViT(**KW)(img).shape == (2, 10)

# local vit feedforward must normalize over channels

def test_local_vit(img):
    from vit_pytorch.local_vit import LocalViT
    assert LocalViT(**KW)(img).shape == (2, 10)

# vivit with moss - with the default causal temporal transformer, logits must depend on the video

@pytest.mark.parametrize('moss_causal', (True, False))
def test_vivit_with_moss_depends_on_input(moss_causal):
    from vit_pytorch.vivit_with_moss import ViViT

    model = ViViT(
        image_size = 32,
        image_patch_size = 8,
        frames = 4,
        frame_patch_size = 1,
        num_classes = 5,
        dim = 64,
        spatial_depth = 1,
        temporal_depth = 1,
        heads = 4,
        mlp_dim = 128,
        moss_causal = moss_causal
    ).eval()

    a = model(torch.randn(1, 3, 4, 32, 32))
    b = model(torch.randn(1, 3, 4, 32, 32))
    assert not torch.allclose(a, b)

# nested tensor navit - default token_dropout_prob in training, and gradients reaching the embeddings

@pytest.mark.parametrize('token_dropout_prob', (None, 0.1))
def test_na_vit_nested_tensor_trains(token_dropout_prob):
    from vit_pytorch.na_vit_nested_tensor import NaViT

    model = NaViT(**KW, token_dropout_prob = token_dropout_prob).train()
    out = model([torch.randn(3, 32, 32), torch.randn(3, 16, 24)])
    out.sum().backward()

    assert out.shape == (2, 10)

    for name in ('pos_embed_height', 'pos_embed_width', 'attn_pool_queries'):
        assert getattr(model, name).grad is not None, f'{name} received no gradient'

    assert all(p.grad is not None for p in model.to_patch_embedding.parameters())

def test_na_vit_nested_tensor_3d_trains():
    from vit_pytorch.na_vit_nested_tensor_3d import NaViT

    model = NaViT(image_size = 32, patch_size = 8, max_frames = 4, frame_patch_size = 2, num_classes = 10, dim = 64, depth = 1, heads = 4, mlp_dim = 128).train()
    out = model([torch.randn(3, 4, 32, 32), torch.randn(3, 2, 32, 16)])
    out.sum().backward()

    assert out.shape == (2, 10)
    assert all(p.grad is not None for p in model.to_patch_embedding.parameters())

# flash attention vit - the non flash path must run and match the flash path

def test_simple_flash_attn_vit_paths_match(img):
    from vit_pytorch.simple_flash_attn_vit import SimpleViT

    flash = SimpleViT(**KW, use_flash = True).eval()
    manual = SimpleViT(**KW, use_flash = False).eval()
    manual.load_state_dict(flash.state_dict())

    assert torch.allclose(flash(img), manual(img), atol = 1e-5)

def test_simple_flash_attn_vit_3d_paths_match():
    from vit_pytorch.simple_flash_attn_vit_3d import SimpleViT

    kw = dict(image_size = 32, image_patch_size = 8, frames = 4, frame_patch_size = 2, num_classes = 10, dim = 96, depth = 1, heads = 4, mlp_dim = 128)

    flash = SimpleViT(**kw, use_flash_attn = True).eval()
    manual = SimpleViT(**kw, use_flash_attn = False).eval()
    manual.load_state_dict(flash.state_dict())

    video = torch.randn(2, 3, 4, 32, 32)
    assert torch.allclose(flash(video), manual(video), atol = 1e-5)
