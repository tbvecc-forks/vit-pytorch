import torch
from torch import nn

from vit_pytorch.simple_vit_with_specialized_cls import Specialized, SimpleViT

def test_specialized_tensor_input_returns_tensor():
    specialized = Specialized([nn.LayerNorm(8), nn.LayerNorm(8)])
    x = torch.randn(2, 17, 8)

    out = specialized(x, token_lens = (1, 16))

    assert torch.is_tensor(out)
    assert out.shape == (2, 17, 8)

def test_specialized_list_input_returns_tuple():
    specialized = Specialized([nn.LayerNorm(8), nn.LayerNorm(8)])
    cls_token, patches = torch.randn(2, 1, 8), torch.randn(2, 16, 8)

    out = specialized([cls_token, patches])

    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == (2, 1, 8) and out[1].shape == (2, 16, 8)

def test_specialized_cls_vit_forward():
    model = SimpleViT(image_size = 32, patch_size = 8, num_classes = 10, dim = 64, depth = 3, heads = 4, mlp_dim = 128)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
