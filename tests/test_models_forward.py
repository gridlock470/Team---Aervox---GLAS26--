"""Forward/backward tests for the model blocks, backbone and multi-task net."""

from __future__ import annotations

import pytest
import torch

from nowcast import config, schema
from nowcast.models.backbone import SpatiotemporalBackbone
from nowcast.models.blocks import Conv3dBlock, ConvLSTMBlock, ConvLSTMCell, num_groups
from nowcast.models.multitask import MultiTaskNowcastNet
from nowcast.models.transformer import SpatiotemporalTransformerBackbone
from nowcast.testing import synthetic
from nowcast.training.losses import MultiTaskLoss

_TINY = {"hidden": 8, "depth": 1}


def _batch(batch_size: int = 2, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = synthetic.make_batch(batch_size, seed=seed)
    return torch.from_numpy(x), torch.from_numpy(y)


def test_num_groups_divides_channels():
    assert num_groups(16) == 8
    assert num_groups(6) == 6
    assert num_groups(1) == 1


def test_convlstm_cell_preserves_shape():
    cell = ConvLSTMCell(4, 6)
    x = torch.randn(2, 4, 8, 9)
    state = cell.init_state(2, (8, 9), x.device, x.dtype)
    h, c = cell(x, state)
    assert h.shape == (2, 6, 8, 9)
    assert c.shape == (2, 6, 8, 9)


def test_convlstm_block_last_hidden_and_full_sequence():
    x = torch.randn(2, 5, 4, 8, 9)
    assert ConvLSTMBlock(4, 6)(x).shape == (2, 6, 8, 9)
    assert ConvLSTMBlock(4, 6, return_sequence=True)(x).shape == (2, 5, 6, 8, 9)


def test_conv3d_block_preserves_dims():
    out = Conv3dBlock(3, 8)(torch.randn(2, 3, 4, 8, 9))
    assert out.shape == (2, 8, 4, 8, 9)


def test_backbone_maps_to_shared_features():
    backbone = SpatiotemporalBackbone()
    x = torch.randn(2, schema.N_CHANNELS, config.INPUT_SEQ_LEN, *config.GRID_SHAPE)
    out = backbone(x)
    assert out.shape == (2, backbone.out_features, *config.GRID_SHAPE)
    assert torch.isfinite(out).all()


def test_transformer_backbone_forward_and_cross_attention_stub():
    backbone = SpatiotemporalTransformerBackbone(embed_dim=8, n_heads=2, depth=1)
    out = backbone(torch.randn(1, schema.N_CHANNELS, 4, 8, 10))
    assert out.shape == (1, 8, 8, 10)
    with pytest.raises(NotImplementedError):
        backbone.cross_attention(torch.randn(1, 3, 8), torch.randn(1, 3, 8))


def test_multitask_forward_returns_per_hazard_lead_maps():
    net = MultiTaskNowcastNet(backbone_kwargs=_TINY)
    x, _ = _batch()
    out = net(x)
    assert set(out) == set(config.HAZARDS)
    for logits in out.values():
        assert logits.shape == (2, schema.N_LEADS, *config.GRID_SHAPE)


def test_predict_proba_is_stacked_and_bounded():
    net = MultiTaskNowcastNet(backbone_kwargs=_TINY)
    x, _ = _batch()
    probs = net.predict_proba(x).detach()
    assert probs.shape == (2, schema.N_HAZARDS, schema.N_LEADS, *config.GRID_SHAPE)
    assert float(probs.min()) >= 0.0
    assert float(probs.max()) <= 1.0


def test_validate_sample_accepts_model_output():
    net = MultiTaskNowcastNet(backbone_kwargs=_TINY)
    x_np, _ = synthetic.make_batch(2, seed=1)
    probs = net.predict_proba(torch.from_numpy(x_np))
    schema.validate_sample(x_np[0], probs[0].detach().numpy())


def test_flash_flood_head_consumes_terrain():
    net = MultiTaskNowcastNet(backbone_kwargs=_TINY).eval()
    x, _ = _batch()
    terrain = torch.randn(2, len(schema.FLASH_FLOOD_EXTRA_CHANNELS), *config.GRID_SHAPE)
    with torch.no_grad():
        with_terrain = net(x, terrain)["flash_flood"]
        zero_terrain = net(x, torch.zeros_like(terrain))["flash_flood"]
    assert not torch.allclose(with_terrain, zero_terrain)


def test_multitask_loss_backward_grads_finite():
    net = MultiTaskNowcastNet(backbone_kwargs=_TINY)
    x, y = _batch()
    loss = MultiTaskLoss()(net(x), y)
    assert loss.ndim == 0
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
