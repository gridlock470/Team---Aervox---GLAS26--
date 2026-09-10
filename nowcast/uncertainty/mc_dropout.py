"""Monte-Carlo dropout: keep dropout active at inference to sample predictions."""

from __future__ import annotations

import torch
from torch import nn

_DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)


def enable_mc_dropout(model: nn.Module) -> int:
    """Put ``model`` in eval mode but re-enable every dropout layer.

    Returns the number of dropout layers switched back to training mode.
    """
    model.eval()
    count = 0
    for module in model.modules():
        if isinstance(module, _DROPOUT_TYPES):
            module.train()
            count += 1
    return count


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    terrain: torch.Tensor | None = None,
    n: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(mean, std)`` over ``n`` stochastic ``predict_proba`` passes.

    ``model`` must expose ``predict_proba(x, terrain)`` (e.g.
    :class:`nowcast.models.multitask.MultiTaskNowcastNet`). Both outputs have the
    shape of a single ``predict_proba`` call.
    """
    enable_mc_dropout(model)
    samples = torch.stack([model.predict_proba(x, terrain) for _ in range(n)])
    return samples.mean(dim=0), samples.std(dim=0)
