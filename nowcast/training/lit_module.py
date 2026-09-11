"""LightningModule wiring the model, loss and metrics into a training loop."""

from __future__ import annotations

import lightning as L
import torch

from nowcast.models.multitask import MultiTaskNowcastNet
from nowcast.training.losses import MultiTaskLoss
from nowcast.training.metrics import NowcastMetrics


class LitNowcast(L.LightningModule):
    """Train :class:`MultiTaskNowcastNet` with :class:`MultiTaskLoss`.

    Batches are ``(x, y, terrain)`` as produced by
    :class:`nowcast.data.datamodule.NowcastDataset`. Logs loss plus CSI and
    PR-AUC each step; optimises with AdamW + cosine annealing.
    """

    def __init__(
        self,
        backbone_kwargs: dict | None = None,
        hazard_weights: dict[str, float] | None = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 20,
        bce_weight: float = 1.0,
        dice_weight: float = 0.5,
        focal_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = MultiTaskNowcastNet(backbone_kwargs=backbone_kwargs)
        self.criterion = MultiTaskLoss(
            hazard_weights=hazard_weights,
            bce=bce_weight,
            dice=dice_weight,
            focal=focal_weight,
        )
        self.metrics = NowcastMetrics()

    def forward(
        self, x: torch.Tensor, terrain: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Delegate to the wrapped model."""
        return self.model(x, terrain)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        x, y, terrain = batch
        out = self.model(x, terrain)
        loss = self.criterion(out, y)
        with torch.no_grad():
            probs = torch.sigmoid(self.model.stack_logits(out))
            scores = self.metrics.compute(probs, y)
        batch_size = x.shape[0]
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_csi", scores["csi"], prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_pr_auc", scores["pr_auc"], batch_size=batch_size)
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Single training step."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Single validation step."""
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> dict:
        """AdamW + :class:`~torch.optim.lr_scheduler.CosineAnnealingLR`."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.hparams.max_epochs)
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
