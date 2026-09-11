"""LightningModule wiring the model, loss and metrics into a training loop."""

from __future__ import annotations

import lightning as L
import torch

from nowcast import config
from nowcast.models.multitask import MultiTaskNowcastNet
from nowcast.training.losses import MultiTaskLoss
from nowcast.training.metrics import NowcastMetrics, event_mask_from_targets


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
        # Defaults come from config rather than being hard-coded here. A local
        # focal_weight=0.0 would silently shadow the library default and leave
        # focal loss switched off however it was configured elsewhere.
        focal_weight: float = config.FOCAL_WEIGHT,
        focal_alpha: float = config.FOCAL_ALPHA,
        focal_gamma: float = config.FOCAL_GAMMA,
        pos_weight: float | None = config.BCE_POS_WEIGHT,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = MultiTaskNowcastNet(backbone_kwargs=backbone_kwargs)
        self.criterion = MultiTaskLoss(
            hazard_weights=hazard_weights,
            bce=bce_weight,
            dice=dice_weight,
            focal=focal_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            pos_weight=pos_weight,
        )
        # Two metric instances. The full threshold sweep is worth its cost once
        # per validation batch but not on every training step, so training uses
        # a coarse grid. CSI at a fixed 0.5 is near-meaningless at this base
        # rate, which is why the sweep exists at all.
        self.metrics = NowcastMetrics(sweep_thresholds=(0.05, 0.2, 0.5))
        self.val_metrics = NowcastMetrics()

    def forward(
        self, x: torch.Tensor, terrain: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Delegate to the wrapped model."""
        return self.model(x, terrain)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        x, y, terrain = batch
        out = self.model(x, terrain)
        loss = self.criterion(out, y)
        is_val = stage != "train"
        with torch.no_grad():
            probs = torch.sigmoid(self.model.stack_logits(out))
            metrics = self.val_metrics if is_val else self.metrics
            # Storm-subset scores only on validation: aggregate CSI is flattered
            # by easy quiet windows, so skill on windows that actually contain an
            # event is the number worth tracking.
            mask = event_mask_from_targets(y) if is_val else None
            scores = metrics.compute(probs, y, event_mask=mask)
        batch_size = x.shape[0]
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_csi", scores["csi"], prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_pr_auc", scores["pr_auc"], batch_size=batch_size)
        # csi_best is meaningless without the threshold that achieved it, so the
        # two are always logged together.
        self.log(f"{stage}_csi_best", scores["csi_best"], prog_bar=is_val,
                 batch_size=batch_size)
        self.log(f"{stage}_csi_best_threshold", scores["csi_best_threshold"],
                 batch_size=batch_size)
        self.log(f"{stage}_frequency_bias", scores["frequency_bias"], batch_size=batch_size)
        self.log(f"{stage}_brier", scores["brier"], batch_size=batch_size)
        if is_val and "storm/csi" in scores:
            self.log("val_storm_csi", scores["storm/csi"], prog_bar=True,
                     batch_size=batch_size)
            self.log("val_storm_n", scores["storm/n_samples"], batch_size=batch_size)
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
