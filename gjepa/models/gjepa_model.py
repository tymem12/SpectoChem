from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch import Tensor
from torch.nn import SmoothL1Loss
from torch.optim import AdamW
from torch_geometric.data import Batch, Data

from gjepa.config import ExperimentConfig, JEPAConfig
from gjepa.models.downstream import LinearProbingClassifier
from gjepa.models.encoders import GNNEncoder
from gjepa.models.gjepa_extractors import Extractor
from gjepa.models.gjepa_predictors import GJEPAPredictor
from gjepa.utils.lr_scheduler import LinearWarmupCosineAnnealingLR  # type: ignore

SVD_VAL_EPS = 1e-5


class GJEPANodeModel(LightningModule):
    """Node-level training with JEPA framework.
    Code partially adopted from: https://github.com/facebookresearch/ijepa
    """

    def __init__(self, config: ExperimentConfig | dict[str, Any]):
        super().__init__()
        if isinstance(config, dict):
            config = ExperimentConfig.from_raw_config(config)

        self.save_hyperparameters({"config": config.model_dump()})
        self.config = config
        assert isinstance(self.config.model, JEPAConfig)

        (
            self.context_encoder,
            self.target_encoder,
            self.target_predictor,
        ) = self.init_model()

        self.context_extractor = Extractor.from_config(
            extractor_type=self.config.model.context_extractor_type,
            encoder=self.context_encoder,
        )
        self.target_extractor = Extractor.from_config(
            extractor_type=self.config.model.target_extractor_type,
            encoder=self.target_encoder,
        )

        self.loss = SmoothL1Loss()
        self.downstream_model = LinearProbingClassifier(
            self.config.dataset.task_type, self.config.dataset.out_channels
        )

        ipe, ipe_scale = self.config.model.ipe, self.config.model.ipe_scale
        ema = self.config.model.ema

        num_epochs = self.config.training.max_epochs
        self.momentum_scheduler = (
            ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
            for i in range(int(ipe * num_epochs * ipe_scale) + 1)
        )

        if self.config.model.context_target_overlap_strategy == "mask":
            self.ctx_mask_vector = torch.nn.Parameter(
                data=torch.randn(1, self.config.dataset.in_channels),
                requires_grad=True,
            )
        else:
            self.register_buffer("ctx_mask_vector", None)

    def init_model(self) -> tuple[GNNEncoder, GNNEncoder, GJEPAPredictor]:
        # build context model (student)
        context_encoder = GNNEncoder(**self.config.model.backbone)

        # build target model (teacher) - self-distillation updates applies
        target_encoder = deepcopy(context_encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False

        # build predictor model
        assert isinstance(self.config.model, JEPAConfig)
        predictor = GJEPAPredictor.from_config(
            predictor_type=self.config.model.target_predictor["name"],
            predictor_kwargs=self.config.model.target_predictor["kwargs"],
        )

        return context_encoder, target_encoder, predictor

    def training_step(self, batch: dict[str, Batch], batch_idx: int) -> Tensor:
        """Training step of I-JEPA methods, based on: https://github.com/facebookresearch/ijepa."""
        # forward context and target
        ctx_graphs = batch["context"]

        assert isinstance(self.config.model, JEPAConfig)
        if self.config.model.context_target_overlap_strategy == "mask":
            ctx_graphs.x[ctx_graphs.target_node_mask] = self.ctx_mask_vector.repeat(
                ctx_graphs.target_node_mask.sum().item(), 1
            )

        context_out = self.context_extractor(ctx_graphs)

        with torch.no_grad():
            self.target_encoder.eval()
            max_num_nodes = max(
                torch.bincount(t.batch.cpu()).amax().item() for t in batch["targets"]
            )
            target_outs = [
                self.target_extractor(target_g, max_num_nodes=max_num_nodes)
                for target_g in batch["targets"]
            ]

        # predict target embeddings
        pred_target, z_target = self.target_predictor(
            context_out=context_out,
            target_outs=target_outs,
        )

        # compute loss
        loss = self.loss(input=pred_target, target=z_target)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def on_validation_epoch_start(self) -> None:
        self._update_downstream_model_with_train_representations()

        num_svd_vals_above_threshold = (
            torch.linalg.svdvals(self.downstream_model.z_train) > SVD_VAL_EPS
        ).sum(dim=-1)
        self.log("num_svd_above_threshold(z_train)", num_svd_vals_above_threshold.float())

        if self.config.training.save_representations and self.current_epoch % 10 == 0:
            assert self.trainer.log_dir is not None
            save_path = Path(self.trainer.log_dir) / f"representations_{self.current_epoch:03d}.pt"
            reprs = {
                "representations": self.downstream_model.z_train,
                "labels": self.downstream_model.y_train,
                "epoch": self.current_epoch,
            }
            torch.save(reprs, save_path)

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        z, y = self.get_z_y_from_batch(batch)
        self.downstream_model.update_test(z, y)

    def on_validation_epoch_end(self) -> None:
        self.downstream_model.fit()
        metrics = self.downstream_model.score(metric_prefix="val_")
        self.log_dict(metrics)

    def on_train_epoch_end(self) -> None:
        """EMA self-distillation weights update for target (teacher) network."""
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            self.log("ema", m)
            for param_context, param_target in zip(
                self.context_encoder.parameters(), self.target_encoder.parameters()
            ):
                param_target.data.mul_(m).add_((1.0 - m) * param_context.detach().data)

    def on_test_start(self) -> None:
        self._update_downstream_model_with_train_representations()

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        z, y = self.get_z_y_from_batch(batch)
        self.downstream_model.update_test(z, y)

    def on_test_epoch_end(self) -> None:
        self.downstream_model.fit()
        metrics = self.downstream_model.score(metric_prefix="test_")
        self.log_dict(metrics)

    def predict_step(
        self, batch: Batch, batch_idx: int, dataloader_idx: int = 0
    ) -> tuple[Tensor, Tensor]:
        return self.get_z_y_from_batch(batch)

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        optim = AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        if self.config.training.scheduler_config is None:
            return {"optimizer": optim}

        scheduler = LinearWarmupCosineAnnealingLR(optim, **self.config.training.scheduler_config)
        return {"optimizer": optim, "lr_scheduler": scheduler}

    def _update_downstream_model_with_train_representations(self) -> None:
        self.downstream_model.reset()
        for batch in self.trainer.datamodule.train_inference_dataloader():  # type: ignore
            # Iterating data_loader manually requires manual device change:
            z, y = self.get_z_y_from_batch(batch.to(self.device))
            self.downstream_model.update_train(z, y)

    def get_z_y_from_batch(self, batch: Data) -> tuple[Tensor, Tensor]:
        z = self.context_encoder(x=batch.x, edge_index=batch.edge_index)
        y = batch.y
        return z[: batch.batch_size], y[: batch.batch_size]
