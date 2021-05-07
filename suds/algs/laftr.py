from __future__ import annotations

from typing import Iterator, Sequence

import torch
import torch.nn as nn
from kit import implements
from torch import Tensor

from shared.configs.arguments import CmnistConfig
from shared.configs.enums import ReconstructionLoss
from shared.data.utils import Batch
from shared.models.configs.classifiers import FcNet
from shared.utils.utils import ModelFn, prod
from suds.algs.ss_base import AdvSemiSupervisedAlg
from suds.models.base import SplitEncoding
from suds.models.classifier import Classifier
from suds.models.configs.classifiers import Residual64x64Net, Strided28x28Net
from suds.optimisation.utils import log_images

__all__ = ["LAFTR"]


class LAFTR(AdvSemiSupervisedAlg):
    adversary: Classifier

    @implements(AdvSemiSupervisedAlg)
    def _build_adversary(self, input_shape: tuple[int, ...], s_dim: int) -> Classifier:
        # TODO: Move into a 'build' method
        adv_input_shape: tuple[int, ...] = (
            input_shape if self.adv_cfg.train_on_recon else (self.enc_cfg.out_dim,)
        )
        adv_fn: ModelFn
        if len(input_shape) > 2 and self.adv_cfg.train_on_recon:
            if isinstance(self.data_cfg, CmnistConfig):
                adv_fn = Strided28x28Net(batch_norm=False)
            else:
                adv_fn = Residual64x64Net(batch_norm=False)

        else:
            adv_fn = FcNet(hidden_dims=self.adv_cfg.adv_hidden_dims, activation=nn.GELU())
            # FcNet first flattens the input
            adv_input_shape = (
                (prod(adv_input_shape),)
                if isinstance(adv_input_shape, Sequence)
                else adv_input_shape
            )

        return Classifier(
            model=adv_fn(adv_input_shape, s_dim),  # type: ignore
            optimizer_kwargs=self.optimizer_kwargs,
            num_classes=s_dim if s_dim > 1 else 2,
        )

    @implements(AdvSemiSupervisedAlg)
    def _step_adversary(
        self,
        train_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        context_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
    ) -> tuple[Tensor, dict[str, float]]:
        """Train the discriminator while keeping the encoder constant.
        Args:
            x_c: x from the context set
            x_t: x from the training set
        """
        self._train("adversary")
        tr_batch = self._sample_train(train_data_itr)

        logging_dict = {}
        # Context-manager enables mixed-precision training
        with torch.cuda.amp.autocast(enabled=self.misc_cfg.use_amp):  # type: ignore
            with torch.no_grad():
                encoding_tr = self.encoder.encode(tr_batch.x, stochastic=True)
                adv_input_tr = self._get_adv_input(encoding_tr)
            adv_loss, _ = self.adversary.routine(adv_input_tr, tr_batch.s)

        self._update_adversary(adv_loss)
        return adv_loss, logging_dict

    @implements(AdvSemiSupervisedAlg)
    def _step_encoder(
        self, x_ctx: Tensor, batch_tr: Batch, warmup: bool
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute all losses."""
        # Compute losses for the encoder.
        self._train("encoder")
        logging_dict = {}
        with torch.cuda.amp.autocast(enabled=self.misc_cfg.use_amp):
            # ============================= recon loss for training set ===========================
            encoding_tr, enc_loss_tr, logging_dict_tr = self.encoder.routine(
                batch_tr.x, self.recon_loss_fn, self.enc_cfg.prior_loss_w
            )

            # ============================= recon loss for context set ============================
            _, enc_loss_ctx, logging_dict_ctx = self.encoder.routine(
                x_ctx, recon_loss_fn=self.recon_loss_fn, prior_loss_w=self.enc_cfg.prior_loss_w
            )
            logging_dict.update({k: v + logging_dict_ctx[k] for k, v in logging_dict_tr.items()})
            enc_loss_tr = 0.5 * (enc_loss_tr + enc_loss_ctx)  # take average of the two recon losses
            enc_loss_tr *= self.adv_cfg.enc_loss_w
            logging_dict["Loss Generator"] = enc_loss_tr
            total_loss = enc_loss_tr
            # ================================= adversarial losses ================================
            if not warmup:
                disc_input_t = self._get_adv_input(encoding_tr)
                disc_loss = self.adversary.routine(data=disc_input_t, targets=batch_tr.y)[0]
                disc_loss *= self.adv_cfg.adv_weight
                # Negate the discriminator's loss to obtain the adversarial loss w.r.t the encoder
                total_loss -= disc_loss
                logging_dict["Loss Discriminator"] = disc_loss

            if self.predictor_y is not None:
                # predictor is on encodings; predict y from the part that is invariant to s
                pred_y_loss, pred_y_acc = self.predictor_y.routine(encoding_tr.zy, batch_tr.y)
                pred_y_loss *= self.adv_cfg.pred_y_loss_w
                logging_dict["Loss Predictor y"] = pred_y_loss.item()
                logging_dict["Accuracy Predictor y"] = pred_y_acc
                total_loss += pred_y_loss
            if self.predictor_s is not None:
                pred_s_loss, pred_s_acc = self.predictor_s.routine(encoding_tr.zs, batch_tr.s)
                pred_s_loss *= self.adv_cfg.pred_s_loss_w
                logging_dict["Loss Predictor s"] = pred_s_loss.item()
                logging_dict["Accuracy Predictor s"] = pred_s_acc
                total_loss += pred_s_loss

        logging_dict["Loss Total"] = total_loss

        self._update_encoder(total_loss)

        return total_loss, logging_dict

    def _get_adv_input(self, encoding: SplitEncoding, detach: bool = False) -> Tensor:
        """Construct the input that the discriminator expects; either zy or reconstructed zy."""
        zs_m, _ = self.encoder.mask(encoding, detach=detach)
        return self.encoder.unsplit_encoding(zs_m)

    @torch.no_grad()
    @implements(AdvSemiSupervisedAlg)
    def _log_recons(self, x: Tensor, itr: int, prefix: str | None = None) -> None:
        """Log reconstructed images."""

        rows_per_block = 8
        num_blocks = 4
        num_samples = num_blocks * rows_per_block

        sample = x[:num_samples]
        encoding = self.encoder.encode(sample, stochastic=False)
        recon = self.encoder.all_recons(encoding, mode="hard")
        recons = [recon.all, recon.zero_s, recon.just_s]
        caption = "original | all | zero_s | just_s"
        to_log: list[Tensor] = [sample]
        for recon_ in recons:
            if self.enc_cfg.recon_loss is ReconstructionLoss.ce:
                to_log.append(recon_.argmax(dim=1).float() / 255)
            else:
                to_log.append(recon_)
        ncols = len(to_log)

        interleaved = torch.stack(to_log, dim=1).view(ncols * num_samples, *sample.shape[1:])

        log_images(
            self.cfg,
            interleaved,
            name="reconstructions",
            step=itr,
            nsamples=[ncols * rows_per_block] * num_blocks,
            ncols=ncols,
            prefix=prefix,
            caption=caption,
        )
