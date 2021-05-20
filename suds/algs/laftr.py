from __future__ import annotations
from typing import Iterator, Sequence

from kit import implements
import torch
from torch import Tensor
import torch.distributions as td
import torch.nn as nn

from shared.configs.arguments import CmnistConfig
from shared.configs.enums import ReconstructionLoss
from shared.data.utils import Batch
from shared.models.configs.classifiers import FcNet
from shared.utils.utils import ModelFn, prod
from suds.algs.adv import AdvSemiSupervisedAlg
from suds.models.base import SplitEncoding
from suds.models.classifier import Classifier
from suds.models.configs.classifiers import Residual64x64Net, Strided28x28Net
from suds.optimisation.utils import log_images

__all__ = ["LAFTR"]


class LAFTR(AdvSemiSupervisedAlg):
    adversary: Classifier

    @implements(AdvSemiSupervisedAlg)
    def _build_adversary(self, input_shape: tuple[int, ...], s_dim: int) -> Classifier:
        """Construct the adversarial network."""
        adv_input_shape: tuple[int, ...] = (
            input_shape if self.adapt_cfg.train_on_recon else (self.enc_cfg.out_dim,)
        )
        adv_fn: ModelFn
        if len(input_shape) > 2 and self.adapt_cfg.train_on_recon:
            if isinstance(self.data_cfg, CmnistConfig):
                adv_fn = Strided28x28Net(batch_norm=False)
            else:
                adv_fn = Residual64x64Net(batch_norm=False)

        else:
            adv_fn = FcNet(hidden_dims=self.adapt_cfg.adv_hidden_dims, activation=nn.GELU())
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

    @staticmethod
    def mixup(x1, x2, alpha: float = 0.1) -> Tensor:
        """Vicinal Risk Minimization reformulation of mix-up which interpolates only between xs."""
        lambda_ = (
            td.Beta(alpha + 1, alpha).sample((x1.size(0), *((1,) * (x1.ndim - 1)))).to(x1.device)
        )
        return lambda_ * x1 + (1 - lambda_) * x2

    @implements(AdvSemiSupervisedAlg)
    def _step_adversary(
        self,
        train_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        context_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
    ) -> tuple[Tensor, dict[str, float]]:
        """Train the adversary while fixing the encoder."""
        self._train("adversary")
        tr_batch = self._sample_train(train_data_itr)
        x, s = tr_batch.x, tr_batch.s
        if self.adapt_cfg.mixup:
            x_ctx = self._sample_context(context_data_itr)
            x = self.mixup(tr_batch.x, x_ctx)

        logging_dict = {}
        # Context-manager enables mixed-precision training
        with torch.cuda.amp.autocast(enabled=self.misc_cfg.use_amp):  # type: ignore
            with torch.no_grad():
                encoding_tr = self.encoder.encode(x, stochastic=True)
                adv_input_tr = self._get_adv_input(encoding_tr)
            adv_loss, _ = self.adversary.routine(adv_input_tr, s)

        self._update_adversary(adv_loss)
        return adv_loss, logging_dict

    @implements(AdvSemiSupervisedAlg)
    def _step_encoder(
        self, x_ctx: Tensor, batch_tr: Batch, warmup: bool
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute losses for the encoder."""
        self._train("encoder")
        logging_dict = {}
        with torch.cuda.amp.autocast(enabled=self.misc_cfg.use_amp):
            # ================================ reconstruction loss =================================
            if self.adapt_cfg.mixup:
                x = self.mixup(batch_tr.x, x_ctx)
                encoding_tr, enc_loss, logging_dict_enc = self.encoder.routine(
                    x, self.recon_loss_fn, self.enc_cfg.prior_loss_w
                )
            else:
                # Concatenate the xs so only one forward pass of the autoencoder is required
                x = torch.cat([batch_tr.x, x_ctx], dim=0)
                encoding, enc_loss, logging_dict_enc = self.encoder.routine(
                    x, self.recon_loss_fn, self.enc_cfg.prior_loss_w
                )
                n_tr = batch_tr.x.size(0)
                # Only the training data is labellled and can be used for computing the adversarial
                # loss so we need to de-concatenate the embeddings
                encoding_tr = SplitEncoding(zs=encoding.zs[:n_tr], zy=encoding.zy[:n_tr])
            logging_dict.update(logging_dict_enc)
            enc_loss *= self.adapt_cfg.enc_loss_w
            logging_dict["Loss Encoder"] = enc_loss
            total_loss = enc_loss
            # ================================= adversarial losses ================================
            if not warmup:
                adv_input_tr = self._get_adv_input(encoding_tr)
                adv_loss = self.adversary.routine(data=adv_input_tr, targets=batch_tr.y)[0]
                adv_loss *= self.adapt_cfg.adv_loss_w
                # Negate the discriminator's loss to obtain the adversarial loss w.r.t the encoder
                total_loss -= adv_loss
                logging_dict["Loss Adversary"] = adv_loss

            if self.predictor_y is not None:
                # predictor is on encodings; predict y from the part that is invariant to s
                pred_y_loss, pred_y_acc = self.predictor_y.routine(encoding_tr.zy, batch_tr.y)
                pred_y_loss *= self.adapt_cfg.pred_y_loss_w
                logging_dict["Loss Predictor y"] = pred_y_loss.item()
                logging_dict["Accuracy Predictor y"] = pred_y_acc
                total_loss += pred_y_loss
            if self.predictor_s is not None:
                pred_s_loss, pred_s_acc = self.predictor_s.routine(encoding_tr.zs, batch_tr.s)
                pred_s_loss *= self.adapt_cfg.pred_s_loss_w
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
        """Log the reconstructed and original images."""

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
