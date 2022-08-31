from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from typing_extensions import Self

from conduit.data.structures import NamedSample, TernarySample
from ranzen import implements
import torch
from torch import Tensor

from src.data.data_module import DataModule
from src.logging import log_images
from src.models.autoencoder import SplitLatentAe
from src.models.discriminator import Discriminator, NeuralDiscriminator

from .base import AdvSemiSupervisedAlg, Components
from .evaluator import Evaluator

__all__ = ["SupportMatching"]


@dataclass
class SupportMatching(AdvSemiSupervisedAlg):
    prior_loss_w: float = 0

    @implements(AdvSemiSupervisedAlg)
    def _discriminator_loss(
        self,
        comp: Components,
        *,
        iterator_tr: Iterator[TernarySample[Tensor]],
        iterator_dep: Iterator[NamedSample[Tensor]],
    ) -> tuple[Tensor, dict[str, float]]:
        """Train the discriminator while keeping the encoder fixed."""
        if isinstance(comp.disc, NeuralDiscriminator):
            comp.train_disc()
            x_tr = self._sample_tr(iterator_tr).x
            x_dep = self._sample_dep(iterator_dep)

            with torch.cuda.amp.autocast(enabled=self.misc_cfg.use_amp):  # type: ignore
                with torch.no_grad():
                    encoding_tr = comp.ae.encode(x_tr)
                    encoding_dep = comp.ae.encode(x_dep)

                logging_dict = {}
                disc_loss = comp.disc.discriminator_loss(
                    fake=encoding_tr.zy,
                    real=encoding_dep.zy,
                )

            return disc_loss, logging_dict
        return torch.zeros((), device=self.device), {}

    @implements(AdvSemiSupervisedAlg)
    def _encoder_loss(
        self, comp: Components, *, x_dep: Tensor, batch_tr: TernarySample[Tensor], warmup: bool
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the losses for the encoder and update its parameters."""
        # Compute losses for the encoder.
        comp.train_ae()
        logging_dict = {}

        with torch.cuda.amp.autocast(enabled=self.misc_cfg.use_amp):  # type: ignore
            # ============================= recon loss for training set ===========================
            encoding_t, enc_loss_tr, logging_dict_tr = comp.ae.training_step(
                batch_tr.x,
                prior_loss_w=self.prior_loss_w,
                s=batch_tr.s if self.s_as_zs else None,  # using s for the reconstruction
            )

            # ============================ recon loss for deployment set ===========================
            encoding_c, enc_loss_dep, logging_dict_dep = comp.ae.training_step(
                x_dep, prior_loss_w=self.prior_loss_w
            )
            logging_dict.update({k: v + logging_dict_dep[k] for k, v in logging_dict_tr.items()})
            enc_loss_tr = 0.5 * (enc_loss_tr + enc_loss_dep)  # take average of the two recon losses
            enc_loss_tr *= self.enc_loss_w
            logging_dict["Loss Encoder"] = enc_loss_tr.detach().cpu().item()
            total_loss = enc_loss_tr
            # ================================= adversarial losses ================================
            if not warmup:
                disc_input_tr = encoding_t.zy
                if isinstance(comp.disc, NeuralDiscriminator):
                    disc_input_dep = encoding_c.zy if self.two_disc_loss else None
                    disc_loss = comp.disc.encoder_loss(fake=disc_input_tr, real=disc_input_dep)
                else:
                    disc_input_dep = encoding_c.zy
                    if not self.twoway_disc_loss:
                        disc_input_dep = disc_input_dep.detach()
                    disc_loss = comp.disc.encoder_loss(fake=disc_input_tr, real=disc_input_dep)

                disc_loss *= self.disc_loss_w
                total_loss += disc_loss
                logging_dict["Loss Discriminator"] = disc_loss.detach().cpu().item()

            if comp.pred_y is not None:
                # predictor is on encodings; predict y from the part that is invariant to s
                pred_y_loss, pred_y_acc = comp.pred_y.training_step(
                    encoding_t.zy,
                    target=batch_tr.y,
                    group_idx=batch_tr.s,
                )
                pred_y_loss *= self.pred_y_loss_w
                logging_dict["Loss Predictor y"] = pred_y_loss.detach().cpu().item()
                logging_dict["Accuracy Predictor y"] = pred_y_acc
                total_loss += pred_y_loss
            if comp.pred_s is not None:
                pred_s_loss, pred_s_acc = comp.pred_s.training_step(
                    encoding_t.zs,
                    target=batch_tr.s,
                    group_idx=batch_tr.s,
                )
                pred_s_loss *= self.pred_s_loss_w
                logging_dict["Loss Predictor s"] = pred_s_loss.detach().cpu().item()
                logging_dict["Accuracy Predictor s"] = pred_s_acc
                total_loss += pred_s_loss

        logging_dict["Loss Total"] = total_loss.detach().cpu().item()

        return total_loss, logging_dict

    @implements(AdvSemiSupervisedAlg)
    def fit(
        self, dm: DataModule, *, ae: SplitLatentAe, disc: Discriminator, evaluator: Evaluator
    ) -> Self:
        if self.s_as_zs and self.zs_dim != dm.card_s:
            raise ValueError(f"zs_dim has to be equal to s_dim ({dm.card_s}) if `s_as_zs` is True.")

        return super().fit(dm=dm, ae=ae, disc=disc, evaluator=evaluator)

    @torch.no_grad()
    @implements(AdvSemiSupervisedAlg)
    def log_recons(
        self, x: Tensor, *, dm: DataModule, ae: SplitLatentAe, itr: int, prefix: str | None = None
    ) -> None:
        """Log the reconstructed and original images."""
        num_blocks = min(4, len(x))
        rows_per_block = min(8, len(x) // num_blocks)
        num_samples = num_blocks * rows_per_block

        sample = x[:num_samples]
        encoding = ae.encode(sample)
        recon = ae.all_recons(encoding, mode="hard")
        recons = [recon.all, recon.zero_s, recon.just_s]

        caption = "original | all | zero_s | just_s"
        if self.s_as_zs:
            recons.append(recon.rand_s)
            caption += " | rand_s"

        to_log = [sample]
        for recon_ in recons:
            to_log.append(recon_)

        ncols = len(to_log)
        interleaved = torch.stack(to_log, dim=1).view(ncols * num_samples, *sample.shape[1:])

        log_images(
            images=interleaved,
            dm=dm,
            name="reconstructions",
            step=itr,
            nsamples=[ncols * rows_per_block] * num_blocks,
            ncols=ncols,
            prefix=prefix,
            caption=caption,
        )
