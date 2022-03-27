from __future__ import annotations
from collections.abc import Iterator

from conduit.data.structures import NamedSample, TernarySample
from ranzen import implements
import torch
from torch import Tensor
import torch.nn as nn

from advrep.models import AutoEncoder, SetDiscriminator, SplitEncoding
from advrep.optimisation import log_images, mmd2
from shared.configs import DiscriminatorMethod
from shared.data.data_module import DataModule
from shared.layers import Aggregator
from shared.models.configs import FcNet

from .adv import AdvSemiSupervisedAlg

__all__ = ["SupportMatching"]


class SupportMatching(AdvSemiSupervisedAlg):
    @implements(AdvSemiSupervisedAlg)
    def _build_discriminator(self, encoder: AutoEncoder, *, dm: DataModule) -> SetDiscriminator:
        """Build the adversarial network."""
        disc_fn = FcNet(hidden_dims=self.alg_cfg.adv_hidden_dims, activation=nn.GELU())

        aggregator = None
        if self.alg_cfg.aggregator_type is not None:
            final_proj = (
                FcNet(self.alg_cfg.aggregator_hidden_dims)
                if self.alg_cfg.aggregator_hidden_dims
                else None
            )
            aggregator_cls: type[Aggregator] = self.alg_cfg.aggregator_type.value
            aggregator = aggregator_cls(
                embed_dim=self.alg_cfg.aggregator_input_dim,
                batch_size=dm.batch_size_tr,
                final_proj=final_proj,
                **self.alg_cfg.aggregator_kwargs,
            )
            backbone = disc_fn(
                input_dim=encoder.encoding_size.zy, target_dim=self.alg_cfg.aggregator_input_dim
            )
            backbone = nn.Sequential(nn.GELU(), backbone)
        else:
            backbone = disc_fn(input_dim=encoder.encoding_size.zy, target_dim=1)

        gn = nn.GroupNorm(num_groups=1, num_channels=encoder.encoding_size.zy)
        backbone = nn.Sequential(gn, backbone)

        return SetDiscriminator(
            backbone=backbone,
            aggregator=aggregator,
            double_adv_loss=self.alg_cfg.double_adv_loss,
            lr=self.adv_lr,
            criterion=self.alg_cfg.adv_loss,
        )

    @implements(AdvSemiSupervisedAlg)
    def _build_encoder(
        self,
        dm: DataModule,
    ) -> AutoEncoder:
        if self.alg_cfg.s_as_zs and self.alg_cfg.zs_dim != dm.card_s:
            raise ValueError(f"zs_dim has to be equal to s_dim ({dm.card_s}) if `s_as_zs` is True.")
        return super()._build_encoder(dm)

    @implements(AdvSemiSupervisedAlg)
    def _step_discriminator(
        self,
        iterator_tr: Iterator[TernarySample[Tensor]],
        *,
        iterator_dep: Iterator[NamedSample[Tensor]],
    ) -> tuple[Tensor, dict[str, float]]:
        """Train the discriminator while keeping the encoder fixed."""
        self._train("discriminator")
        x_tr = self._sample_tr(iterator_tr).x
        x_dep = self._sample_dep(iterator_dep)

        with torch.cuda.amp.autocast(enabled=self.misc_cfg.use_amp):  # type: ignore
            with torch.no_grad():
                encoding_tr = self.encoder.encode(x_tr)
                encoding_dep = self.encoder.encode(x_dep)

            logging_dict = {}
            disc_input_tr = self._get_disc_input(encoding_tr)
            disc_input_dep = self._get_disc_input(encoding_dep)
            disc_loss = self.discriminator.discriminator_loss(fake=disc_input_tr, real=disc_input_dep)  # type: ignore

        self._update_discriminator(disc_loss)

        return disc_loss, logging_dict

    @implements(AdvSemiSupervisedAlg)
    def _step_encoder(
        self, x_dep: Tensor, *, batch_tr: TernarySample[Tensor], warmup: bool
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the losses for the encoder and update its parameters."""
        # Compute losses for the encoder.
        self._train("encoder")
        logging_dict = {}

        with torch.cuda.amp.autocast(enabled=self.misc_cfg.use_amp):  # type: ignore
            # ============================= recon loss for training set ===========================
            encoding_t, enc_loss_tr, logging_dict_tr = self.encoder.training_step(
                batch_tr.x,
                prior_loss_w=self.alg_cfg.prior_loss_w,
                s=batch_tr.s if self.alg_cfg.s_as_zs else None,  # using s for the reconstruction
            )

            # ============================ recon loss for deployment set ===========================
            encoding_c, enc_loss_dep, logging_dict_dep = self.encoder.training_step(
                x_dep, prior_loss_w=self.alg_cfg.prior_loss_w
            )
            logging_dict.update({k: v + logging_dict_dep[k] for k, v in logging_dict_tr.items()})
            enc_loss_tr = 0.5 * (enc_loss_tr + enc_loss_dep)  # take average of the two recon losses
            enc_loss_tr *= self.alg_cfg.enc_loss_w
            logging_dict["Loss Encoder"] = enc_loss_tr.detach().cpu().item()
            total_loss = enc_loss_tr
            # ================================= adversarial losses ================================
            if not warmup:
                disc_input_tr = self._get_disc_input(encoding_t)
                disc_input_dep = self._get_disc_input(encoding_c)

                if self.alg_cfg.adv_method is DiscriminatorMethod.nn:
                    disc_loss = self.discriminator.encoder_loss(
                        fake=disc_input_tr, real=disc_input_dep
                    )

                else:
                    x = disc_input_tr
                    y = self._get_disc_input(encoding_c).detach()
                    disc_loss = mmd2(
                        x=x,
                        y=y,
                        kernel=self.alg_cfg.mmd_kernel,
                        scales=self.alg_cfg.mmd_scales,
                        wts=self.alg_cfg.mmd_wts,
                        add_dot=self.alg_cfg.mmd_add_dot,
                    )
                disc_loss *= self.alg_cfg.adv_loss_w
                total_loss += disc_loss
                logging_dict["Loss Discriminator"] = disc_loss.detach().cpu().item()

            if self.predictor_y is not None:
                # predictor is on encodings; predict y from the part that is invariant to s
                pred_y_loss, pred_y_acc = self.predictor_y.training_step(
                    encoding_t.zy, target=batch_tr.y
                )
                pred_y_loss *= self.alg_cfg.pred_y_loss_w
                logging_dict["Loss Predictor y"] = pred_y_loss.detach().cpu().item()
                logging_dict["Accuracy Predictor y"] = pred_y_acc
                total_loss += pred_y_loss
            if self.predictor_s is not None:
                pred_s_loss, pred_s_acc = self.predictor_s.training_step(
                    encoding_t.zs, target=batch_tr.s
                )
                pred_s_loss *= self.alg_cfg.pred_s_loss_w
                logging_dict["Loss Predictor s"] = pred_s_loss.detach().cpu().item()
                logging_dict["Accuracy Predictor s"] = pred_s_acc
                total_loss += pred_s_loss

        logging_dict["Loss Total"] = total_loss.detach().cpu().item()

        self._update_encoder(total_loss)

        return total_loss, logging_dict

    @torch.no_grad()
    def _get_disc_input(self, encoding: SplitEncoding) -> Tensor:
        """Construct the input that the discriminator expects; either zy or reconstructed zy."""
        return encoding.zy

    @torch.no_grad()
    @implements(AdvSemiSupervisedAlg)
    def log_recons(self, x: Tensor, *, dm: DataModule, itr: int, prefix: str | None = None) -> None:
        """Log the reconstructed and original images."""
        num_blocks = min(4, len(x))
        rows_per_block = min(8, len(x) // num_blocks)
        num_samples = num_blocks * rows_per_block

        sample = x[:num_samples]
        encoding = self.encoder.encode(sample)
        recon = self.encoder.all_recons(encoding, mode="hard")
        recons = [recon.all, recon.zero_s, recon.just_s]

        caption = "original | all | zero_s | just_s"
        if self.alg_cfg.s_as_zs:
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
