from __future__ import annotations
from collections.abc import Iterator, Sequence

from kit import implements
import torch
from torch import Tensor
import torch.nn as nn

from advrep.models import AutoEncoder, Discriminator, SplitEncoding
from advrep.models.configs import Residual64x64Net, Strided28x28Net
from advrep.optimisation import log_attention, log_images, mmd2
from shared.configs import (
    AggregatorType,
    CmnistConfig,
    Config,
    DiscriminatorMethod,
    ReconstructionLoss,
)
from shared.data import Batch
from shared.layers import Aggregator, GatedAttentionAggregator, KvqAttentionAggregator
from shared.models.configs import FcNet, ModelAggregatorWrapper
from shared.utils import ModelFn, prod

from .adv import AdvSemiSupervisedAlg

__all__ = ["SupportMatching"]


class SupportMatching(AdvSemiSupervisedAlg):
    adversary: Discriminator

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        if self.adapt_cfg.aggregator_type != AggregatorType.none:
            self.eff_batch_size *= self.adapt_cfg.bag_size

    @implements(AdvSemiSupervisedAlg)
    def _build_adversary(self, input_shape: tuple[int, ...], s_dim: int) -> Discriminator:
        """Build the adversarial network."""
        disc_input_shape: tuple[int, ...] = (
            input_shape if self.adapt_cfg.train_on_recon else (self.enc_cfg.out_dim,)
        )
        disc_fn: ModelFn
        if len(input_shape) > 2 and self.adapt_cfg.train_on_recon:
            if isinstance(self.data_cfg, CmnistConfig):
                disc_fn = Strided28x28Net(batch_norm=False)
            else:
                disc_fn = Residual64x64Net(batch_norm=False)

        else:
            disc_fn = FcNet(hidden_dims=self.adapt_cfg.adv_hidden_dims, activation=nn.GELU())
            # FcNet first flattens the input
            disc_input_shape = (
                (prod(disc_input_shape),)
                if isinstance(disc_input_shape, Sequence)
                else disc_input_shape
            )

        if self.adapt_cfg.aggregator_type is not AggregatorType.none:
            final_proj = (
                FcNet(self.adapt_cfg.aggregator_hidden_dims)
                if self.adapt_cfg.aggregator_hidden_dims
                else None
            )
            aggregator: Aggregator
            if self.adapt_cfg.aggregator_type is AggregatorType.kvq:
                aggregator = KvqAttentionAggregator(
                    latent_dim=self.adapt_cfg.aggregator_input_dim,
                    bag_size=self.adapt_cfg.bag_size,
                    final_proj=final_proj,
                    **self.adapt_cfg.aggregator_kwargs,
                )
            else:
                aggregator = GatedAttentionAggregator(
                    in_dim=self.adapt_cfg.aggregator_input_dim,
                    bag_size=self.adapt_cfg.bag_size,
                    final_proj=final_proj,
                    **self.adapt_cfg.aggregator_kwargs,
                )
            disc_fn = ModelAggregatorWrapper(
                disc_fn, aggregator, input_dim=self.adapt_cfg.aggregator_input_dim
            )

        return Discriminator(
            model=disc_fn(disc_input_shape, 1),  # type: ignore
            double_adv_loss=self.adapt_cfg.double_adv_loss,
            optimizer_kwargs=self.optimizer_kwargs,
            criterion=self.adapt_cfg.adv_loss,
        )

    @implements(AdvSemiSupervisedAlg)
    def _build_encoder(
        self,
        input_shape: tuple[int, ...],
        s_dim: int,
        feature_group_slices: dict[str, list[slice]] | None = None,
    ) -> AutoEncoder:
        if self.adapt_cfg.s_as_zs and self.adapt_cfg.zs_dim != s_dim:
            raise ValueError(f"zs_dim has to be equal to s_dim ({s_dim}) if `s_as_zs` is True.")
        return super()._build_encoder(input_shape, s_dim, feature_group_slices)

    @implements(AdvSemiSupervisedAlg)
    def _step_adversary(
        self,
        train_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
        context_data_itr: Iterator[tuple[Tensor, Tensor, Tensor]],
    ) -> tuple[Tensor, dict[str, float]]:
        """Train the discriminator while keeping the encoder fixed."""
        self._train("adversary")
        x_tr = self._sample_train(train_data_itr).x
        x_ctx = self._sample_context(context_data_itr=context_data_itr)

        with torch.cuda.amp.autocast(enabled=self.misc_cfg.use_amp):  # type: ignore
            encoding_tr = self.encoder.encode(x_tr, stochastic=True)
            if not self.adapt_cfg.train_on_recon:
                encoding_ctx = self.encoder.encode(x_ctx, stochastic=True)

            if self.adapt_cfg.train_on_recon:
                adv_input_ctx = x_ctx

            adv_loss = x_ctx.new_zeros(())
            logging_dict = {}
            adv_input_tr = self._get_adv_input(encoding_tr)
            adv_input_tr = adv_input_tr.detach()

            if not self.adapt_cfg.train_on_recon:
                with torch.no_grad():
                    adv_input_ctx = self._get_adv_input(encoding_ctx)  # type: ignore

            adv_loss = self.adversary.discriminator_loss(fake=adv_input_tr, real=adv_input_ctx)  # type: ignore

        self._update_adversary(adv_loss)

        return adv_loss, logging_dict

    @implements(AdvSemiSupervisedAlg)
    def _step_encoder(
        self, x_ctx: Tensor, batch_tr: Batch, warmup: bool
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the losses for the encoder and update its parameters."""
        # Compute losses for the encoder.
        self._train("encoder")
        logging_dict = {}

        with torch.cuda.amp.autocast(enabled=self.misc_cfg.use_amp):
            # ============================= recon loss for training set ===========================
            encoding_t, enc_loss_tr, logging_dict_tr = self.encoder.routine(
                batch_tr.x,
                recon_loss_fn=self.recon_loss_fn,
                prior_loss_w=self.adapt_cfg.prior_loss_w,
                s=batch_tr.s if self.adapt_cfg.s_as_zs else None,  # using s for the reconstruction
            )

            # ============================= recon loss for context set ============================
            encoding_c, enc_loss_ctx, logging_dict_ctx = self.encoder.routine(
                x_ctx, self.recon_loss_fn, self.adapt_cfg.prior_loss_w
            )
            logging_dict.update({k: v + logging_dict_ctx[k] for k, v in logging_dict_tr.items()})
            enc_loss_tr = 0.5 * (enc_loss_tr + enc_loss_ctx)  # take average of the two recon losses
            enc_loss_tr *= self.adapt_cfg.enc_loss_w
            logging_dict["Loss Generator"] = enc_loss_tr
            total_loss = enc_loss_tr
            # ================================= adversarial losses ================================
            if not warmup:
                disc_input_tr = self._get_adv_input(encoding_t)
                disc_input_ctx = self._get_adv_input(encoding_c)

                if self.adapt_cfg.adv_method is DiscriminatorMethod.nn:
                    disc_loss = self.adversary.encoder_loss(fake=disc_input_tr, real=disc_input_ctx)

                else:
                    x = disc_input_tr
                    y = self._get_adv_input(encoding_c, detach=True)
                    disc_loss = mmd2(
                        x=x,
                        y=y,
                        kernel=self.adapt_cfg.mmd_kernel,
                        scales=self.adapt_cfg.mmd_scales,
                        wts=self.adapt_cfg.mmd_wts,
                        add_dot=self.adapt_cfg.mmd_add_dot,
                    )
                disc_loss *= self.adapt_cfg.adv_loss_w
                total_loss += disc_loss
                logging_dict["Loss Discriminator"] = disc_loss

            if self.predictor_y is not None:
                # predictor is on encodings; predict y from the part that is invariant to s
                pred_y_loss, pred_y_acc = self.predictor_y.routine(encoding_t.zy, batch_tr.y)
                pred_y_loss *= self.adapt_cfg.pred_y_loss_w
                logging_dict["Loss Predictor y"] = pred_y_loss.item()
                logging_dict["Accuracy Predictor y"] = pred_y_acc
                total_loss += pred_y_loss
            if self.predictor_s is not None:
                pred_s_loss, pred_s_acc = self.predictor_s.routine(encoding_t.zs, batch_tr.s)
                pred_s_loss *= self.adapt_cfg.pred_s_loss_w
                logging_dict["Loss Predictor s"] = pred_s_loss.item()
                logging_dict["Accuracy Predictor s"] = pred_s_acc
                total_loss += pred_s_loss

        logging_dict["Loss Total"] = total_loss

        self._update_encoder(total_loss)

        return total_loss, logging_dict

    def _get_adv_input(self, encoding: SplitEncoding, detach: bool = False) -> Tensor:
        """Construct the input that the discriminator expects; either zy or reconstructed zy."""
        if self.adapt_cfg.train_on_recon:
            zs_m, _ = self.encoder.mask(encoding, random=True, detach=detach)
            recon = self.encoder.decode(zs_m, mode="relaxed")
            if self.enc_cfg.recon_loss is ReconstructionLoss.ce:
                recon = recon.argmax(dim=1).float() / 255
                if not isinstance(self.data_cfg, CmnistConfig):
                    recon = recon * 2 - 1
            return recon
        else:
            zs_m, _ = self.encoder.mask(encoding, detach=detach)
            return self.encoder.unsplit_encoding(zs_m)

    @torch.no_grad()
    @implements(AdvSemiSupervisedAlg)
    def _log_recons(self, x: Tensor, itr: int, prefix: str | None = None) -> None:
        """Log the reconstructed and original images."""

        rows_per_block = 8
        num_blocks = 4
        if self.adapt_cfg.aggregator_type is AggregatorType.none:
            num_sampled_bags = 0  # this is only defined here to make the linter happy
            num_samples = num_blocks * rows_per_block
        else:
            # take enough bags to have 32 samples
            num_sampled_bags = ((num_blocks * rows_per_block - 1) // self.adapt_cfg.bag_size) + 1
            num_samples = num_sampled_bags * self.adapt_cfg.bag_size

        sample = x[:num_samples]
        encoding = self.encoder.encode(sample, stochastic=False)
        recon = self.encoder.all_recons(encoding, mode="hard")
        recons = [recon.all, recon.zero_s, recon.just_s]

        caption = "original | all | zero_s | just_s"
        if self.adapt_cfg.train_on_recon or self.adapt_cfg.s_as_zs:
            recons.append(recon.rand_s)
            caption += " | rand_s"

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

        if self.adapt_cfg.aggregator_type is AggregatorType.gated:
            self.adversary(self._get_adv_input(encoding))
            assert isinstance(self.adversary.model[-1], Aggregator)  # type: ignore
            attention_weights = self.adversary.model[-1].attention_weights  # type: ignore
            log_attention(
                self.cfg,
                images=sample,
                attention_weights=attention_weights,  # type: ignore
                name="attention Weights",
                step=itr,
                nbags=num_sampled_bags,
                ncols=ncols,
                prefix=prefix,
            )
