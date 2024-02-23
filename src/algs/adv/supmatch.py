from dataclasses import dataclass
from typing import Optional, cast
from typing_extensions import Self, override

from conduit.data.structures import TernarySample
from loguru import logger
import torch
from torch import Tensor

from src.arch.predictors import SetPredictor
from src.data.data_module import DataModule
from src.models import Model, SplitLatentAe
from src.models.discriminator import BinaryDiscriminator, NeuralDiscriminator
from src.utils import to_item

from .base import AdvSemiSupervisedAlg, Components, IterDep, IterTr
from .evaluator import Evaluator
from .scorer import Scorer

__all__ = ["SupportMatching"]


@dataclass(repr=False, eq=False)
class SupportMatching(AdvSemiSupervisedAlg):
    @override
    def _get_data_iterators(self, dm: DataModule) -> tuple[IterTr, IterDep]:
        if (self.disc_loss_w > 0) or (self.num_disc_updates > 0):
            if dm.deployment_ids is None:
                logger.warning(
                    "Support matching is enabled but without any balancing of the deployment set "
                    "- this can be achieved by setting 'deployment_ids'."
                )
        dl_tr = dm.train_dataloader(balance=True)
        # The batch size needs to be consistent for the aggregation layer in the setwise neural
        # discriminator
        batch_size: int = dl_tr.batch_sampler.batch_size  # type: ignore
        dl_dep = dm.deployment_dataloader(
            batch_size=batch_size if dm.deployment_ids is None else dm.batch_size_tr
        )
        return iter(dl_tr), iter(dl_dep)

    @override
    def _encoder_loss(
        self,
        comp: Components[BinaryDiscriminator],
        *,
        x_dep: Tensor,
        batch_tr: TernarySample[Tensor],
        warmup: bool,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the losses for the encoder."""
        # Compute losses for the encoder.
        logging_dict = {}
        total_loss = x_dep.new_zeros(())

        with torch.cuda.amp.autocast(enabled=self.use_amp):  # type: ignore
            if self.enc_loss_w > 0:
                # ============================= recon loss for training set ===========================
                encoding_tr, enc_loss_tr, logging_dict_tr = comp.ae.training_step(
                    batch_tr.x,
                    prior_loss_w=self.prior_loss_w,
                    s=batch_tr.s if self.s_as_zs else None,  # using s for the reconstruction
                )

                # ============================ recon loss for deployment set ===========================
                encoding_dep, enc_loss_dep, logging_dict_dep = comp.ae.training_step(
                    x_dep, prior_loss_w=self.prior_loss_w
                )
                logging_dict.update(
                    {k: (v + logging_dict_dep[k]) / 2 for k, v in logging_dict_tr.items()}
                )
                enc_loss_tr = (enc_loss_tr + enc_loss_dep) / 2
                enc_loss_tr *= self.enc_loss_w
                logging_dict["loss/autoencoder"] = to_item(enc_loss_tr)
                total_loss += enc_loss_tr
            else:
                encoding_tr = comp.ae.encode(batch_tr.x)
                encoding_dep = comp.ae.encode(x_dep)
            # ================================= adversarial losses ================================
            if not warmup:
                disc_input_tr = encoding_tr.zy
                if isinstance(comp.disc, NeuralDiscriminator):
                    disc_input_dep = encoding_dep.zy if self.twoway_disc_loss else None
                    disc_loss = comp.disc.encoder_loss(fake=disc_input_tr, real=disc_input_dep)
                else:
                    disc_input_dep = encoding_dep.zy
                    if not self.twoway_disc_loss:
                        disc_input_dep = disc_input_dep.detach()
                    disc_loss = comp.disc.encoder_loss(fake=disc_input_tr, real=disc_input_dep)

                disc_loss *= self.disc_loss_w
                total_loss += disc_loss
                logging_dict["loss/discriminator"] = to_item(disc_loss)

            loss_pred, ld_pred = self._predictor_loss(
                comp=comp, zy=encoding_tr.zy, zs=encoding_tr.zs, y=batch_tr.y, s=batch_tr.s
            )
            logging_dict.update(ld_pred)
            total_loss += loss_pred

        logging_dict["loss/total"] = to_item(total_loss)

        return total_loss, logging_dict

    def _discriminator_loss(
        self, comp: Components[BinaryDiscriminator], *, iterator_tr: IterTr, iterator_dep: IterDep
    ) -> tuple[Tensor, dict[str, float]]:
        """Train the discriminator while keeping the encoder fixed."""
        if isinstance(comp.disc, NeuralDiscriminator):
            x_tr = self._sample_tr(iterator_tr).x
            x_dep = self._sample_dep(iterator_dep)

            with torch.cuda.amp.autocast(enabled=self.use_amp):  # type: ignore
                with torch.no_grad():
                    encoding_tr = comp.ae.encode(x_tr)
                    encoding_dep = comp.ae.encode(x_dep)

                disc_loss = comp.disc.discriminator_loss(fake=encoding_tr.zy, real=encoding_dep.zy)

            return disc_loss, {}
        return torch.zeros((), device=self.device), {}

    def _update_discriminator(self, disc: BinaryDiscriminator) -> None:
        if isinstance(disc, NeuralDiscriminator):
            self._clip_gradients(disc.parameters())
            disc.step(grad_scaler=self.grad_scaler, scaler_update=True)
            disc.zero_grad()

    @override
    def discriminator_step(
        self, comp: Components[BinaryDiscriminator], *, iterator_tr: IterTr, iterator_dep: IterDep
    ) -> None:
        if isinstance(comp.disc, NeuralDiscriminator):
            # Train the discriminator on its own for a number of iterations
            for _ in range(self.num_disc_updates):
                for _ in range(self.ga_steps):
                    loss, _ = self._discriminator_loss(
                        comp=comp, iterator_tr=iterator_tr, iterator_dep=iterator_dep
                    )
                    self.backward(loss / self.ga_steps)
                self._update_discriminator(comp.disc)

    @override
    def fit(self, dm: DataModule, *, ae: SplitLatentAe, disc: Model, evaluator: Evaluator) -> Self:
        if self.s_as_zs and ae.zs_dim != dm.card_s:
            raise ValueError(f"zs_dim has to be equal to s_dim ({dm.card_s}) if `s_as_zs` is True.")

        return super().fit(dm=dm, ae=ae, disc=disc, evaluator=evaluator)

    def fit_evaluate_score(
        self,
        dm: DataModule,
        *,
        ae: SplitLatentAe,
        disc: BinaryDiscriminator,
        evaluator: Evaluator,
        scorer: Scorer,
    ) -> Optional[float]:
        """First fit, then evaluate, then score."""
        disc_model_sd0 = None
        if isinstance(disc, NeuralDiscriminator) and isinstance(disc.model, SetPredictor):
            disc_model_sd0 = disc.model.state_dict()
        assert isinstance(disc, Model)
        super().fit_and_evaluate(dm=dm, ae=ae, disc=disc, evaluator=evaluator)
        # TODO: Generalise this to other discriminator types and architectures
        if disc_model_sd0 is not None:
            disc = cast(NeuralDiscriminator, disc)
            disc.model.load_state_dict(disc_model_sd0)
            assert isinstance(disc.model, SetPredictor)
            return scorer.run(dm=dm, ae=ae, disc=disc.model, device=self.device, use_wandb=True)
