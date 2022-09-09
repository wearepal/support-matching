from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from typing_extensions import Self

from conduit.data.structures import TernarySample
from loguru import logger
from ranzen import implements
import torch
from torch import Tensor

from src.arch.predictors import SetPredictor
from src.data.data_module import DataModule
from src.models.autoencoder import SplitLatentAe
from src.models.discriminator import BinaryDiscriminator, NeuralDiscriminator
from src.utils import to_item

from .base import AdvSemiSupervisedAlg, Components, IterDep, IterTr
from .evaluator import Evaluator
from .scorer import Scorer

__all__ = ["SupportMatching"]


@dataclass(eq=False)
class SupportMatching(AdvSemiSupervisedAlg):
    @implements(AdvSemiSupervisedAlg)
    def _get_data_iterators(self, dm: DataModule) -> Tuple[IterTr, IterDep]:
        if (self.disc_loss_w > 0) or (self.num_disc_updates > 0):
            if dm.deployment_ids is None:
                logger.warning(
                    "Support matching is enabled but without any balancing of the deployment set "
                    "- this can be achieved either by setting 'deployment_ids'."
                )
        dl_tr = dm.train_dataloader(balance=True)
        # The batch size needs to be consistent for the aggregation layer in the setwise neural
        # discriminator
        dl_dep = dm.deployment_dataloader(batch_size=dm.batch_size_tr)
        return iter(dl_tr), iter(dl_dep)

    @implements(AdvSemiSupervisedAlg)
    def _encoder_loss(
        self,
        comp: Components[BinaryDiscriminator],
        *,
        x_dep: Tensor,
        batch_tr: TernarySample[Tensor],
        warmup: bool,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute the losses for the encoder and update its parameters."""
        breakpoint()
        # Compute losses for the encoder.
        logging_dict = {}

        with torch.cuda.amp.autocast(enabled=self.use_amp):  # type: ignore
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
            logging_dict["Loss Encoder"] = to_item(enc_loss_tr)
            total_loss = enc_loss_tr
            # ================================= adversarial losses ================================
            if not warmup:
                disc_input_tr = encoding_t.zy
                if isinstance(comp.disc, NeuralDiscriminator):
                    disc_input_dep = encoding_c.zy if self.twoway_disc_loss else None
                    disc_loss = comp.disc.encoder_loss(fake=disc_input_tr, real=disc_input_dep)
                else:
                    disc_input_dep = encoding_c.zy
                    if not self.twoway_disc_loss:
                        disc_input_dep = disc_input_dep.detach()
                    disc_loss = comp.disc.encoder_loss(fake=disc_input_tr, real=disc_input_dep)

                disc_loss *= self.disc_loss_w
                total_loss += disc_loss
                logging_dict["Loss Discriminator"] = to_item(disc_loss)

            loss_pred, ld_pred = self._predictor_loss(
                comp=comp,
                zy=encoding_t.zy,
                zs=encoding_t.zs,
                y=batch_tr.y,
                s=batch_tr.s,
            )
            logging_dict.update(ld_pred)
            total_loss += loss_pred

        logging_dict["Loss Total"] = to_item(total_loss)

        return total_loss, logging_dict

    def _discriminator_loss(
        self,
        comp: Components[BinaryDiscriminator],
        *,
        iterator_tr: IterTr,
        iterator_dep: IterDep,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Train the discriminator while keeping the encoder fixed."""
        if isinstance(comp.disc, NeuralDiscriminator):
            x_tr = self._sample_tr(iterator_tr).x
            x_dep = self._sample_dep(iterator_dep)

            with torch.cuda.amp.autocast(enabled=self.use_amp):  # type: ignore
                with torch.no_grad():
                    encoding_tr = comp.ae.encode(x_tr)
                    encoding_dep = comp.ae.encode(x_dep)

                disc_loss = comp.disc.discriminator_loss(
                    fake=encoding_tr.zy,
                    real=encoding_dep.zy,
                )

            return disc_loss, {}
        return torch.zeros((), device=self.device), {}

    def _update_discriminator(self, disc: BinaryDiscriminator) -> None:
        if isinstance(disc, NeuralDiscriminator):
            self._clip_gradients(disc.parameters())
            disc.step(grad_scaler=self.grad_scaler)
            disc.zero_grad()
            if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
                self.grad_scaler.update()

    @implements(AdvSemiSupervisedAlg)
    def discriminator_step(
        self,
        comp: Components[BinaryDiscriminator],
        *,
        iterator_tr: IterTr,
        iterator_dep: IterDep,
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

    @implements(AdvSemiSupervisedAlg)
    def fit(
        self, dm: DataModule, *, ae: SplitLatentAe, disc: BinaryDiscriminator, evaluator: Evaluator
    ) -> Self:
        if self.s_as_zs and self.zs_dim != dm.card_s:
            raise ValueError(f"zs_dim has to be equal to s_dim ({dm.card_s}) if `s_as_zs` is True.")

        return super().fit(dm=dm, ae=ae, disc=disc, evaluator=evaluator)

    @implements(AdvSemiSupervisedAlg)
    def run(
        self,
        dm: DataModule,
        *,
        ae: SplitLatentAe,
        disc: BinaryDiscriminator,
        evaluator: Evaluator,
        scorer: Optional[Scorer] = None,
    ) -> Optional[float]:
        self.fit(dm=dm, ae=ae, disc=disc, evaluator=evaluator)
        # TODO: Generalise this to other discriminator types and architectures
        if (
            (scorer is not None)
            and isinstance(disc, NeuralDiscriminator)
            and isinstance(disc.model, SetPredictor)
        ):
            return scorer.run(dm=dm, ae=ae, disc=disc.model, device=self.device)
