from dataclasses import dataclass
from typing import Dict, Tuple
from typing_extensions import Self

from conduit.data.structures import TernarySample
from ranzen import implements
from ranzen.torch import cross_entropy_loss
import torch
from torch import Tensor

from src.data.data_module import DataModule
from src.models import Model
from src.models.autoencoder import SplitLatentAe
from src.utils import to_item

from .base import AdvSemiSupervisedAlg, Components, IterDep, IterTr
from .evaluator import Evaluator

__all__ = ["MiMin"]


@dataclass(eq=False)
class MiMin(AdvSemiSupervisedAlg):
    label_smoothing: float = 0.0

    @implements(AdvSemiSupervisedAlg)
    def _encoder_loss(
        self,
        comp: Components[Model],
        *,
        x_dep: Tensor,
        batch_tr: TernarySample[Tensor],
        warmup: bool,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute the losses for the encoder and update its parameters."""
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
            _, enc_loss_dep, logging_dict_dep = comp.ae.training_step(
                x_dep, prior_loss_w=self.prior_loss_w
            )
            logging_dict.update({k: v + logging_dict_dep[k] for k, v in logging_dict_tr.items()})
            enc_loss_tr = 0.5 * (enc_loss_tr + enc_loss_dep)  # take average of the two recon losses
            enc_loss_tr *= self.enc_loss_w
            logging_dict["loss/autoencoder"] = to_item(enc_loss_tr)
            total_loss = enc_loss_tr
            # ================================= adversarial losses ================================
            if not warmup:
                disc_loss = -cross_entropy_loss(
                    comp.disc(encoding_t.zy),
                    target=batch_tr.s,
                    label_smoothing=self.label_smoothing,
                )
                disc_loss *= self.disc_loss_w
                total_loss += disc_loss
                logging_dict["loss/discriminator"] = to_item(disc_loss)

            loss_pred, ld_pred = self._predictor_loss(
                comp=comp,
                zy=encoding_t.zy,
                zs=encoding_t.zs,
                y=batch_tr.y,
                s=batch_tr.s,
            )
            logging_dict.update(ld_pred)
            total_loss += loss_pred

        logging_dict["loss/total"] = to_item(total_loss)

        return total_loss, logging_dict

    def _discriminator_loss(
        self,
        comp: Components[Model],
        *,
        iterator_tr: IterTr,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Train the discriminator while keeping the encoder fixed."""
        batch_tr = self._sample_tr(iterator_tr)
        with torch.cuda.amp.autocast(enabled=self.use_amp):  # type: ignore
            with torch.no_grad():
                encoding_tr = comp.ae.encode(batch_tr.x)

            logits = comp.disc(encoding_tr.zy)
            disc_loss = cross_entropy_loss(
                input=logits,
                target=batch_tr.s,
                label_smoothing=self.label_smoothing,
            )

        return disc_loss, {}

    def _update_discriminator(self, disc: Model) -> None:
        self._clip_gradients(disc.parameters())
        disc.step(grad_scaler=self.grad_scaler)
        disc.zero_grad()
        if self.grad_scaler is not None:  # Apply scaling for mixed-precision training
            self.grad_scaler.update()

    @implements(AdvSemiSupervisedAlg)
    def discriminator_step(
        self,
        comp: Components[Model],
        *,
        iterator_tr: IterTr,
        iterator_dep: IterDep,
    ) -> None:
        # Train the discriminator on its own for a number of iterations
        for _ in range(self.num_disc_updates):
            for _ in range(self.ga_steps):
                loss, _ = self._discriminator_loss(comp=comp, iterator_tr=iterator_tr)
                self.backward(loss / self.ga_steps)
            self._update_discriminator(comp.disc)

    @implements(AdvSemiSupervisedAlg)
    def fit(self, dm: DataModule, *, ae: SplitLatentAe, disc: Model, evaluator: Evaluator) -> Self:
        if self.s_as_zs and self.zs_dim != dm.card_s:
            raise ValueError(f"zs_dim has to be equal to s_dim ({dm.card_s}) if `s_as_zs` is True.")

        return super().run(dm=dm, ae=ae, disc=disc, evaluator=evaluator)
