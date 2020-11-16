"""Main training file"""
from typing import Dict, NamedTuple, Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Literal

from fdm.models import Classifier, PartitionedAeInn
from shared.configs import FdmArgs

__all__ = ["update_inn", "update_disc_on_inn"]


class InnComponents(NamedTuple):
    """Things needed to run the INN model."""

    inn: PartitionedAeInn
    disc_ensemble: nn.ModuleList
    predictor: Optional[Classifier]
    type_: Literal["inn"] = "inn"


def update_disc_on_inn(
    args: FdmArgs, x_c: Tensor, x_t: Tensor, models: InnComponents, warmup: bool = False
) -> Tuple[Tensor, float]:
    """Train the discriminator while keeping the generator constant.

    Args:
        x_c: x from the context set
        x_t: x from the training set
    """
    models.inn.eval()
    models.disc_ensemble.train()

    ones = x_c.new_ones((x_c.size(0),))
    zeros = x_t.new_zeros((x_t.size(0),))
    invariances = ["s"]

    for _ in range(args.num_disc_updates):
        encoding_t = models.inn.encode(x_t, stochastic=args.stochastic)
        encoding_c = models.inn.encode(x_c, stochastic=args.stochastic)
        # disc_input_c: Tensor
        # if args.train_on_recon:
        #     disc_input_c = inn.decode(encoding_c).detach()  # just reconstruct
        disc_input_t: Tensor
        if args.train_on_recon:
            disc_input_t = models.inn.decode(encoding_t).detach()  # just reconstruct

        disc_loss = x_c.new_zeros(())
        for invariance in invariances:
            if not args.train_on_recon:
                disc_input_t = get_disc_input(
                    args, models.inn, encoding_t, invariant_to=invariance, random=False
                )
                disc_input_t = disc_input_t.detach()
            disc_input_c = get_disc_input(
                args, models.inn, encoding_c, invariant_to=invariance, random=False
            )
            disc_input_c = disc_input_c.detach()

            for disc in models.disc_ensemble:
                # discriminator is trained to distinguish `disc_input_c` and `disc_input_t`
                disc_loss_true, disc_acc_true = disc.routine(disc_input_c, ones)
                disc_loss_false, disc_acc_false = disc.routine(disc_input_t, zeros)
                disc_loss += disc_loss_true + disc_loss_false

        if not warmup:
            for disc in models.disc_ensemble:
                disc.zero_grad()
            disc_loss.backward()
            for disc in models.disc_ensemble:
                disc.step()

    return disc_loss, 0.5 * (disc_acc_true + disc_acc_false)  # statistics from last step


def update_inn(
    args: FdmArgs, x_c: Tensor, x_t: Tensor, models: InnComponents, disc_weight: float
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute all losses.

    Args:
        x_t: x from the training set
    """
    # Compute losses for the generator.
    models.disc_ensemble.eval()
    if models.predictor is not None:
        models.predictor.eval()
    models.inn.train()
    logging_dict = {}
    do_recon_stability = args.train_on_recon and args.recon_stability_weight > 0

    # ================================ NLL loss for training set ================================
    # the following code is also in inn.routine() but we need to access ae_enc directly
    zero = x_t.new_zeros((x_t.size(0), 1))
    enc_t, sum_ldj_t, ae_enc_t = models.inn.encode_with_ae_enc(
        x_t, sum_ldj=zero, stochastic=args.stochastic
    )
    nll = models.inn.nll(enc_t, sum_ldj_t)

    # ================================ NLL loss for context set =================================
    # we need a NLL loss for x_c because...
    # ...when we train on encodings, the network will otherwise just falsify encodings for x_c
    # ...when we train on recons, the GAN loss has it too easy to distinguish the two
    zero = x_c.new_zeros((x_c.size(0), 1))
    enc_c, sum_ldj_c, ae_enc_c = models.inn.encode_with_ae_enc(
        x_c, sum_ldj=zero, stochastic=args.stochastic
    )
    nll += models.inn.nll(enc_c, sum_ldj_c)
    nll *= 0.5  # take average of the two nll losses

    invariances = ["s"]

    # ones = x_c.new_ones((x_c.size(0),))
    zeros = x_c.new_zeros((x_c.size(0),))

    recon_loss = x_t.new_zeros(())
    disc_loss = x_t.new_zeros(())
    pred_loss = x_t.new_zeros(())
    if args.train_on_recon and args.pred_weight > 0:
        assert models.predictor is not None
        pred_original, _ = models.predictor.routine(x_c, zeros)

    for invariance in invariances:
        # ================================= recon stability loss ==================================
        if do_recon_stability and isinstance(models.inn, PartitionedAeInn):
            disc_input_inv, ae_recon_inv = get_disc_input(
                args, models.inn, enc_c, invariant_to=invariance, with_ae_enc=True, random=False
            )
            recon_loss += args.recon_stability_weight * F.mse_loss(ae_recon_inv, ae_enc_c)
        else:
            disc_input_inv = get_disc_input(
                args, models.inn, enc_c, invariant_to=invariance, random=False
            )

        # ================================== adversarial losses ===================================
        # discriminators
        disc_loss_i = x_t.new_zeros(())
        disc_acc_inv = 0.0
        for disc in models.disc_ensemble:
            disc_loss_k, disc_acc_k = disc.routine(disc_input_inv, zeros)
            disc_loss_i += disc_loss_k
            disc_acc_inv += disc_acc_k
        disc_loss_i /= args.num_discs
        disc_acc_inv /= args.num_discs
        logging_dict[f"Accuracy Disc {invariance}"] = disc_acc_inv

        disc_loss += disc_loss_i

        # some weird predictor loss
        if args.train_on_recon and args.pred_weight > 0:
            assert models.predictor is not None
            pred_changed, _ = models.predictor.routine(disc_input_inv, zeros)
            pred_loss += (pred_original - pred_changed).abs()

    nll *= args.nll_weight
    disc_loss *= disc_weight
    recon_loss *= args.recon_stability_weight
    pred_loss *= args.pred_weight

    gen_loss = nll + recon_loss + disc_loss + pred_loss

    # Update the generator's parameters
    models.inn.zero_grad()
    gen_loss.backward()
    models.inn.step()

    final_logging = {
        "Loss Adversarial": disc_loss.item(),
        "Loss NLL": nll.item(),
        "Loss Reconstruction": recon_loss.item(),
        "Loss Generator": gen_loss.item(),
        "Loss Prediction": pred_loss.item(),
    }
    logging_dict.update(final_logging)

    return gen_loss, logging_dict


def get_disc_input(
    args: FdmArgs,
    inn: PartitionedAeInn,
    encoding: Tensor,
    invariant_to: Literal["s", "y"] = "s",
    with_ae_enc: bool = False,
    random: bool = True,
) -> Tensor:
    """Construct the input that the discriminator expects."""
    if args.train_on_recon:
        zs_m, zy_m = inn.mask(encoding, random=random)
        if with_ae_enc:
            return inn.decode_with_ae_enc(zs_m if invariant_to == "s" else zy_m)
        else:
            return inn.decode(zs_m if invariant_to == "s" else zy_m)
        # if args.recon_loss == "ce":
        #     recon = recon.argmax(dim=1).float() / 255
        #     if args.dataset != "cmnist":
        #         recon = recon * 2 - 1
    else:
        zs_m, zy_m = inn.mask(encoding)
        return zs_m if invariant_to == "s" else zy_m
