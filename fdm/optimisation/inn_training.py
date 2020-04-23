"""Main training file"""
from typing import Dict, List, Tuple, Literal

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fdm.configs import VaeArgs
from fdm.models import PartitionedAeInn

__all__ = ["update_inn", "update_disc_on_inn"]


def update_disc_on_inn(
    args: VaeArgs,
    x_c: Tensor,
    x_t: Tensor,
    inn: PartitionedAeInn,
    disc_ensemble: nn.ModuleList,
    warmup: bool = False,
) -> Tuple[Tensor, float]:
    """Train the discriminator while keeping the generator constant.

    Args:
        x_c: x from the context set
        x_t: x from the training set
    """
    inn.eval()
    disc_ensemble.train()

    ones = x_c.new_ones((x_c.size(0),))
    zeros = x_t.new_zeros((x_t.size(0),))
    # in case of the three-way split, we have to check more than one invariance
    invariances: List[Literal["s", "y"]] = ["s", "y"]

    for _ in range(args.num_disc_updates):
        encoding_t = inn.encode(x_t, stochastic=True)
        encoding_c = inn.encode(x_c, stochastic=True)
        disc_input_c: Tensor
        if args.train_on_recon:
            disc_input_c = inn.decode(encoding_c).detach()  # just reconstruct

        disc_loss = x_c.new_zeros(())
        for invariance in invariances:
            disc_input_t = get_disc_input(args, inn, encoding_t, invariant_to=invariance)
            disc_input_t = disc_input_t.detach()
            if not args.train_on_recon:
                disc_input_c = get_disc_input(args, inn, encoding_c, invariant_to=invariance)
                disc_input_c = disc_input_c.detach()

            for disc in disc_ensemble:
                # discriminator is trained to distinguish `disc_input_c` and `disc_input_t`
                disc_loss_true, disc_acc_true = disc.routine(disc_input_c, ones)
                disc_loss_false, disc_acc_false = disc.routine(disc_input_t, zeros)
                disc_loss += disc_loss_true + disc_loss_false

        if not warmup:
            for disc in disc_ensemble:
                disc.zero_grad()
            disc_loss.backward()
            for disc in disc_ensemble:
                disc.step()

    return disc_loss, 0.5 * (disc_acc_true + disc_acc_false)  # statistics from last step


def update_inn(
    args: VaeArgs,
    x_c: Tensor,
    x_t: Tensor,
    inn: PartitionedAeInn,
    disc_ensemble: nn.ModuleList,
    pred_s_weight: float,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute all losses.

    Args:
        x_t: x from the training set
    """
    # Compute losses for the generator.
    disc_ensemble.eval()
    inn.train()
    logging_dict = {}
    do_recon_stability = args.train_on_recon and args.recon_stability_weight > 0

    # ================================ NLL loss for training set ================================
    # the following code is also in inn.routine() but we need to access ae_enc directly
    zero = x_t.new_zeros((x_t.size(0), 1))
    enc, sum_ldj, ae_enc = inn.encode_with_ae_enc(x_t, sum_ldj=zero)
    nll = inn.nll(enc, sum_ldj)

    # ================================ NLL loss for context set =================================
    # we need a NLL loss for x_c because...
    # ...when we train on encodings, the network will otherwise just falsify encodings for x_c
    # ...when we train on recons, the GAN loss has it too easy to distinguish the two
    zero = x_c.new_zeros((x_c.size(0), 1))
    enc_c, sum_ldj_c, _ = inn.encode_with_ae_enc(x_c, sum_ldj=zero)
    nll += inn.nll(enc_c, sum_ldj_c)
    nll *= 0.5  # take average of the two nll losses

    # =================================== recon stability loss ====================================
    recon_loss = x_t.new_zeros(())
    if do_recon_stability and isinstance(inn, PartitionedAeInn):
        disc_input, ae_recon = get_disc_input(args, inn, enc, invariant_to="s", with_ae_enc=True)
        disc_input_y, ae_recon_y = get_disc_input(
            args, inn, enc, invariant_to="y", with_ae_enc=True
        )
        recon_loss += args.recon_stability_weight * F.mse_loss(ae_recon, ae_enc)
        recon_loss += args.recon_stability_weight * F.mse_loss(ae_recon_y, ae_enc)
    else:
        disc_input = get_disc_input(args, inn, enc, invariant_to="s")
        disc_input_y = get_disc_input(args, inn, enc, invariant_to="y")

    # ==================================== adversarial losses =====================================
    zeros = x_t.new_zeros((x_t.size(0),))

    # discriminators
    disc_loss = x_t.new_zeros(())
    disc_acc_inv_s = 0.0
    for disc in disc_ensemble:
        disc_loss_k, disc_acc_k = disc.routine(disc_input, zeros)
        disc_loss += disc_loss_k
        disc_acc_inv_s += disc_acc_k
    disc_loss /= args.num_discs
    disc_acc_inv_s /= args.num_discs

    # discriminators
    disc_loss_y = x_t.new_zeros(())
    disc_acc_inv_y = 0.0
    for disc in disc_ensemble:
        disc_loss_k, disc_acc_k = disc.routine(disc_input_y, zeros)
        disc_loss_y += disc_loss_k
        disc_acc_inv_y += disc_acc_k
    disc_loss_y /= args.num_discs
    disc_acc_inv_y /= args.num_discs

    disc_loss += disc_loss_y

    nll *= args.nll_weight
    disc_loss *= pred_s_weight
    recon_loss *= args.recon_stability_weight

    gen_loss = nll + recon_loss - disc_loss

    # Update the generator's parameters
    inn.zero_grad()
    gen_loss.backward()
    inn.step()

    final_logging = {
        "Loss Adversarial": disc_loss.item(),
        "Accuracy Disc": disc_acc_inv_s,
        "Accuracy Disc 2": disc_acc_inv_y,
        "Loss NLL": nll.item(),
        "Loss Reconstruction": recon_loss.item(),
        "Loss Generator": gen_loss.item(),
    }
    logging_dict.update(final_logging)

    return gen_loss, logging_dict


def get_disc_input(
    args: VaeArgs,
    inn: PartitionedAeInn,
    encoding: Tensor,
    invariant_to: Literal["s", "y"] = "s",
    with_ae_enc: bool = False,
) -> Tensor:
    """Construct the input that the discriminator expects."""
    if args.train_on_recon:
        zs_m, zy_m = inn.mask(encoding, random=True)
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
