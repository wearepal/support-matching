from typing import Callable, Dict, Optional, Sequence, Tuple, overload

import torch
import torch.distributions as td
from torch import Tensor
from torch.utils.data import DataLoader

from clustering.layers import Bijector
from shared.configs import ClusterArgs
from shared.utils import DLogistic, MixtureDistribution, prod  # to_discrete, logistic_distribution

from .autoencoder import AutoEncoder
from .base import Encoder, ModelBase

__all__ = ["AeInn"]


class AeInn(ModelBase, Encoder):
    """Wrapper for classifier models."""

    model: Bijector

    def __init__(
        self,
        args: ClusterArgs,
        model: Bijector,
        autoencoder: AutoEncoder,
        input_shape: Sequence[int],
        optimizer_args: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            args: Runtime arguments.
            model: nn.Module. INN model to wrap around.
            input_shape: Tuple or List. Shape (excluding batch dimension) of the
            input data.
            optimizer_args: Dictionary. Arguments to pass to the optimizer.

        Returns:
            None
        """
        super().__init__(model, optimizer_kwargs=optimizer_args)
        self.input_shape = input_shape
        self.base_density: td.Distribution

        if args.inn_idf:
            probs = 5 * [1 / 5]
            dist_params = [(0, 0.5), (2, 0.5), (-2, 0.5), (4, 0.5), (-4, 0.5)]
            components = [DLogistic(loc, scale) for loc, scale in dist_params]
            self.base_density = MixtureDistribution(probs=probs, components=components)
        else:
            # if args.inn_base_density == "logistic":
            #     self.base_density = logistic_distribution(
            #         torch.zeros(1, device=args._device),
            #         torch.ones(1, device=args._device) * args.inn_base_density_std,
            #     )
            # elif args.inn_base_density == "uniform":
            #     self.base_density = td.Uniform(
            #         low=-torch.ones(1, device=args._device) * args.inn_base_density_std,
            #         high=torch.ones(1, device=args._device) * args.inn_base_density_std,
            #     )
            # else:
            self.base_density = td.Normal(0, 1.0)  # args.inn_base_density_std)
        x_dim: int = input_shape[0]

        self.x_dim: int = x_dim
        if len(input_shape) < 2:
            self.output_dim = self.input_shape[0]
        else:
            self.x_dim = x_dim
            self.output_dim = prod(self.input_shape)

        self.autoencoder = autoencoder

    def decode_with_ae_enc(self, z: Tensor, discretize: bool = False) -> Tuple[Tensor, Tensor]:
        ae_enc, _ = self.model(z, sum_ldj=None, reverse=True)
        x = self.autoencoder.decode(ae_enc, discretize=discretize)
        return x, ae_enc

    def decode(self, z: Tensor, discretize: bool = False) -> Tensor:
        return self.decode_with_ae_enc(z, discretize=discretize)[0]

    @overload
    def encode_with_ae_enc(
        self, inputs: Tensor, sum_ldj: Tensor, stochastic: bool = ...
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ...

    @overload
    def encode_with_ae_enc(
        self, inputs: Tensor, sum_ldj: None = ..., stochastic: bool = ...
    ) -> Tuple[Tensor, None, Tensor]:
        ...

    def encode_with_ae_enc(
        self, inputs: Tensor, sum_ldj: Optional[Tensor] = None, stochastic: bool = False
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        ae_enc = self.autoencoder.encode(inputs, stochastic=stochastic)
        outputs, sum_ldj = self.model(ae_enc, sum_ldj=sum_ldj, reverse=False)
        return outputs, sum_ldj, ae_enc

    def encode(self, inputs: Tensor, stochastic: bool = False) -> Tensor:
        return self.encode_with_ae_enc(inputs, stochastic=stochastic)[0]

    def compute_log_pz(self, z: Tensor) -> Tensor:
        """Log of the base probability: log(p(z))"""
        return self.base_density.log_prob(z)

    def nll(self, z: Tensor, sum_logdet: Tensor) -> Tensor:
        log_pz = self.compute_log_pz(z)
        log_px = log_pz.sum() - sum_logdet.sum()
        # if z.dim() > 2:
        #     log_px_per_dim = log_px / z.nelement()
        #     bits_per_dim = -(log_px_per_dim - np.log(256)) / np.log(2)
        #     return bits_per_dim
        # else:
        return -log_px / z.nelement()

    def routine(self, data: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """Training routine for the Split INN.

        Args:
            data: Tensor. Input Data to the INN.

        Returns:
            Tuple of classification loss (Tensor) and accuracy (float)
        """
        zero = data.new_zeros(data.size(0), 1)
        z, sum_ldj, _ = self.encode_with_ae_enc(data, sum_ldj=zero)
        nll = self.nll(z, sum_ldj)

        return z, nll

    def forward(self, inputs: Tensor, reverse: bool = False) -> Tensor:
        if reverse:
            return self.decode(inputs)
        else:
            return self.encode(inputs)

    def fit_ae(
        self,
        train_data: DataLoader,
        epochs: int,
        device: torch.device,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        kl_weight: float,
    ) -> None:
        print("===> Fitting Auto-encoder to the training data....")
        self.autoencoder.train()
        self.autoencoder.fit(train_data, epochs, device, loss_fn, kl_weight)
        self.autoencoder.eval()

    def train(self) -> None:
        self.model.train()
        self.autoencoder.eval()

    def eval(self) -> None:
        self.model.eval()
        self.autoencoder.eval()
