"""Simple, fully-connected autoencoder factory."""

from dataclasses import dataclass

from torch import nn

from src.arch.autoencoder.base import AeFactory, AePair
from src.arch.common import Activation, NormType

__all__ = ["SimpleAE"]


@dataclass
class SimpleAE(AeFactory):
    hidden_dim: int
    num_hidden: int
    latent_dim: int
    activation: Activation = Activation.GELU
    norm: NormType | None = NormType.LN
    dropout_prob: float = 0.0

    def __call__(self, input_shape: tuple[int, int, int]) -> AePair:
        encoder: list[nn.Sequential] = []
        decoder: list[nn.Sequential] = []

        c_in = input_shape[0]
        c_out = self.hidden_dim

        for i in range(self.num_hidden):
            encoder.append(self._linear_block(c_in, c_out))
            decoder.append(self._linear_block(c_out, c_in, last=i == 0))
            c_in = c_out

        encoder.append(self._linear_block(c_in, self.latent_dim, last=True))
        decoder.append(self._linear_block(self.latent_dim, c_in))
        decoder = decoder[::-1]

        return AePair(
            encoder=nn.Sequential(*encoder),
            decoder=nn.Sequential(*decoder),
            latent_dim=self.latent_dim,
        )

    def _linear_block(
        self, in_features: int, out_features: int, *, last: bool = False
    ) -> nn.Sequential:
        block = nn.Sequential()
        block.append(nn.Linear(in_features, out_features))
        if not last:
            if self.norm is not None:
                block.append(self.norm.init(out_features))
            block.append(self.activation.init())
            if self.dropout_prob > 0:
                block.append(nn.Dropout(p=self.dropout_prob))
        return block
