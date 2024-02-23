from torch import Tensor
from torch.autograd.function import Function, NestedIOFunction
import torch.distributions as td
from torch.nn import functional as F

__all__ = ["round_ste", "sample_concrete", "discretize"]


class RoundSTE(Function):
    @staticmethod
    def forward(ctx: NestedIOFunction, tensor: Tensor) -> Tensor:  # type: ignore
        return tensor.round()

    @staticmethod
    def backward(ctx: NestedIOFunction, *grad_outputs: Tensor) -> tuple[Tensor]:
        """Straight-through estimator"""
        return grad_outputs  # type: ignore


def round_ste(x: Tensor) -> Tensor:
    return RoundSTE.apply(x)  # type: ignore


def discretize(inputs: Tensor, *, dim: int = 1) -> Tensor:
    if inputs.dim() <= 1 or inputs.size(1) <= 1:
        return inputs.round()
    argmax = inputs.argmax(dim=1)
    return F.one_hot(argmax, num_classes=inputs.size(1))


def sample_concrete(logits: Tensor, *, temperature: float) -> Tensor:
    """Sample from the concrete/gumbel softmax distribution for
    differentiable discretization.
    """
    if logits.dim() <= 1 or logits.size(1) <= 1:
        Concrete = td.RelaxedBernoulli
    else:
        Concrete = td.RelaxedOneHotCategorical
    concrete = Concrete(logits=logits, temperature=temperature)
    return concrete.rsample()  # type: ignore
