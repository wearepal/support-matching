import torch
from torch import Tensor, jit
from torch.nn import functional as F

__all__ = [
    "RoundSTE",
    "logit",
    "normalized_softmax",
    "sum_except_batch",
    "to_discrete",
]


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        return inputs.round()

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-through estimator
        """
        return grad_output


def to_discrete(inputs, dim=1):
    if inputs.dim() <= 1 or inputs.size(1) <= 1:
        return inputs.round()
    else:
        argmax = inputs.argmax(dim=1)
        return F.one_hot(argmax, num_classes=inputs.size(1))


def logit(p, eps=1e-8):
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p / (1.0 - p))


def sum_except_batch(x, keepdim: bool = False):
    return x.flatten(start_dim=1).sum(-1, keepdim=keepdim)


@jit.script
def normalized_softmax(logits: Tensor) -> Tensor:
    max_logits, _ = logits.max(dim=1, keepdim=True)
    unnormalized = torch.exp(logits - max_logits)
    return unnormalized / unnormalized.norm(p=2, dim=-1, keepdim=True)
