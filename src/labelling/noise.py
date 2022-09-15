from enum import Enum
from typing import Optional, Union

from ranzen import str_to_enum
import torch
from torch import Tensor

__all__ = [
    "ClnMetric",
    "centroidal_label_noise",
    "sample_noise_indices",
    "uniform_label_noise",
]


@torch.no_grad()
def sample_noise_indices(
    labels: Tensor, *, level: float, generator: Optional[torch.Generator] = None
) -> Tensor:
    if not 0 <= level <= 1:
        raise ValueError("'noise_level' must be in the range [0, 1].")
    num_to_flip = round(level * len(labels))
    return torch.randperm(len(labels), generator=generator)[:num_to_flip]


@torch.no_grad()
def uniform_label_noise(
    labels: Tensor,
    *,
    indices: Tensor,
    generator: Optional[torch.Generator] = None,
    inplace: bool = True,
) -> Tensor:
    if not inplace:
        labels = labels.clone()
    unique, unique_inv = labels.unique(return_inverse=True)
    unique_inv[indices] += torch.randint(
        low=1,
        high=len(unique),
        size=(len(indices),),
        generator=generator,
    )
    unique_inv[indices] %= len(unique)
    return unique[unique_inv]


class ClnMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


@torch.no_grad()
def centroidal_label_noise(
    labels: Tensor,
    *,
    indices: Tensor,
    encodings: Tensor,
    generator: Optional[torch.Generator] = None,
    inplace: bool = True,
    metric: Union[str, ClnMetric] = ClnMetric.COSINE,
    temperature: float = 1.0,
) -> Tensor:
    assert len(labels) == len(encodings)
    assert temperature > 0
    if isinstance(metric, str):
        metric = str_to_enum(metric.upper(), enum=ClnMetric)

    unique, inv = labels.unique(return_inverse=True)
    if metric is ClnMetric.COSINE:
        norm = encodings.norm(dim=1, p=2, keepdim=True)  # type: ignore
        eps = torch.finfo(encodings.dtype).eps
        norm.clamp_min_(eps)
        if not inplace:
            encodings = encodings.clone()
        encodings /= norm

    zeros = encodings.new_zeros(size=(len(unique), encodings.size(1)))
    scatter_inds = inv.unsqueeze(-1).expand_as(encodings)
    centroids = torch.scatter_reduce(
        input=zeros,
        src=encodings,
        index=scatter_inds,
        dim=0,
        reduce="mean",
        include_self=False,
    )
    encodings_ln = encodings[indices]
    if metric is ClnMetric.COSINE:
        sim = encodings_ln @ centroids.t()
    else:
        sim = torch.cdist(x1=encodings_ln, x2=centroids, p=2).neg()
    del encodings
    num = sim.div_(temperature).exp_()
    # Set the probability of the current label to 0 and exclude it from the partition function
    row_inds = torch.arange(len(num), device=num.device, dtype=torch.long)
    num[row_inds, inv[indices]] = 0.0
    denom = num.sum(dim=1, keepdim=True)
    probs = num / denom
    new_labels = torch.multinomial(
        probs,
        num_samples=1,
        replacement=False,
        generator=generator,
    )
    del probs
    if not inplace:
        labels = labels.clone()
    labels[indices] = new_labels.squeeze(1)
    return labels
