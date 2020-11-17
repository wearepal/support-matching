from typing import List, Optional, Tuple

from fdm import layers
from fdm.models import Classifier
from shared.configs import FdmArgs, InnRM, InnSc
from shared.utils import ModelFn

__all__ = ["build_discriminator", "build_fc_inn", "build_conv_inn"]


def build_discriminator(
    input_shape: Tuple[int, ...],
    target_dim: int,
    model_fn: ModelFn,
    optimizer_kwargs: Optional[dict] = None,
) -> Classifier:
    in_dim = input_shape[0]

    num_classes = target_dim if target_dim > 1 else 2
    return Classifier(
        model_fn(in_dim, target_dim), num_classes=num_classes, optimizer_kwargs=optimizer_kwargs
    )


def build_fc_inn(
    args: FdmArgs, input_shape: Tuple[int, ...], level_depth: Optional[int] = None
) -> layers.Bijector:
    """Build the model with args.inn_depth many layers

    If args.inn_glow is true, then each layer includes 1x1 convolutions.
    """
    input_dim = input_shape[0]
    level_depth = level_depth or args.inn_level_depth

    chain: List[layers.Bijector] = [layers.Flatten()] if len(input_shape) > 1 else []
    for i in range(level_depth):
        if args.inn_batch_norm:
            chain += [layers.MovingBatchNorm1d(input_dim, bn_lag=args.inn_bn_lag)]
        if args.inn_glow:
            chain += [layers.InvertibleLinear(input_dim)]
        chain += [
            layers.MaskedCouplingLayer(
                input_dim=input_dim,
                hidden_dims=args.inn_coupling_depth * [args.inn_coupling_channels],
                mask_type="alternate",
                swap=(i % 2 == 0) and not args.inn_glow,
                scaling=args.inn_scaling,
            )
        ]

    # one last mixing of the channels
    if args.inn_glow:
        chain += [layers.InvertibleLinear(input_dim)]
    else:
        chain += [layers.RandomPermutation(input_dim)]

    return layers.BijectorChain(chain)


def _block(args: FdmArgs, input_dim: int) -> layers.Bijector:
    """Construct one block of the conv INN"""
    _chain: List[layers.Bijector] = []

    if args.inn_idf:
        _chain += [
            layers.IntegerDiscreteFlow(input_dim, hidden_channels=args.inn_coupling_channels)
        ]
        _chain += [layers.RandomPermutation(input_dim)]
    else:
        if args.inn_batch_norm:
            _chain += [layers.MovingBatchNorm2d(input_dim, bn_lag=args.inn_bn_lag)]
        if args.inn_glow:
            _chain += [layers.Invertible1x1Conv(input_dim, use_lr_decomp=True)]
        else:
            _chain += [layers.RandomPermutation(input_dim)]

        if args.inn_scaling == InnSc.none:
            _chain += [
                layers.AdditiveCouplingLayer(
                    input_dim,
                    hidden_channels=args.inn_coupling_channels,
                    num_blocks=args.inn_coupling_depth,
                    pcnt_to_transform=0.25,
                )
            ]
        elif args.inn_scaling == InnSc.sigmoid0_5:
            _chain += [
                layers.AffineCouplingLayer(
                    input_dim,
                    num_blocks=args.inn_coupling_depth,
                    hidden_channels=args.inn_coupling_channels,
                )
            ]
        else:
            raise ValueError(f"Scaling {args.inn_scaling} is not supported")

    # if args.inn_jit:
    #     block = jit.script(block)
    return layers.BijectorChain(_chain)


def _build_multi_scale_chain(
    args: FdmArgs, input_dim, factor_splits, unsqueeze=False
) -> List[layers.Bijector]:
    chain: List[layers.Bijector] = []

    for i in range(args.inn_levels):
        level: List[layers.Bijector] = []
        squeeze: layers.Bijector
        if args.inn_reshape_method == InnRM.haar:
            squeeze = layers.HaarDownsampling(input_dim)
        else:
            squeeze = layers.SqueezeLayer(2)

        if not unsqueeze:
            level += [squeeze]

        input_dim *= 4

        level += [_block(args, input_dim) for _ in range(args.inn_level_depth)]

        if unsqueeze:  # when unsqueezing, the unsqueeze layer has to come after the block
            level += [layers.InvertBijector(to_invert=squeeze)]

        level_layer = layers.BijectorChain(level)
        # if args.inn_jit:
        #     level_layer = jit.script(level_layer)
        chain.append(level_layer)
        if i in factor_splits:
            input_dim = round(factor_splits[i] * input_dim)
    return chain


def build_conv_inn(args: FdmArgs, input_shape: Tuple[int, ...]) -> layers.Bijector:
    input_dim = input_shape[0]

    full_chain: List[layers.Bijector] = []

    factor_splits = {int(k): float(v) for k, v in args.inn_factor_splits.items()}
    main_chain = _build_multi_scale_chain(args, input_dim, factor_splits)

    if args.inn_oxbow_net:
        up_chain = _build_multi_scale_chain(args, input_dim, factor_splits, unsqueeze=True)
        # TODO: don't flatten here and instead split correctly along the channel axis
        full_chain += [layers.OxbowNet(main_chain, up_chain, factor_splits), layers.Flatten()]
    else:
        full_chain += [layers.FactorOut(main_chain, factor_splits)]

    # flattened_shape = int(product(input_shape))
    # full_chain += [layers.RandomPermutation(flattened_shape)]

    # if args.inn_jit:
    #     model = jit.script(model)
    return layers.BijectorChain(full_chain)
