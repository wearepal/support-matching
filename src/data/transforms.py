from functools import partial

import torchvision.transforms as T
import torchvision.transforms.functional as TF

__all__ = ["random_rot90"]


def random_rot90() -> T.RandomApply:
    return T.RandomApply([partial(TF.rotate, angle=(i * 90)) for i in range(0, 4)])
