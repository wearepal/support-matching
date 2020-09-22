import sys
from typing import List, Sequence, Tuple, Optional

__all__ = ["accept_prefixes", "confirm_empty"]


def accept_prefixes(prefixes: Sequence[str], args: Optional[List[str]] = None) -> List[str]:
    """Remove the given prefixes from the args which makes them usable."""
    args = sys.argv[1:] if args is None else args
    for prefix in prefixes:
        args = [arg.replace(prefix, "--") for arg in args]
    return args


def confirm_empty(args: List[str], *, to_ignore: Tuple[str, ...]) -> None:
    """Confirm that the given args don't contain any args that wouldn't be ignored."""
    for arg in args:
        if not arg.startswith("--") or arg.startswith(to_ignore):
            continue
        print(
            f'Unrecognized flag: "{arg}".\n\n'
            "Use no prefix for those flags that will be passed to all parts of the code.\n"
            "Use --b- to prefix those flags that will only be passed to the baseline code.\n"
            "Use --c- to prefix those flags that will only be passed to the clustering code.\n"
            "Use --d- to prefix those flags that will only be passed to the disentangling code.\n"
            "Use --e- to prefix those flags that will be passed to clustering and disentangling.\n"
            "So, for example: --a-dataset cmnist --c-epochs 100 --e-enc-channels 32"
        )
        raise ValueError(f"unknown commandline argument: {arg}")
