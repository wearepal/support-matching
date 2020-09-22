import sys
from typing import List, Sequence, Tuple
from typing_extensions import Final

__all__ = ["FLAG_PREFIXES", "check_args", "accept_prefixes", "confirm_empty"]

FLAG_PREFIXES: Final = ("--a-", "--b-", "--c-", "--d-", "--e-")


def check_args() -> List[str]:
    """Confirm that the arguments have the right prefixes."""
    raw_args = sys.argv[1:]
    for arg in raw_args:
        if not arg.startswith("--") or arg.startswith(FLAG_PREFIXES):
            continue
        print(
            f"Flag with wrong prefix: \"{arg}\".\n\n"
            "Use --a- to prefix those flags that will be passed to all parts of the code.\n"
            "Use --b- to prefix those flags that will only be passed to the baseline code.\n"
            "Use --c- to prefix those flags that will only be passed to the clustering code.\n"
            "Use --d- to prefix those flags that will only be passed to the disentangling code.\n"
            "Use --e- to prefix those flags that will be passed to clustering and disentangling.\n"
            "So, for example: --a-dataset cmnist --c-epochs 100 --e-enc-channels 32"
        )
        joined_prefixes = "'" + "', '".join(FLAG_PREFIXES) + "'"
        raise RuntimeError(f"flag '{arg}' doesn't have one of the prefixes {joined_prefixes}.")
    return raw_args


def accept_prefixes(args: List[str], prefixes: Sequence[str]) -> List[str]:
    """Remove the given prefixes from the args which makes them usable."""
    for prefix in prefixes:
        args = [arg.replace(prefix, "--") for arg in args]
    return args


def confirm_empty(args: List[str], *, to_ignore: Tuple[str, ...]) -> None:
    """Confirm that the given args don't contain any args that wouldn't be ignored."""
    for arg in args:
        if arg.startswith("--") and not arg.startswith(to_ignore):
            raise ValueError(f"unknown commandline argument: {arg}")
