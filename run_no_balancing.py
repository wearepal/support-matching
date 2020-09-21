"""Only run the fair representation code but pretend that we ran both."""
import sys

from fdm.optimisation import main as fdm_main


def main():
    """Only run the fair representation code but pretend that we ran both."""
    raw_args = sys.argv[1:]
    if not all(
        (not arg.startswith("--")) or arg.startswith(("--c-", "--d-", "--b-")) for arg in raw_args
    ):
        print(
            "\nUse --b- to prefix those flags that will be passed to both parts of the code.\n"
            "Use --d- to prefix those flags that will only be passed to the disentangling code.\n"
            "So, for example: --b-dataset cmnist --c-epochs 100"
        )
        raise RuntimeError("all flags have to use the prefix '--b-', '--c-' or '--d-'.")
    dis_args = [arg.replace("--d-", "--").replace("--b-", "--") for arg in raw_args]
    fdm_main(raw_args=dis_args, known_only=True)


if __name__ == "__main__":
    main()
