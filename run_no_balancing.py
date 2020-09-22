"""Only run the fair representation code but pretend that we ran both."""
from fdm.optimisation import main as fdm_main
from shared.utils import accept_prefixes, check_args


def main() -> None:
    """Only run the fair representation code but pretend that we ran both."""
    raw_args = check_args()
    dis_args = accept_prefixes(raw_args, ("--a-", "--d-", "--e-"))
    fdm_main(raw_args=dis_args, known_only=True)


if __name__ == "__main__":
    main()
