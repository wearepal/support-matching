from src.relay import SplitRelay


def main() -> None:
    SplitRelay.with_hydra(instantiate_recursively=False, clear_cache=True, root="conf")


if __name__ == "__main__":
    main()
