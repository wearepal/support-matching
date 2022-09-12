from src.relay import ArtifactGenRelay


def main() -> None:
    ArtifactGenRelay.with_hydra(instantiate_recursively=False, clear_cache=True, root="conf")


if __name__ == "__main__":
    main()
