from argparse import Action
from typing import Any, Dict, List, get_type_hints

import ethicml as em
from ethicml.data.tabular_data.adult import AdultSplits
from tap import Tap
from typing_extensions import Literal

__all__ = ["BaseArgs", "StoreDictKeyPair"]


class StoreDictKeyPair(Action):
    """Action for parsing dictionaries on the commandline."""

    def __init__(
        self, option_strings: Any, key_type: type, value_type: type, *args: Any, **kwargs: Any
    ):
        self._key_type = key_type
        self._value_type = value_type
        super().__init__(option_strings, *args, **kwargs)

    def __call__(self, parser: Any, namespace: Any, values: Any, option_string: Any = None) -> None:
        my_dict = {}
        for key_value in values:
            key, value = key_value.split("=")
            my_dict[self._key_type(key.strip())] = self._value_type(value.strip())
        setattr(namespace, self.dest, my_dict)


class BaseArgs(Tap):
    """General data set settings."""

    dataset: Literal["adult", "cmnist", "celeba", "genfaces"] = "cmnist"

    data_pcnt: float = 1.0  # data pcnt should be a real value > 0, and up to 1
    biased_train: bool = True  # if True, make the training set biased, dependent on mixing factor
    mixing_factor: float = 0.0  # How much of context should be mixed into training?
    context_pcnt: float = 0.4
    test_pcnt: float = 0.2
    data_split_seed: int = 888
    root: str = ""

    # Dataset manipulation
    missing_s: List[int] = []

    # Adult data set feature settings
    drop_native: bool = True
    adult_split: AdultSplits = "Sex"
    drop_discrete: bool = False
    balanced_context: bool = False
    balanced_test: bool = True
    balance_all_quadrants: bool = True
    oversample: bool = False  # Whether to oversample when doing weighted sampling.

    # Colored MNIST settings
    scale: float = 0.02
    greyscale: bool = False
    background: bool = False
    black: bool = True
    binarize: bool = True
    rotate_data: bool = False
    shift_data: bool = False
    color_correlation: float = 1.0
    padding: int = 2  # by how many pixels to pad the cmnist images by
    quant_level: Literal["3", "5", "8"] = "8"  # number of bits that encode color
    # the subsample flags work like this: you give it a class id and a fraction in the form of a
    # float. the class id is given by class_id = y * s_count + s, so for binary s and y, the
    # correspondance is like this:
    # 0: y=0/s=0, 1: y=0/s=1, 2: y=1/s=0, 3: y=1/s=1
    subsample_context: Dict[int, float] = {}
    subsample_train: Dict[int, float] = {}
    input_noise: bool = True  # add uniform noise to the input
    filter_labels: List[int] = []
    colors: List[int] = []

    # CelebA settings
    celeba_sens_attr: em.CelebAttrs = "Male"
    celeba_target_attr: em.CelebAttrs = "Smiling"

    # GenFaces settings
    genfaces_sens_attr: em.GenfacesAttributes = "gender"
    genfaces_target_attr: em.GenfacesAttributes = "emotion"

    # Cluster settings
    cluster_label_file: str = ""

    # General settings
    use_wandb: bool = True

    # Global variables
    _s_dim: int
    _y_dim: int

    def add_arguments(self) -> None:
        super().add_arguments()
        self.add_argument(
            "--subsample-context",
            action=StoreDictKeyPair,
            nargs="*",
            default={},
            type=str,
            key_type=int,
            value_type=float,
        )
        self.add_argument(
            "--subsample-train",
            action=StoreDictKeyPair,
            nargs="*",
            default={},
            type=str,
            key_type=int,
            value_type=float,
        )

    def process_args(self) -> None:
        super().process_args()
        if not 0 < self.data_pcnt <= 1:
            raise ValueError("data_pcnt has to be between 0 and 1")

    def convert_arg_line_to_args(self, arg_line: str) -> List[str]:
        """Parse each line like a YAML file."""
        arg_line = arg_line.split("#", maxsplit=1)[0]  # split off comments
        if not arg_line.strip():  # empty line
            return []
        key, value = arg_line.split(sep=":", maxsplit=1)
        key = key.rstrip()
        value = value.strip()
        if key[0] in (" ", "\t"):
            key = key.strip()
            return [f"{key}={value}"]
        if not value:  # no associated value
            values = []
        elif value[0] == '"' and value[-1] == '"':  # if wrapped in quotes, don't split further
            values = [value[1:-1]]
        else:
            values = value.split()
        if len(values) == 1 and values[0] in ("true", "false"):
            values = [values[0].title()]
        key = key.replace("_", "-")
        return [f"--{key}"] + values

    def as_dict(self) -> Dict[str, Any]:
        """Returns the member variables corresponding to the class variable arguments.

        :return: A dictionary mapping each argument's name to its value.
        """
        if not self._parsed:
            raise ValueError("You should call `parse_args` before retrieving arguments.")

        return {var: getattr(self, var) for var in self._get_argument_names()}

    def _get_annotations(self) -> Dict[str, Any]:
        """Returns a dictionary mapping variable names to their type annotations."""
        all_annotations = self._get_from_self_and_super(
            extract_func=lambda super_class: dict(get_type_hints(super_class))
        )
        return {k: v for k, v in all_annotations.items() if not k.startswith("_")}

    def _add_arguments(self) -> None:
        """Add arguments to self in the order they are defined as class variables (so the help string is in order)."""
        # Add class variables (in order)
        for variable in self._annotations:
            if variable in self.argument_buffer:
                name_or_flags, kwargs = self.argument_buffer[variable]
                self._add_argument(*name_or_flags, **kwargs)
            else:
                flag_name = variable.replace("_", "-") if self._underscores_to_dashes else variable
                self._add_argument(f"--{flag_name}")

        # Add any arguments that were added manually in add_arguments but aren't class variables (in order)
        for variable, (name_or_flags, kwargs) in self.argument_buffer.items():
            if variable not in self._annotations:
                self._add_argument(*name_or_flags, **kwargs)

    def __str__(self) -> str:
        """Returns a string representation of self.

        Returns:
            A formatted string representation of the dictionary of all arguments.
        """
        args_dict = self.as_dict()
        formatted = []
        for k in sorted(args_dict):
            v = args_dict[k]
            formatted.append(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}")
        return "Namespace(" + ", ".join(formatted) + ")"
