"""Submit an array of jobs all at once.

`BASE_FLAGS` determines the flags that all the jobs will share.
`OPTIONS` specifies flags with multiple possible values. The script will submit jobs with *all*
possible combinations.
"""
from itertools import product
from subprocess import run
from time import sleep
from typing import Dict, List, Union, Tuple

BASE_CMD = "python {}"
SLOTS = 6
SCRIPT = "run_cl.py @flags/adult_clustering_a_new_hope.yaml"
EMPTY_LIST: List[int] = []  # needed for type hints
BASE_FLAGS: Dict[str, Union[str, int, float, bool, List[int]]] = dict(
    epochs=200,
    pseudo_labeler="ranking",
    gpu=1,
    batch_size=4000,
    enc_path="/its/home/tk324/dev/fair-dist-matching/experiments/finn/1591138957.9711626/encoder",
)
OPTIONS: Dict[str, Union[List[str], List[int], List[float], List[bool], List[List[int]]]] = dict(
    finetune_encoder=[True, False],
    lr=[1e-3, 1e-4],
    sup_bce_weight=[0., 1.],
    sup_ce_weight=[0., .3, 1.],
    k_num=[2, 3, 4],
    weight_decay=[1e-4, 1e-3],
)


def main():
    reordered_options: List[List[Tuple[str, Union[str, int, float, bool, List[int]]]]]
    reordered_options = [[(k, v) for v in values] for k, values in OPTIONS.items()]
    all_combinations = product(*reordered_options)
    option_dicts = [dict(list_of_tuples) for list_of_tuples in all_combinations]
    print(f"{len(option_dicts)} combinations.")
    for i, option_dict in enumerate(option_dicts, start=1):
        flags = {**BASE_FLAGS, **option_dict}
        cmd = BASE_CMD.format(SCRIPT)  # SLOTS, SCRIPT)
        for k, v in flags.items():
            cmd += f" --{k.replace('_', '-')}"
            if isinstance(v, str):
                cmd += f' "{v}"'
            elif isinstance(v, list):
                if v:
                    cmd += " " + " ".join(str(value) for value in v)
            else:
                cmd += f" {v}"
        print(cmd)
        sleep(1)  # give the user some time to abort and the cluster some time to process the job
        run(cmd, shell=True)#, check=True)
        print(f"------------ submitted job {i} of {len(option_dicts)} -----------")


if __name__ == "__main__":
    main()
