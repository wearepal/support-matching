"""Submit an array of jobs all at once.

`BASE_FLAGS` determines the flags that all the jobs will share.
`OPTIONS` specifies flags with multiple possible values. The script will submit jobs with *all*
possible combinations.
"""
from itertools import product
from subprocess import run
from time import sleep
from typing import Dict, List, Tuple, Union

BASE_CMD = "qsub -pe smpslots {} python.job {}"
SLOTS = 6
SCRIPT = "run.py"
EMPTY_LIST: List[int] = []  # needed for type hints
BASE_FLAGS: Dict[str, Union[str, int, float, bool, List[int]]] = dict(
    dataset="cmnist",
    enc_levels=4,
    enc_chan=16,
    recon_loss="l2",
    encoder="ae",
    kl_weight=1,
    vae_std_tform="exp",
    log_freq=50,
    filter_labels=[2, 4],
    context_pcnt=0.66666666,
    enc_epochs=90,
    epochs=100,
    lower_threshold=0.3,
    upper_threshold=0.7,
    enc_path="/mnt/data/tk324/fair-cluster-matching/experiments/finn/1590522332.9774063/encoder",
)
OPTIONS: Dict[str, Union[List[str], List[int], List[float], List[bool], List[List[int]]]] = dict(
    method=["pl_enc_no_norm", "pl_output"],
    cl_hidden_dims=[[30], EMPTY_LIST],
    finetune_encoder=[True, False],
    finetune_lr=[1e-6, 1e-5],
    freeze_layers=[2, 3],
    lr=[1e-3, 1e-2],
    pseudo_labeler=["ranking", "cosine"],
)


def main():
    reordered_options: List[List[Tuple[str, Union[str, int, float, bool, List[int]]]]]
    reordered_options = [[(k, v) for v in values] for k, values in OPTIONS.items()]
    all_combinations = product(*reordered_options)
    option_dicts = [dict(list_of_tuples) for list_of_tuples in all_combinations]
    print(f"{len(option_dicts)} combinations.")
    for i, option_dict in enumerate(option_dicts, start=1):
        flags = {**BASE_FLAGS, **option_dict}
        cmd = BASE_CMD.format(SLOTS, SCRIPT)
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
        run(cmd, shell=True, check=True)
        print(f"------------ submitted job {i} of {len(option_dicts)} -----------")


if __name__ == "__main__":
    main()
