"""Submit an array of jobs all at once.

`BASE_FLAGS` determines the flags that all the jobs will share.
`OPTIONS` specifies flags with multiple possible values. The script will submit jobs with *all*
possible combinations.
"""
from itertools import product
from subprocess import run
from time import sleep
from typing import Dict, List, Union, Tuple

# BASE_CMD = "qsub -pe smpslots {} python.job {}"
BASE_CMD = "python {}"
SLOTS = 6
SCRIPT = "run_cl.py"
EMPTY_LIST: List[int] = []  # needed for type hints
BASE_FLAGS: Dict[str, Union[str, int, float, bool, List[int]]] = dict(
    dataset="adult",
    drop_native=True,
    mixing_factor=0.0,
    input_noise=False,
    # ====== encoder ======,
    enc_channels=35,
    enc_epochs=100,
    enc_lr=1e-2,
    enc_levels=0,
    recon_loss="mixed",
    init_channels=61,
    kl_weight=0.01,
    # ===== clustering ====,
    val_freq=100,
    lr=1e-2,
    use_multi_head=True,
    encoder="vae",
    epochs=100,
    lower_threshold=0.3,
    upper_threshold=0.7,
    enc_path="/its/home/tk324/dev/fair-dist-matching/experiments/finn/1591052263.8719523/encoder",
    gpu=-1,
)
OPTIONS: Dict[str, Union[List[str], List[int], List[float], List[bool], List[List[int]]]] = dict(
    method=["pl_enc_no_norm", "pl_output"],
    cl_hidden_dims=[[30], EMPTY_LIST],
    labeler_hidden_dims=[[30], EMPTY_LIST],
    finetune_encoder=[True, False],
    finetune_lr=[1e-5],
    lr=[1e-3, 1e-2],
    pseudo_labeler=["ranking", "cosine"],
    weight_decay=[1e-4, 0],
    k_num=[5, 3],
)


def main():
    reordered_options: List[List[Tuple[str, Union[str, int, float, bool, List[int]]]]]
    reordered_options = [[(k, v) for v in values] for k, values in OPTIONS.items()]
    all_combinations = product(*reordered_options)
    option_dicts = [dict(list_of_tuples) for list_of_tuples in all_combinations]
    print(f"{len(option_dicts)} combinations.")
    for i, option_dict in enumerate(option_dicts, start=1):
        flags = {**BASE_FLAGS, **option_dict}
        cmd = BASE_CMD.format(SCRIPT) # SLOTS, SCRIPT)
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
