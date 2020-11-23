import math

import numpy as np


class Cell:
    def __init__(self, round_to: int):
        self.round_to = round_to

    def __call__(self, data):
        mean = np.mean(data)
        if math.isnan(mean):
            return "N/A"
        std = np.std(data)
        round_level = self.round_to if std > 2 * pow(10, -self.round_to) else self.round_to + 1
        return f"{round(mean, round_level)} $\\pm$ {round(std, round_level)}"


def generate_table(
    df,
    type_renames,
    metrics,
    query,
    with_classifier: bool,
    metrics_renames: dict = None,
    round_to: int = 2,
):
    col_renames = {"data": "type", "method": "classifier"}
    if metrics_renames is not None:
        col_renames.update(metrics_renames)
    df = df.rename(columns=col_renames, inplace=False)
    df = df.replace({"type": type_renames}, inplace=False)
    base_cols = ["type", "approach", "classifier"] if with_classifier else ["type", "approach"]
    return (
        df[base_cols + metrics]
        .groupby(base_cols, sort=False)
        .agg(Cell(round_to=round_to))
        .query(query)
    )


def to_latex(df):
    print(df.reset_index(level=0, drop=True, inplace=False).to_latex(escape=False))


def simple_to_latex(table):
    print(table.to_latex(index=False, escape=False))
    

def merge_cols(df, correct_col: str, incorrect_col: str):
    df[correct_col] = df[correct_col].combine_first(df[incorrect_col])
