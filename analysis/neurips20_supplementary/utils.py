import math
import numpy as np


def cell(data):
    mean = round(np.mean(data), 3)
    if math.isnan(mean):
        return "N/A"
    return f"{mean:.3f} $\\pm$ {round(np.std(data), 3):.3f}"


def generate_table(df, type_renames, metrics, query):
    df.rename(columns={"data": "type", "method": "classifier"}, inplace=True)
    df["type"].replace(type_renames, inplace=True)
    return df[["type", "approach", "classifier"] + metrics].groupby(["type", "approach", "classifier"]).agg(cell).query(query)