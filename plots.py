from pathlib import Path
import pandas as pd
import seaborn as sns


def main():

    two_digits_to_plot = pd.DataFrame(
        columns=[
            "Dataset",
            "Experiment",
            "Approach",
            "Seed",
            "Scale",
            "Accuracy",
            "TPR",
            "TNR",
            "Renyi preds and s",
            "prob_pos_sex_Male_0.0",
            "prob_pos_sex_Male_1.0",
            "prob_pos_sex_Male_0.0-sex_Male_1.0",
            "prob_pos_sex_Male_0.0/sex_Male_1.0",
            "TPR_sex_Male_0.0",
            "TPR_sex_Male_1.0",
            "TPR_sex_Male_0.0-sex_Male_1.0",
            "TPR_sex_Male_0.0/sex_Male_1.0",
            "TNR_sex_Male_0.0",
            "TNR_sex_Male_1.0",
            "TNR_sex_Male_0.0-sex_Male_1.0",
            "TNR_sex_Male_0.0/sex_Male_1.0",
        ]
    )

    results_dir = Path('.').parent / "results"
    for csv_file in results_dir.rglob("*.csv"):
        print(str(csv_file).split("/"))
        if str(csv_file).split("/")[3] == "dro":
            _, dataset, experiment, approach, subset, seed, eta, scale, file = str(csv_file).split(
                "/"
            )
            # if subset == "2digits":
            #     temp_df = pd.read_csv(csv_file).drop("Scale", axis=1)
            #     two_digits_to_plot = two_digits_to_plot.append(
            #         {
            #             "Dataset": dataset,
            #             "Experiment": experiment,
            #             "Approach": approach,
            #             "Seed": seed,
            #             "Scale": scale,
            #             **{k: v[0] for k, v in temp_df.to_dict().items()},
            #         },
            #         ignore_index=True,
            #     )
        else:
            _, dataset, experiment, approach, subset, seed, scale, file = str(csv_file).split("/")
            if subset == "2digits" and approach == "full":
                temp_df = pd.read_csv(csv_file).drop("Scale", axis=1)
                two_digits_to_plot = two_digits_to_plot.append(
                    {
                        "Dataset": dataset,
                        "Experiment": experiment,
                        "Approach": approach,
                        "Seed": seed,
                        "Scale": scale,
                        **{k: v[0] for k, v in temp_df.to_dict().items()},
                    },
                    ignore_index=True,
                )
    sns_plot = sns.lineplot(x="Scale", y="Accuracy", data=two_digits_to_plot)
    sns_plot.get_figure().savefig("test.png")


if __name__ == "__main__":
    main()
