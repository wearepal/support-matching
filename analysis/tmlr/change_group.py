# %%
import wandb

# %%
api = wandb.Api()
# %%
run = api.run("predictive-analytics-lab/hyaline/aj6canem")
# %%
run.group = "dro_baseline_2023-09-27_eta_0.4"
run.update()
# %%
