# Notes

## Pre-training an autoencoder for support-matching

### Run support-matching without the discriminator and save the weights

Turn off the discriminator
```yaml
alg:
    ga_steps: 1
    num_disc_updates: 0
    twoway_disc_loss: false
    prior_loss_w: 0
    pred_y_loss_w: 0
    pred_s_loss_w: 0
    warmup_steps: 0
    disc_loss_w: 0
```

Set `artifact_name` (it’s a top-level config value) to a string.

### Use the weight artifact

Select `ae_arch=artifact` and then set `ae_arch.artifact_name` to whatever you chose in step 1.


## Saving pre-defined dataset splits

```python
>>> from conduit.data.datasets.vision import NICOPP
>>> from src.data.splitter import save_split_inds_as_artifact
>>> import wandb
>>> run = wandb.init(project="support-matching", entity= "predictive-analytics-lab", dir="local_logging")
>>> ds = NICOPP(root="/srv/galene0/shared/data")
>>> import numpy as np
>>> import torch
>>> train_inds = torch.as_tensor(np.nonzero(ds.metadata["split"] == NICOPP.Split.TRAIN.value)[0])
>>> test_inds = torch.as_tensor(np.nonzero(ds.metadata["split"] == NICOPP.Split.TEST.value)[0])
>>> dep_inds = torch.as_tensor(np.nonzero(ds.metadata["split"] == NICOPP.Split.VAL.value)[0])
>>> save_split_inds_as_artifact(
... run=run,
... train_inds=train_inds,
... test_inds=test_inds,
... dep_inds=dep_inds,
... ds=ds,
... seed=0,
... artifact_name="nicopp_change_is_hard_split",
... )
>>> run.finish()
```
