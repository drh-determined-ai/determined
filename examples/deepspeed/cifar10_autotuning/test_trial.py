# IPython log file

import deepspeed

import determined.pytorch.deepspeed as det_ds

from model_def import CIFARTrial


config = {"hyperparameters": {"deepspeed_config": "ds_config.json"}}
context = det_ds.DeepSpeedTrialContext.from_config(config)

deepspeed.init_distributed()

trial = CIFARTrial(context)

