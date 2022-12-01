python -m pip install deepspeed
jq --slurp '.[0] + .[1]' ds_config.json ds_config_fp16_tune.json > .ds_config.json
deepspeed --autotuning tune autotuning.py --deepspeed --deepspeed_config .ds_config.json
mv ./autotuning_results/profile_model_info/model_info.json ds_tuned.json
