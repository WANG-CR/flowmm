#!/bin/bash

# Define variables
ckpt='/root/code/flowmm/runs/trash/2024-12-11/12-28-48/null_params_null-rfm_cspnet_20NN-sszai141/rfmcsp-conditional-2DGraphene_wHami/0uco2c3g/checkpoints/epoch=1644-step=21385.ckpt'
subdir='eval'
slope='1.0'

# Execute the commands
# python scripts_model/evaluate.py reconstruct "${ckpt}" --subdir "${subdir}" --inference_anneal_slope "${slope}" --stage test --single_gpu && \
# python scripts_model/evaluate.py consolidate "${ckpt}" --subdir "${subdir}" && \
# python scripts_model/evaluate.py old_eval_metrics "${ckpt}" --subdir "${subdir}" --stage test && \
# python scripts_model/evaluate.py lattice_metrics "${ckpt}" --subdir "${subdir}" --stage test

# python scripts_model/evaluate.py reconstruct "${ckpt}" --subdir "${subdir}" --inference_anneal_slope "${slope}" --stage test

python scripts_model/evaluate.py consolidate "${ckpt}" --subdir "${subdir}" && \
python scripts_model/evaluate.py old_eval_metrics "${ckpt}" --subdir "${subdir}" --stage test