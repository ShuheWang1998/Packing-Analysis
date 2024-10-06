#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train.py examples/full_multi_gpu/llama3_full_predict.yaml
