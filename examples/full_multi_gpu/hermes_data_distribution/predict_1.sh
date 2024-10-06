#!/bin/bash

python -m torch.distributed.run \
    --nproc_per_node $MLP_WORKER_GPU \
    --nnodes $MLP_WORKER_NUM \
    --node_rank $MLP_ROLE_INDEX \
    --master_addr $MLP_WORKER_0_HOST \
    --master_port $MLP_WORKER_0_PORT \
    src/train.py \
    --deepspeed ./examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --do_predict \
    --model_name_or_path /ML-A100/public/model/Meta-Llama-3-70B-Instruct \
    --dataset hermes_05 \
    --dataset_dir ./data \
    --template llama3 \
    --finetuning_type full \
    --output_dir saves/data_distribution/hermes_5 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 64 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --do_sample False \
    --temperature 0 \
    --max_new_tokens 512 \
    --bf16
