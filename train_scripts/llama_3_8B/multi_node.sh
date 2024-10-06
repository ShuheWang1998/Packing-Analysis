MODEL=
DATANAME=
OUTPUTDIR=



python -m torch.distributed.run \
    --nproc_per_node 8 \
    --nnodes 4 \
    --node_rank $PET_NODE_RANK \
    --master_addr $PET_MASTER_ADDR \
    --master_port $PET_MASTER_PORT \
    ./src/train.py \
    --deepspeed ./examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL \
    --dataset $DATANAME \
    --dataset_dir ./data \
    --template llama3 \
    --finetuning_type full \
    --output_dir $OUTPUTDIR \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 64 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_ratio 0.2 \
    --save_steps 7000 \
    --eval_steps 4000 \
    --evaluation_strategy steps \
    --learning_rate 1e-5 \
    --num_train_epochs 4.0 \
    --val_size 0.1 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --packing True \
    --packing_style full-no-cut \
    --bf16

# for batch training:
# --packing False
# for random packing:
# --packing True
# --packing_style full