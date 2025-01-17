#!/bin/sh
export WANDB_PROJECT=instruct_dial-t0
export WANDB_DISABLED=true

cd scripts
module load gcc-7.4

deepspeed ./run_traint0.py \
    --model_name_or_path prakharz/DIAL-T0 \
    --do_train \
    --do_eval \
    --train_file text2textfiles/train.json \
    --validation_file text2textfiles/validation.json \
    --text_column prompt \
    --target_column output \
    --output_dir ./output \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps 36\
    --gradient_checkpointing \
    --learning_rate 5e-05 \
    --deepspeed t0ds-config.json \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_total_limit 10\
    --bf16\
    --evaluation_strategy steps\
    --num_train_epochs 5\
    --load_best_model_at_end\
    --metric_for_best_model f1 \
    --save_steps 200\
    --eval_steps 200\
    --logging_steps 25
