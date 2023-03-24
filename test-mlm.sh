#!/bin/sh
python run_mlm.py \
    --model_name_or_path _test/mem-gbert-large \
    --do_train true \
    --train_file _test/oscar-tiny.txt \
    --validation_split_percentage 5 \
    --output_dir _test/mlm-test \
    --overwrite_output_dir true \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 1 \
    --logging_first_step true \
    --max_seq_length 1024 \
    --max_steps 2 \
