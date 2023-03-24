#!/bin/sh
TOKENIZERS_PARALLELISM=true python run_mlm.py \
    --model_name_or_path _test/mem-bert-base-german-cased \
    --do_train true \
    --train_file _test/oscar-tiny.txt \
    --validation_split_percentage 5 \
    --line_by_line true \
    --output_dir _test/mlm-test \
    --overwrite_output_dir true \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --logging_strategy steps \
    --logging_steps 1 \
    --logging_first_step true \
    --max_seq_length 512 \
    --knn_memory_multiprocessing false \
