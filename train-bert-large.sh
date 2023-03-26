#!/bin/sh

TOKENIZERS_PARALLELISM=true python run_mlm.py \
    --model_name_or_path _test/mem-gbert-large\
    --knn_memory_multiprocessing false \
    --do_whole_word_masking true \
    --do_train true \
    --train_file /pfs/work7/workspace/scratch/ma_lennkell-mem_bert/wiki-dump-de.txt \
    --validation_split_percentage 1 \
    --line_by_line true \
    --output_dir /pfs/work7/workspace/scratch/ma_lennkell-mem_bert/pretraining/wiki-de/mem-gbert-large \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 256 \
    --logging_strategy steps \
    --logging_steps 1 \
    --logging_first_step true \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --seed 42 \
