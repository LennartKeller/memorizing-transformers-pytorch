#!/bin/sh
rm -rf .tmp/
TOKENIZERS_PARALLELISM=true WANDB_DISABLED=false python run_mlm.py \
    --model_name_or_path _test/mem-gbert-large \
    --knn_memory_multiprocessing true \
    --do_whole_word_masking true \
    --do_train true \
    --do_eval true \
    --train_file /pfs/work7/workspace/scratch/ma_lennkell-mem_bert/wiki-dump-de-tiny.txt \
    --validation_split_percentage 1 \
    --line_by_line true \
    --output_dir _test/mlm-test \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 1 \
    --logging_first_step true \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --seed 42 \
    --bf16 true \
    --label_names labels \
