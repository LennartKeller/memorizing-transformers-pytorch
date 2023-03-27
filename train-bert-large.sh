#!/bin/sh
eval "$(conda shell.bash hook)"
export TOKENIZERS_PARALLELISM=true
export WANDB_DIR=/pfs/work7/workspace/scratch/ma_lennkell-mem_bert/wandb
rm -rf .tmp/
conda activate mem_bert && python run_mlm.py \
    --model_name_or_path _test/mem-gbert-large\
    --knn_memory_multiprocessing true \
    --do_whole_word_masking false \
    --do_train true \
    --do_eval \
    --train_file /pfs/work7/workspace/scratch/ma_lennkell-mem_bert/wiki-dump-de.txt \
    --validation_split_percentage 1 \
    --line_by_line true \
    --output_dir /pfs/work7/workspace/scratch/ma_lennkell-mem_bert/pretraining/wiki-de/mem-gbert-large \
    --run_name mem-gbert-large \
    --max_steps 250000 \
    --warmup_ratio 0.05 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 100 \
    --logging_first_step true \
    --evaluation_strategy steps \
    --eval_steps 5000 \
    --seed 42 \
    --bf16 true \
    --label_names labels \