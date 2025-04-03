#!/bin/bash


ckpt=output_zhihan_vanilla_Full_double
folder=zhihan_vanilla_dnabert2_Full_double
type="test"

# --model_name_or_path  "${ckpt_dir}/${ckpt}/${folder}_${checkpoint}" \
ckpt_dir=/scratch/hlv8980/GERM_ICML

# checkpoints=( "H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4" "reconstructed" "covid" )
checkpoints=( "2" )
output_base_dir="/projects/p32013/DNABERT-meta/fimba-attack"

for checkpoint in "${checkpoints[@]}"; do
    accelerate launch --config_file 1gpu_no_mp.yaml validate.py \
    --seed 3000 \
    --preprocessing_num_workers 0 \
    --model_type bert \
    --max_seq_length 128 \
    --mlm_probability 0.15 \
    --per_device_eval_batch_size 64 \
    --data_cache_dir .hf_data \
    --model_cache_dir .hf_cache \
    --model_name_or_path  "${ckpt_dir}/${ckpt}/${folder}_${checkpoint}" \
    --output_dir "${output_base_dir}/${type}/${checkpoint}" \
    --validation_file "/projects/p32013/DNABERT-meta/BERT-Attack/GUE/mouse/$checkpoint/dev.csv" \
    --train_file "/projects/p32013/DNABERT-meta/BERT-Attack/GUE/mouse/$checkpoint/train.csv" \
    --trust_remote_code \
    --gradient_accumulation_steps 1 \
    --run_name "dnabert2_val" 
done
