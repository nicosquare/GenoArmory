#!/usr/bin/env bash

tasks=( "H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4" )

function runexp {
    local task=$1

    export GUE_DIR='/projects/p32013/DNABERT-meta/GUE'

    gpu=0                  
    mname="/scratch/hlv8980/GERM_ICML/output_zhihan_vanilla_Full_double/zhihan_vanilla_dnabert2_Full_double_${task}"
    alr=1e-1                # Step size of gradient ascent
    amag=6e-1               # Magnitude of initial perturbation
    anorm=0                 # Maximum norm of adversarial perturbation
    asteps=2                # Number of gradient ascent steps for adversary
    lr=1e-5                 # Learning rate
    bsize=32                # Batch size
    gas=1                   # Gradient accumulation
    seqlen=256              # Maximum sequence length
    hdp=0.1                 # Hidden layer dropout
    adp=0                   # Attention dropout
    ts=2000                 # Number of training steps
    ws=100                  # Warm-up steps
    seed=42                 # Random seed
    wd=1e-2                 # Weight decay
    model_type='dnabert'    # Model type

    expname=${model_type}_${task}

    python examples/run_glue_freelb2.py \
      --model_type ${model_type} \
      --model_name_or_path ${mname} \
      --task_name ${task} \
      --do_train \
      --do_eval \
      --do_lower_case \
      --data_dir $GUE_DIR/$task \
      --max_seq_length ${seqlen} \
      --per_gpu_train_batch_size ${bsize} --gradient_accumulation_steps ${gas} \
      --learning_rate ${lr} --weight_decay ${wd} \
      --gpu ${gpu} \
      --output_dir checkpoints/${expname}/ \
      --hidden_dropout_prob ${hdp} --attention_probs_dropout_prob ${adp} \
      --adv-lr ${alr} --adv-init-mag ${amag} --adv-max-norm ${anorm} --adv-steps ${asteps} \
      --expname ${expname} --evaluate_during_training \
      --max_steps ${ts} --warmup_steps ${ws} --seed ${seed} \
      --logging_steps 100 --save_steps 100 \
      --num_label 2 \
      --overwrite_output_dir \
      > logs/${expname}.log

    echo "---
${task} finish
---"

}

# Iterate over each task in the list
for task in "${tasks[@]}"; do
    runexp "$task"
done
