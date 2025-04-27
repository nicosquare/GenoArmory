cd /projects/p32013/DNABERT-meta/PGD

# tasks=( "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4")
tasks=( "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4")

model='nt2'

for task in "${tasks[@]}"; do
    python test.py --data_dir /projects/p32013/DNABERT-meta/GUE/${task}/five_percent \
                --model_name_or_path /scratch/hlv8980/Attack_Benchmark/models/${model}/${task}/textfooler \
                --task_name 0 --num_label 2 --n_gpu 1 \
                --max_seq_length 256 --batch_size 16 \
                --output_dir /projects/p32013/DNABERT-meta/PGD/results/AT/${model}/${task} \
                --model_type nt --overwrite_cache \
                --tokenizer_name /scratch/hlv8980/Attack_Benchmark/models/${model}/${task}/origin 
    echo "${task} finished"
done


