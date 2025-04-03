cd /projects/p32013/DNABERT-meta/robustness

tasks=("H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac" "prom_core_all" "prom_core_notata" "prom_core_tata" "prom_300_all" "prom_300_notata" "prom_300_tata" "tf0" "tf1" "tf2" "tf3" "tf4" "0" "1" "2" "3" "4" "reconstructed" "covid")

for task in "${tasks[@]}"; do
    python test.py --data_dir /projects/p32013/DNABERT-meta/GUE/${task} \
                --model_name_or_path /scratch/hlv8980/GERM_ICML/output_zhihan_vanilla_Full_double/zhihan_vanilla_dnabert2_Full_double_${task} \
                --task_name 0 --num_label 2 --n_gpu 1 \
                --max_seq_length 256 --batch_size 32 \
                --output_dir /projects/p32013/DNABERT-meta/robustness/results/${task} \
    
    echo "${task} finished"
done


