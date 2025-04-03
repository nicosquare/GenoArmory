cd /projects/p32013/DNABERT-meta/robustness

python test.py --data_dir /projects/p32013/DNABERT-meta/BERT-Attack/GUE/mouse/0 \
               --model_name_or_path /scratch/hlv8980/GERM_ICML/output_zhihan_vanilla_Full_double/zhihan_vanilla_dnabert2_Full_double_0 \
               --task_name 0 --num_label 2 --n_gpu 1 \
               --max_seq_length 256 --batch_size 32 \
               --output_dir /projects/p32013/DNABERT-meta/robustness/results/0 \


