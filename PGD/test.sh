cd /projects/p32013/DNABERT-meta/PGD

python test.py --data_dir /projects/p32013/DNABERT-meta/BERT-Attack/GUE/mouse/0 \
               --model_name_or_path /projects/p32013/DNABERT-meta/hyena-dna/hyena/output_pipe/0/origin \
               --task_name 0 --num_label 2 --n_gpu 1 \
               --max_seq_length 256 --batch_size 32 \
               --output_dir /projects/p32013/DNABERT-meta/PGD/results/0 \
               --model_type hyena


