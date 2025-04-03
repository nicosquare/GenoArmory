import os
import argparse

parser = argparse.ArgumentParser(description='Run adversarial training pipeline for specified task.')
parser.add_argument('--task', type=str, required=True, help='Task name to process')

args = parser.parse_args()
task = args.task
 
# 1.2 Use TextFooler as the attack method to produce adversarial examples:
command3 = f'python attack_classification_simplified.py --dataset_path /projects/p32013/DNABERT-meta/GUE/{task}/cat.csv ' \
          '--target_model bert ' \
          f'--target_model_path /scratch/hlv8980/GERM_ICML/output_zhihan_vanilla_Full_double/zhihan_vanilla_dnabert2_Full_double_{task} ' \
          '--max_seq_length 256 --batch_size 32 ' \
          '--counter_fitting_embeddings_path  /projects/p32013/DNABERT-meta/TextFooler/embeddings/subword_dnabert2_embeddings.txt ' \
          '--counter_fitting_cos_sim_path /projects/p32013/DNABERT-meta/TextFooler/cos_sim_counter_fitting/cos_sim_counter_fitting.npy ' \
          '--USE_cache_path /projects/p32013/DNABERT-meta/TextFooler/tf_cache ' \
          f'--nclasses 2 --output_dir adv_results/{task}' 
          

command4 = 'python get_pure_adversaries.py ' \
  f'--adversaries_path adv_results/{task}/adversaries.txt ' \
  f'--output_path /projects/p32013/DNABERT-meta/GUE/{task}/attacked_data ' \
  '--times 1 ' \
  '--change 0 ' \
  '--txtortsv tsv ' \
  '--datasize 9662'

# 1.3 Construct the training data
command5 = 'python combine_data.py ' \
  f'--add_file /projects/p32013/DNABERT-meta/GUE/{task}/attacked_data/pure_adversaries.tsv ' \
  '--change_label 2 ' \
  f'--original_dataset /projects/p32013/DNABERT-meta/GUE/{task} ' \
  f'--output_path /projects/p32013/DNABERT-meta/GUE/{task}/combined_data/2times_adv_0-3/ ' \
  '--isMR 0'

command6 = 'python run_simplification.py ' \
  '--complex_threshold 3000 ' \
  '--ratio 0.25 ' \
  '--syn_num 20 ' \
  '--most_freq_num 10 ' \
  '--simplify_version random_freq_v1 ' \
  f'--file_to_simplify /projects/p32013/DNABERT-meta/GUE/{task}/combined_data/2times_adv_0-3/train.tsv ' \
  f'--output_path /projects/p32013/DNABERT-meta/GUE/{task}/simplified_data/2times_adv_0-3/ '

command7 = 'python combine_data.py ' \
  f'--add_file /projects/p32013/DNABERT-meta/GUE/{task}/simplified_data/2times_adv_0-3/train.tsv ' \
  '--change_label 4 ' \
  f'--original_dataset /projects/p32013/DNABERT-meta/GUE/{task}/combined_data/2times_adv_0-3/ ' \
  f'--output_path /projects/p32013/DNABERT-meta/GUE/{task}/combined_data/4times_adv_0-7/ --isMR 0 '

# Step2. Train our proposed model on the constructed training data
command8 = 'python run_classification_adv.py ' \
  '--task_name 0 ' \
  '--max_seq_len 128 ' \
  '--do_train ' \
  '--do_eval ' \
  '--attention 2 ' \
  f'--data_dir /projects/p32013/DNABERT-meta/GUE/{task}/combined_data/4times_adv_0-7 ' \
  f'--output_dir /projects/p32013/DNABERT-meta/ADFAR/src/experiments/GUE/{task}/4times_adv_double_0-7 ' \
  f'--model_name_or_path /scratch/hlv8980/GERM_ICML/output_zhihan_vanilla_Full_double/zhihan_vanilla_dnabert2_Full_double_{task} ' \
  '--per_device_train_batch_size 2 ' \
  '--per_device_eval_batch_size 2 ' \
  '--save_total_limit 2 ' \
  '--learning_rate 3e-5 ' \
  '--num_train_epochs 5 ' \
  '--svd_reserve_size 0 ' \
  '--evaluation_strategy epoch ' \
  '--overwrite_output_dir '

# os.system(command3)
# os.system(command4)
# os.system(command5)
# os.system(command6)
# os.system(command7)
os.system(command8)

