import os

# Preparation work:
# Pre-compute the cosine similarity scores between word pairs
# based on the counter-fitting word embeddings
command1 = 'python comp_cos_sim_mat.py embeddings/subword_dnabert2_embeddings.txt'


# 1.2 Use TextFooler as the attack method to produce adversarial examples:
command3 = 'python attack_classification_simplified.py --dataset_path /projects/p32013/DNABERT-meta/GUE/0/cat.csv ' \
          '--target_model bert ' \
          '--target_model_path /scratch/hlv8980/GERM_ICML/output_zhihan_vanilla_Full_double/zhihan_vanilla_dnabert2_Full_double_0 ' \
          '--max_seq_length 256 --batch_size 32 ' \
          '--counter_fitting_embeddings_path  /projects/p32013/DNABERT-meta/TextFooler/embeddings/subword_dnabert2_embeddings.txt ' \
          '--counter_fitting_cos_sim_path /projects/p32013/DNABERT-meta/TextFooler/cos_sim_counter_fitting/cos_sim_counter_fitting.npy ' \
          '--USE_cache_path /projects/p32013/DNABERT-meta/TextFooler/tf_cache ' \
          '--nclasses 2 --output_dir adv_results/0' 
          

command4 = 'python get_pure_adversaries.py ' \
  '--adversaries_path adv_results/0/adversaries.txt ' \
  '--output_path /projects/p32013/DNABERT-meta/GUE/0/attacked_data ' \
  '--times 1 ' \
  '--change 0 ' \
  '--txtortsv tsv ' \
  '--datasize 9662'

# 1.3 Construct the training data
command5 = 'python combine_data.py ' \
  '--add_file /projects/p32013/DNABERT-meta/GUE/0/attacked_data/pure_adversaries.tsv ' \
  '--change_label 2 ' \
  '--original_dataset /projects/p32013/DNABERT-meta/GUE/0 ' \
  '--output_path /projects/p32013/DNABERT-meta/GUE/0/combined_data/2times_adv_0-3/ ' \
  '--isMR 0'

command6 = 'python run_simplification.py ' \
  '--complex_threshold 3000 ' \
  '--ratio 0.25 ' \
  '--syn_num 20 ' \
  '--most_freq_num 10 ' \
  '--simplify_version random_freq_v1 ' \
  '--file_to_simplify /projects/p32013/DNABERT-meta/GUE/0/combined_data/2times_adv_0-3/train.tsv ' \
  '--output_path /projects/p32013/DNABERT-meta/GUE/0/simplified_data/2times_adv_0-3/ '

command7 = 'python combine_data.py ' \
  '--add_file /projects/p32013/DNABERT-meta/GUE/0/simplified_data/2times_adv_0-3/train.tsv ' \
  '--change_label 4 ' \
  '--original_dataset /projects/p32013/DNABERT-meta/GUE/0/combined_data/2times_adv_0-3/ ' \
  '--output_path /projects/p32013/DNABERT-meta/GUE/0/combined_data/4times_adv_0-7/ --isMR 0 '

# Step2. Train our proposed model on the constructed training data
command8 = 'python run_classification_adv.py ' \
  '--task_name 0 ' \
  '--max_seq_len 128 ' \
  '--do_train ' \
  '--do_eval ' \
  '--attention 2 ' \
  '--data_dir /projects/p32013/DNABERT-meta/GUE/0/combined_data/4times_adv_0-7 ' \
  '--output_dir experiments/GUE/0/4times_adv_double_0-7 ' \
  '--model_name_or_path /scratch/hlv8980/GERM_ICML/output_zhihan_vanilla_Full_double/zhihan_vanilla_dnabert2_Full_double_0 ' \
  '--per_device_train_batch_size 16 ' \
  '--per_device_eval_batch_size 16 ' \
  '--learning_rate 3e-5 ' \
  '--num_train_epochs 5 ' \
  '--svd_reserve_size 0 ' \
  '--evaluation_strategy epoch ' \
  '--overwrite_output_dir '

# Step3. Attack our proposed model and get the evaluation results
command9 = 'python attack_classification_triple_v3.py ' \
  '--synonym_num 30 ' \
  '--ratio 0.3 ' \
  '--syn_num 20 ' \
  '--most_freq_num 10 ' \
  '--simplify_version v2 ' \
  '--simplify2_version random_freq_v1 ' \
  '--do_simplify ' \
  '--simp_sim_threshold 3000 ' \
  '--dataset_path data/MR/original_data/test ' \
  '--target_model bert ' \
  '--target_model_path experiments/MR/4times_adv_double_0-7 ' \
  '--counter_fitting_cos_sim_path cos_sim_counter_fitting.npy ' \
  '--USE_cache_path TextFooler/USE ' \
  '--output_dir attack/MR/4times_adv_double_0-7/ ' \
  '--data_size 1000 '



# os.system(command1)

# os.system(command3)
# os.system(command4)
# os.system(command5)
# os.system(command6)
# os.system(command7)
os.system(command8)

# os.system(command9)
