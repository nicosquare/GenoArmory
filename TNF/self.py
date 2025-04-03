import os
import pandas as pd

def majority_vote(row):
    votes = [row['bertattack'], row['xgb'], row['lgb']]
    return 1 if sum(votes) > len(votes) // 2 else 0

def process_files(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'lgb.csv':
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)
                
                df['label'] = df.apply(majority_vote, axis=1)
                result_df = df[['sequence', 'label']]
                
                output_file_path = os.path.join(subdir, 'self.csv')
                result_df.to_csv(output_file_path, index=False)
                print(f"file save at {output_file_path}")

root_directory = '/projects/p32013/DNABERT-meta/TNF/results'

process_files(root_directory)
