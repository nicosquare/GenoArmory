import os
import re
import csv

def extract_adv_sequences(input_file):
    """ 读取 adversaries.txt 并提取对抗序列 """
    adv_sequences = []
    if not os.path.exists(input_file):
        return adv_sequences  # 如果文件不存在，返回空列表

    with open(input_file, 'r', encoding='utf-8') as file:
        data = file.readlines()
    
    for line in data:
        match = re.search(r'adv sent \((\d+)\):\s*([A-Z ]+)', line)
        if match:
            label = 1 - int(match.group(1))  # 计算 label
            sequence = match.group(2).replace(" ", "")  # 去除空格
            adv_sequences.append([sequence, label])
    
    return adv_sequences

def process_folders(input_folder):
    """ 遍历 input_folder 下的所有子文件夹，每个子文件夹生成一个 CSV 文件 """
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        
        # 只处理目录
        if not os.path.isdir(folder_path):
            continue

        input_file = os.path.join(folder_path, "adversaries.txt")
        output_file = os.path.join(folder_path, "invert.csv")

        adv_sequences = extract_adv_sequences(input_file)
        
        if adv_sequences:
            # 生成 CSV 文件
            with open(output_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["sequence", "label"])
                writer.writerows(adv_sequences)

            print(f"Processed: {input_file} → {output_file}")
        else:
            print(f"Skipped (no valid data): {input_file}")

# 设置输入目录
input_dir = "/projects/p32013/DNABERT-meta/TextFooler/output"

# 运行处理
process_folders(input_dir)
